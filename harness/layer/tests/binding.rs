//! Native runs of the exact guest function on full-size synthetic tensors.
//! The emulator runs of the same cases live in `pci-layer-host forge`.
use pci_layer::*;
use pci_receipts::*;
use sha2::{Digest, Sha256};

fn sha(b: &[u8]) -> Hash {
    Sha256::digest(b).into()
}

fn synthetic(seed: u64) -> Vec<Vec<u8>> {
    let mut s = seed;
    TENSOR_LEN
        .iter()
        .enumerate()
        .map(|(slot, &n)| {
            let norm = [0, 4, 5, 8].contains(&slot);
            (0..n)
                .flat_map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    // Small finite BF16 weights; norms near 1.
                    let mant = (s & 0x7f) as u16;
                    let sign = ((s >> 8) & 1) as u16;
                    let exp: u16 = if norm {
                        127
                    } else {
                        117 + ((s >> 9) % 4) as u16
                    };
                    ((sign << 15) | (exp << 7) | mant).to_le_bytes()
                })
                .collect()
        })
        .collect()
}

fn x(t: usize) -> Vec<f32> {
    (0..t * 2560)
        .map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0)
        .collect()
}

struct Fixture {
    tensors: Vec<Vec<u8>>,
    record: Vec<u8>,
    receipt: Receipt,
    output: Vec<f32>,
}

fn fixture(t: usize) -> Fixture {
    let tensors = synthetic(0x1234_5678_9abc_def1);
    let record = encode([1; 32], [2; 32], [3; 32], 4, &x(t), &tensors);
    let o = execute(&record, sha).expect("honest record");
    Fixture {
        tensors,
        record,
        receipt: o.receipt,
        output: o.output,
    }
}

fn flipped(r: &[u8], at: usize, mask: u8) -> Vec<u8> {
    let mut r = r.to_vec();
    r[at] ^= mask;
    r
}

fn tensor_offset(t: usize, slot: usize) -> usize {
    FIXED + t * 2560 * 4 + TENSOR_LEN[..slot].iter().map(|n| 2 * n).sum::<usize>()
}

#[test]
fn honest_receipt_binds_enrollment_digests_and_both_activations() {
    let t = 3;
    let f = fixture(t);
    let digests: Vec<Hash> = f.tensors.iter().map(|b| sha(b)).collect();
    assert_eq!(f.receipt.weights, weights_root(Role::Layer(4), &digests));
    assert_eq!(
        f.receipt.input,
        activation_root([1; 32], [3; 32], 4, t as u32, &x(t)).unwrap()
    );
    assert_eq!(
        f.receipt.output,
        activation_root([1; 32], [3; 32], 5, t as u32, &f.output).unwrap()
    );
    assert_eq!((f.receipt.positions, f.receipt.role), (3, Role::Layer(4)));
    // Causality: the first two positions do not depend on the third.
    let prefix = encode([1; 32], [2; 32], [3; 32], 4, &x(t)[..2 * 2560], &f.tensors);
    let p = execute(&prefix, sha).unwrap();
    assert_eq!(
        p.output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        f.output[..2 * 2560]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn every_tensor_byte_class_moves_only_the_weights_root() {
    let t = 1;
    let f = fixture(t);
    for slot in 0..11 {
        for at in [0, 2 * TENSOR_LEN[slot] - 1] {
            let r = flipped(&f.record, tensor_offset(t, slot) + at, 0x01);
            let g = execute(&r, sha).expect("still a well-formed record");
            assert_ne!(g.receipt.weights, f.receipt.weights, "slot {slot}");
            assert_eq!(g.receipt.input, f.receipt.input);
            assert_ne!(digest_of(&r, &g.receipt), digest_of(&f.record, &f.receipt));
        }
    }
}

#[test]
fn request_model_and_suite_change_the_committed_digest() {
    let f = fixture(1);
    for at in [8, 40, 72] {
        let r = flipped(&f.record, at + 31, 0x80);
        let g = execute(&r, sha).unwrap();
        assert_ne!(digest_of(&r, &g.receipt), digest_of(&f.record, &f.receipt));
        if at != 40 {
            // Model and request are inside both activation roots.
            assert_ne!(g.receipt.input, f.receipt.input);
            assert_ne!(g.receipt.output, f.receipt.output);
        }
    }
}

#[test]
fn profile_flags_and_shape_fields_are_rejected() {
    let t = 2;
    let f = fixture(t);
    // Every bit of the 64-byte profile header, the index and positions.
    for byte in 104..FIXED {
        for bit in 0..8 {
            let r = flipped(&f.record, byte, 1 << bit);
            let e = execute(&r, sha).err();
            if byte == 104 && bit < 5 && (4 ^ (1u32 << bit)) < 36 {
                // Another in-range layer index is well formed; it binds a different role.
                let g = execute(&r, sha).unwrap();
                assert_ne!(g.receipt.role, f.receipt.role);
                assert_ne!(g.receipt.weights, f.receipt.weights);
                continue;
            }
            assert!(e.is_some(), "byte {byte} bit {bit} accepted");
        }
    }
    let mut r = f.record.clone();
    r.push(0);
    assert_eq!(execute(&r, sha).err(), Some("record length"));
    r.truncate(f.record.len() - 1);
    assert_eq!(execute(&r, sha).err(), Some("record length"));
    assert_eq!(execute(&flipped(&f.record, 0, 1), sha).err(), Some("magic"));
}

#[test]
fn activation_change_moves_input_root_and_nonfinite_is_rejected() {
    let t = 2;
    let f = fixture(t);
    let r = flipped(&f.record, FIXED + 4 * 2600, 0x01);
    let g = execute(&r, sha).unwrap();
    assert_ne!(g.receipt.input, f.receipt.input);
    assert_eq!(g.receipt.weights, f.receipt.weights);
    let mut nan = f.record.clone();
    nan[FIXED..FIXED + 4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    assert_eq!(execute(&nan, sha).err(), Some("nonfinite input activation"));
}
