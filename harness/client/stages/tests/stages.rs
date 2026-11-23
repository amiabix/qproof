//! Embedding and head-shard relations against independent native references,
//! with planted ties and mutated records. Run with `--release` (the tied matrix
//! is full size: 151,936 x 2,560 BF16).
use pci_receipts::{DIM, VOCAB, activation_root, consumed_root};
use pci_stages::{ROW_BYTES, embed, embed_weights, head, head_weights, rows};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A finite BF16 of magnitude in [2^-10, 2^-4).
    fn small_bf16(&mut self) -> u16 {
        let r = self.next();
        let sign = ((r >> 20) & 1) as u16;
        let exp = 117 + (r % 6) as u16;
        sign << 15 | exp << 7 | ((r >> 8) & 0x7f) as u16
    }
}

const M: [u8; 32] = [1; 32];
const S: [u8; 32] = [2; 32];
const Q: [u8; 32] = [3; 32];

/// Planted winners: rows 7 and 9 identical (same shard), and rows
/// 65535/65536 identical across the 2^16 boundary; all four dominate.
const PLANT_IN: [u32; 2] = [7, 9];
const PLANT_ACROSS: [u32; 2] = [65535, 65536];

struct Fixture {
    tied: Vec<u8>,
    tree: rows::Tree,
    norm: Vec<u8>,
    a36: Vec<f32>,
    t: u32,
    r: u32,
}

fn xn_of(a36: &[f32], norm: &[u8], t: u32, r: u32) -> Vec<Vec<f32>> {
    // Final RMSNorm of the last r positions through qwen_common's HEAD path.
    let h = head::header(t, r, 1);
    let mut seg: Vec<u8> = h.to_bytes().to_vec();
    seg.extend(a36.iter().flat_map(|v| v.to_bits().to_le_bytes()));
    for p in norm.chunks_exact(2) {
        seg.extend((u32::from(u16::from_le_bytes([p[0], p[1]])) << 16).to_le_bytes());
    }
    let words: Vec<u32> = seg.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 4 * words.len()) };
    let out = std::cell::RefCell::new(Vec::new());
    let mut obs = |_: usize, v: &[f32]| out.borrow_mut().push(v.to_vec());
    let mut claim = |_: usize, o: &mut [f32]| o.fill(0.0);
    qwen_common::run_with(bytes, qwen_common::dot_soft, qwen_common::qdot_soft, &mut obs, Some(&mut claim));
    out.into_inner()
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let mut rng = Rng(0x243f6a8885a308d3);
        let (t, r) = (3u32, 2u32);
        let a36: Vec<f32> = (0..t * DIM)
            .map(|_| f32::from_bits(u32::from(rng.small_bf16()) << 16) * 64.0)
            .collect();
        let norm: Vec<u8> = (0..DIM)
            .flat_map(|_| (0x3f80 | (rng.next() & 0x7f) as u16).to_le_bytes())
            .collect();
        let mut tied = vec![0u8; VOCAB as usize * ROW_BYTES];
        for c in tied.chunks_exact_mut(2) {
            c.copy_from_slice(&rng.small_bf16().to_le_bytes());
        }
        // Plant rows aligned with the normalized input of the first prediction
        // so they win it: weight sign(xn_j) * 1.0 for each column.
        let xn = xn_of(&a36, &norm, t, r);
        let plant: Vec<u8> = xn[0]
            .iter()
            .flat_map(|v| (if *v < 0.0 { 0xbf80u16 } else { 0x3f80 }).to_le_bytes())
            .collect();
        for id in PLANT_IN.iter().chain(&PLANT_ACROSS) {
            tied[*id as usize * ROW_BYTES..][..ROW_BYTES].copy_from_slice(&plant);
        }
        let tree = rows::Tree::build(sha, &tied);
        Fixture { tied, tree, norm, a36, t, r }
    })
}

/// Native full-vocabulary greedy reference: every logit, argmax lowest ID.
fn reference(f: &Fixture) -> (Vec<u32>, Vec<Vec<f32>>) {
    let xn = xn_of(&f.a36, &f.norm, f.t, f.r);
    let mut ids = vec![];
    let mut all = vec![];
    for x in &xn {
        let mut logits = vec![0.0f32; VOCAB as usize];
        pci_layer::project(x, &f.tied, &mut logits);
        ids.push(qwen_common::argmax(&logits) as u32);
        all.push(logits);
    }
    (ids, all)
}

#[test]
fn row_tree_paths_climb_to_root() {
    let f = fixture();
    let root = f.tree.root(sha);
    for id in [0, 1, 7, 65535, 65536, VOCAB - 1] {
        let row = &f.tied[id as usize * ROW_BYTES..][..ROW_BYTES];
        let path = f.tree.path(0, id);
        assert_eq!(rows::climb(sha, rows::leaf(sha, id, row), id, 0, &path), root);
        // The same row and path claimed for a neighbouring ID does not climb.
        assert_ne!(rows::climb(sha, rows::leaf(sha, id ^ 1, row), id ^ 1, 0, &path), root);
    }
}

#[test]
fn embed_matches_reference_and_rejects_mutations() {
    let f = fixture();
    let root = f.tree.root(sha);
    let ids = [12522u32, 7, 12522, VOCAB - 1];
    let rec = embed::encode(M, S, Q, &ids, &f.tied, &f.tree);
    let out = embed::execute(&rec, sha).unwrap();
    let a0: Vec<f32> = ids
        .iter()
        .flat_map(|&id| pci_layer::widen(&f.tied[id as usize * ROW_BYTES..][..ROW_BYTES]).collect::<Vec<_>>())
        .collect();
    assert_eq!(out.receipt.input, consumed_root(&ids[..2], &ids[2..]));
    assert_eq!(out.receipt.output, activation_root(M, Q, 0, 4, &a0).unwrap());
    assert_eq!(out.receipt.weights, embed_weights(sha, &root));
    assert_eq!(out.digest, out.receipt.digest(M, S, Q));
    assert_eq!(out.output.len(), a0.len() * 4);

    // A modified row byte at one position: openings disagree.
    let mut bad = rec.clone();
    bad[embed::FIXED + 4 * 4 + 100] ^= 1;
    assert_eq!(embed::execute(&bad, sha).err(), Some("openings disagree on the row root"));
    // With a single position, a modified row yields a different (unpinned) weights root.
    let one = embed::encode(M, S, Q, &[5], &f.tied, &f.tree);
    let mut bad = one.clone();
    bad[embed::FIXED + 4 + 3] ^= 0x10;
    assert_ne!(embed::execute(&bad, sha).unwrap().receipt.weights, embed_weights(sha, &root));
    // A different ID with the original row/path: no longer opens under the root.
    let mut bad = one.clone();
    bad[embed::FIXED..embed::FIXED + 4].copy_from_slice(&4u32.to_le_bytes());
    assert_ne!(embed::execute(&bad, sha).unwrap().receipt.weights, embed_weights(sha, &root));
    // Out-of-vocabulary ID, trailing bytes, truncation, wrong magic, zero positions.
    let mut bad = one.clone();
    bad[embed::FIXED..embed::FIXED + 4].copy_from_slice(&VOCAB.to_le_bytes());
    assert_eq!(embed::execute(&bad, sha).err(), Some("token id out of range"));
    let mut bad = one.clone();
    bad.push(0);
    assert_eq!(embed::execute(&bad, sha).err(), Some("record length"));
    assert_eq!(embed::execute(&one[..one.len() - 1], sha).err(), Some("record length"));
    let mut bad = one.clone();
    bad[0] = b'X';
    assert_eq!(embed::execute(&bad, sha).err(), Some("magic"));
    let mut bad = one.clone();
    bad[104..108].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(embed::execute(&bad, sha).err(), Some("positions"));
    // Request binding: a different request ID changes output root and digest.
    let other = embed::execute(&embed::encode(M, S, [9; 32], &[5], &f.tied, &f.tree), sha).unwrap();
    let base = embed::execute(&one, sha).unwrap();
    assert_ne!(other.receipt.output, base.receipt.output);
    assert_ne!(other.digest, base.digest);
}

fn shards(f: &Fixture, log2: u32) -> Vec<head::Outcome> {
    (0..head::shard_count(log2))
        .map(|k| {
            let rec = head::encode(M, S, Q, log2, k, f.r, &f.a36, &f.norm, &f.tied, &f.tree);
            head::execute(&rec, sha).unwrap()
        })
        .collect()
}

#[test]
fn sharded_head_equals_full_vocabulary_argmax_with_ties() {
    let f = fixture();
    let (want, logits) = reference(f);
    // The planted rows tie at the top of prediction 0; the lowest planted ID wins.
    let top = logits[0][PLANT_IN[0] as usize];
    for id in PLANT_IN.iter().chain(&PLANT_ACROSS) {
        assert_eq!(logits[0][*id as usize].to_bits(), top.to_bits());
    }
    assert_eq!(want[0], PLANT_IN[0]);
    let root = f.tree.root(sha);
    let weights = head_weights(sha, &sha(&f.norm), &root);
    let a36_root = activation_root(M, Q, 36, f.t, &f.a36).unwrap();
    for log2 in [14, 16, 18] {
        let outs = shards(f, log2);
        let receipts: Vec<_> = outs.iter().map(|o| o.receipt.clone()).collect();
        for (k, o) in outs.iter().enumerate() {
            assert_eq!(o.receipt.weights, weights);
            assert_eq!(o.receipt.input, a36_root);
            assert_eq!((o.receipt.first_prediction, o.receipt.predictions), (f.t - f.r, f.r));
            assert_eq!(o.digest, o.receipt.digest(sha, M, S, Q));
            // Shard logits are bit-identical to the full reference.
            let first = (k as u32) << log2;
            let n = head::shard_rows(log2, k as u32) as usize;
            for (j, l) in o.logits.chunks_exact(n).enumerate() {
                let r = &logits[j][first as usize..first as usize + n];
                assert!(l.iter().zip(r).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
        assert_eq!(head::combine(&receipts).unwrap(), want, "log2 {log2}");
        if log2 == 16 {
            // The cross-boundary tie: shard 0 holds 7 (and 65535), shard 1 holds 65536.
            assert_eq!(receipts[0].best[0].0, 7);
            assert_eq!(receipts[1].best[0].0, 65536);
        }
    }
}

#[test]
fn combine_ties_and_nonfinite() {
    let mk = |shard: u32, best: Vec<(u32, u32)>| head::ShardReceipt {
        shard_log2: 4,
        shard,
        positions: 1,
        first_prediction: 0,
        predictions: best.len() as u32,
        input: [0; 32],
        weights: [0; 32],
        best,
    };
    let one = 1.0f32.to_bits();
    // Equal values: the earlier shard (lower ID) wins.
    let r = head::combine(&[mk(0, vec![(3, one)]), mk(1, vec![(17, one)])]).unwrap();
    assert_eq!(r, vec![3]);
    // +0 and -0 tie.
    let r = head::combine(&[mk(0, vec![(3, (-0.0f32).to_bits())]), mk(1, vec![(17, 0)])]).unwrap();
    assert_eq!(r, vec![3]);
    // A strictly larger later value wins.
    let r = head::combine(&[mk(0, vec![(3, one)]), mk(1, vec![(17, 2.0f32.to_bits())])]).unwrap();
    assert_eq!(r, vec![17]);
    // NaN/inf anywhere rejects.
    assert!(head::combine(&[mk(0, vec![(3, f32::NAN.to_bits())]), mk(1, vec![(17, one)])]).is_err());
    assert!(head::combine(&[mk(0, vec![(3, one)]), mk(1, vec![(17, f32::INFINITY.to_bits())])]).is_err());
    assert!(head::combine(&[mk(0, vec![(3, one)]), mk(1, vec![])]).is_err());
}

#[test]
fn head_shard_rejects_mutations() {
    let f = fixture();
    let root = f.tree.root(sha);
    let weights = head_weights(sha, &sha(&f.norm), &root);
    let log2 = 16;
    let rec = head::encode(M, S, Q, log2, 1, f.r, &f.a36, &f.norm, &f.tied, &f.tree);
    let ok = head::execute(&rec, sha).unwrap();
    assert_eq!(ok.receipt.weights, weights);
    let act = (f.t * DIM * 4) as usize;
    let rows_at = head::FIXED + act + ROW_BYTES;
    // A modified weight byte, a modified norm byte: different weights root.
    for at in [rows_at + 5, head::FIXED + act + 3] {
        let mut bad = rec.clone();
        bad[at] ^= 0x01;
        assert_ne!(head::execute(&bad, sha).unwrap().receipt.weights, weights);
    }
    // Shard index swapped while keeping rows and path: does not open under the root.
    let mut bad = rec.clone();
    bad[108..112].copy_from_slice(&0u32.to_le_bytes());
    assert_ne!(head::execute(&bad, sha).unwrap().receipt.weights, weights);
    // A36 modified: different input root.
    let mut bad = rec.clone();
    bad[head::FIXED + 9] ^= 1;
    assert_ne!(head::execute(&bad, sha).unwrap().receipt.input, ok.receipt.input);
    // R > T, R = 0, shard out of range, shard size out of range, trailing byte.
    for (at, v, e) in [
        (116, f.t + 1, "positions"),
        (116, 0, "positions"),
        (108, head::shard_count(log2), "shard"),
        (104, 3, "shard"),
        (104, 19, "shard"),
    ] {
        let mut bad = rec.clone();
        bad[at..at + 4].copy_from_slice(&v.to_le_bytes());
        assert_eq!(head::execute(&bad, sha).err(), Some(e));
    }
    let mut bad = rec.clone();
    bad.push(0);
    assert_eq!(head::execute(&bad, sha).err(), Some("record length"));
    // Nonfinite A36 value.
    let mut bad = rec.clone();
    bad[head::FIXED..head::FIXED + 4].copy_from_slice(&f32::INFINITY.to_bits().to_le_bytes());
    assert_eq!(head::execute(&bad, sha).err(), Some("nonfinite input activation"));
    // A shorter window (fewer predictions) is a different, distinguishable receipt.
    let mut fewer = rec.clone();
    fewer[116..120].copy_from_slice(&1u32.to_le_bytes());
    let fewer = head::execute(&fewer, sha).unwrap();
    assert_eq!(fewer.receipt.first_prediction, f.t - 1);
    assert_ne!(fewer.digest, ok.digest);
    // Last, partial shard: VOCAB is not a multiple of 2^16.
    let last = head::shard_count(log2) - 1;
    let rec = head::encode(M, S, Q, log2, last, f.r, &f.a36, &f.norm, &f.tied, &f.tree);
    assert_eq!(head::execute(&rec, sha).unwrap().receipt.weights, weights);
}

#[test]
fn batched_dotf8_kernel_is_a_bit_identical_head_kernel() {
    let f = fixture();
    let mut k = pci_dotf8::Kernel::new(pci_dotf8::SoftDotF8);
    let mut dotf8 = |x: &[u32], m: &[u8], out: &mut [f32]| {
        let p = pci_dotf8::Prepared::new(x, DIM as usize);
        k.project(&p, m, out);
    };
    for (log2, shard) in [(9, 0), (9, 128), (12, 37), (16, 2)] {
        let rec = head::encode(M, S, Q, log2, shard, f.r, &f.a36, &f.norm, &f.tied, &f.tree);
        let a = head::execute(&rec, sha).unwrap();
        let b = head::execute_with(&rec, sha, &mut dotf8).unwrap();
        assert_eq!(a.digest, b.digest, "shard {log2}/{shard}");
        assert!(a.logits.iter().zip(&b.logits).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}
