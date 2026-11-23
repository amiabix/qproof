//! The batched layer equals the per-position reference bit for bit, and rejects
//! exactly what the reference rejects.
use pci_layer::*;
use pci_layer_bench::{batched, per_position, sha};
use pci_receipts::Hash;
use qwen_common::{BATCH_GROUPS, Header, LAYER, dot_soft, qdot_soft, run, run_layer_batched};

struct Rng(u64);
impl Rng {
    fn f(&mut self, scale: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
    }
}

/// `run_layer_batched` with claimed products against `run` with the same f32 matrices.
#[test]
fn batched_layer_equals_run_on_random_models() {
    for (seed, n) in [(1u64, 1usize), (2, 2), (3, 5), (4, 9)] {
        let mut r = Rng(seed * 0x9e37_79b9_7f4a_7c15);
        let h = Header {
            kind: LAYER,
            dim: 64,
            hidden: 96,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            vocab: 0,
            n_tok: n,
            rope_theta: 1e6,
            eps: 1e-6,
            flags: 0,
            n_out: 0,
        };
        let (dim, qd, kvd, hid, hd) = (64, 64, 32, 96, 16);
        let mut v = |k: usize, s: f32| -> Vec<f32> { (0..k).map(|_| r.f(s)).collect() };
        let x = v(n * dim, 4.0);
        let input_ln = v(dim, 2.0);
        let mats: Vec<(Vec<f32>, usize, usize)> = [(qd, dim), (kvd, dim), (kvd, dim), (dim, qd), (hid, dim), (hid, dim), (dim, hid)]
            .iter()
            .map(|&(rows, cols)| (v(rows * cols, 0.5), rows, cols))
            .collect();
        let (q_norm, k_norm, post_ln) = (v(hd, 2.0), v(hd, 2.0), v(dim, 2.0));
        let f = |s: &[f32]| -> Vec<u8> { s.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let mut own = h.to_bytes().to_vec();
        for part in [&x, &input_ln, &mats[0].0, &mats[1].0, &mats[2].0, &mats[3].0, &q_norm, &k_norm, &post_ln, &mats[4].0, &mats[5].0, &mats[6].0] {
            own.extend(f(part));
        }
        let mut claimed = h.to_bytes().to_vec();
        for part in [&x, &input_ln, &q_norm, &k_norm, &post_ln] {
            claimed.extend(f(part));
        }
        let aligned = |b: Vec<u8>| -> Vec<u64> {
            b.chunks(8).map(|c| { let mut w = [0u8; 8]; w[..c.len()].copy_from_slice(c); u64::from_le_bytes(w) }).collect()
        };
        let (own_w, claimed_w) = (aligned(own.clone()), aligned(claimed.clone()));
        let bytes = |w: &[u64], len: usize| -> Vec<u8> { w.iter().flat_map(|v| v.to_le_bytes()).take(len).collect() };
        let want = run(&bytes(&own_w, own.len()), dot_soft, qdot_soft);
        let mut calls = vec![];
        let got = run_layer_batched(&bytes(&claimed_w, claimed.len()), dot_soft, &mut |g, xs, outs| {
            calls.push(g);
            for (&m, out) in BATCH_GROUPS[g].iter().zip(outs.iter_mut()) {
                let (w, rows, cols) = &mats[m];
                for (xt, ot) in xs.chunks_exact(*cols).zip(out.chunks_exact_mut(*rows)) {
                    for (o, row) in ot.iter_mut().zip(w.chunks_exact(*cols)) {
                        let mut acc = 0.0f32;
                        for (a, b) in row.iter().zip(xt) {
                            acc += a * b;
                        }
                        *o = acc;
                    }
                }
            }
        });
        assert_eq!(calls, [0, 1, 2, 3]);
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "seed {seed} n {n} value {i}");
        }
        // Non-vacuity: a different input moves the output.
        assert_ne!(f(&want), f(&x));
    }
}

fn synthetic(seed: u64) -> Vec<Vec<u8>> {
    let mut s = seed;
    TENSOR_LEN
        .iter()
        .enumerate()
        .map(|(slot, &n)| {
            let norm = [0, 4, 5, 8].contains(&slot);
            (0..n)
                .flat_map(|i| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let mant = (s & 0x7f) as u16;
                    let sign = ((s >> 8) & 1) as u16;
                    let exp: u16 = if norm { 127 } else { 117 + ((s >> 9) % 4) as u16 };
                    // A few exact zeros and subnormals in the matrices: fallback coverage.
                    let w = if !norm && i % 997 == 0 { sign << 15 | mant } else { (sign << 15) | (exp << 7) | mant };
                    w.to_le_bytes()
                })
                .collect()
        })
        .collect()
}

fn x(t: usize) -> Vec<f32> {
    (0..t * 2560).map(|i| ((i * 7919) % 2001) as f32 / 1000.0 - 1.0).collect()
}

#[test]
fn full_size_batched_and_per_position_equal_reference_and_reject_alike() {
    let t = 3;
    let tensors = synthetic(0x1234_5678_9abc_def1);
    let record = encode([1; 32], [2; 32], [3; 32], 4, &x(t), &tensors);
    let reference = execute(&record, sha).expect("honest record");
    for (name, o) in [("batched", batched(&record)), ("per-position", per_position(&record))] {
        let o = o.expect(name);
        assert!(o.receipt == reference.receipt, "{name} receipt");
        assert_eq!(digest_of(&record, &o.receipt), digest_of(&record, &reference.receipt));
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&o.output), bits(&reference.output), "{name} output bits");
    }
    // The reference batched kernel through execute_batched as well.
    let r = execute_batched(&record, sha, &mut project_batch_reference).unwrap();
    assert!(r.receipt == reference.receipt);

    // Mutations: same verdict (and same receipt when accepted) as the reference. Each
    // mutated record is built, checked and dropped in turn (they are 202 MB each).
    let w0 = FIXED + t * 2560 * 4;
    let mut cases: Vec<(&str, Box<dyn Fn(&mut Vec<u8>)>)> = vec![
        ("magic", Box::new(|r| r[0] ^= 1)),
        ("model", Box::new(|r| r[8] ^= 1)),
        ("request", Box::new(|r| r[72] ^= 0x80)),
        ("index", Box::new(|r| r[104] ^= 1)),
        ("index out of range", Box::new(|r| r[107] ^= 0x80)),
        ("positions", Box::new(|r| r[108] ^= 1)),
        ("profile eps", Box::new(|r| r[112 + 36] ^= 1)),
        ("profile flags", Box::new(|r| r[112 + 40] ^= 4)),
        ("activation bit", Box::new(|r| r[FIXED + 5] ^= 0x10)),
        ("activation to inf", Box::new(|r| r[FIXED..FIXED + 4].copy_from_slice(&f32::INFINITY.to_le_bytes()))),
        ("first weight byte", Box::new(move |r| r[w0] ^= 1)),
        ("last weight byte", Box::new(|r| *r.last_mut().unwrap() ^= 1)),
        ("trailing byte", Box::new(|r| r.push(0))),
        ("short", Box::new(|r| {
            r.pop();
        })),
    ];
    for slot in 0..11 {
        let at = w0 + TENSOR_LEN[..slot].iter().map(|n| 2 * n).sum::<usize>() + 2 * (TENSOR_LEN[slot] / 2) + 1;
        cases.push(("tensor", Box::new(move |r| r[at] ^= 1)));
    }
    let digest = |o: &Result<Outcome, &'static str>| -> Result<Hash, &'static str> {
        o.as_ref().map(|o| o.receipt.digest([0; 32], [0; 32], [0; 32])).map_err(|e| *e)
    };
    let honest = digest(&Ok(reference));
    let mut rec = record.clone();
    for (name, mutate) in &cases {
        rec.clone_from(&record);
        mutate(&mut rec);
        let want = digest(&execute(&rec, sha));
        assert_eq!(digest(&batched(&rec)), want, "{name}");
        assert_ne!(want, honest, "{name} moved nothing");
    }
}
