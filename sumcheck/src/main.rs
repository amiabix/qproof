//! qwen-sumcheck MODEL_DIR TOKENS
//!
//! Proves the 7 matrix-vector products of Qwen3-4B layer 0 for the given tokens
//! with sumcheck against a Ligero commitment to each weight matrix, over the
//! BN254 scalar field, with weights and activations encoded losslessly as
//! integers. The weights are committed; x and y are public.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use ark_serialize::CanonicalSerialize;
use qwen_common::{HEADER_BYTES, Header, LAYER, MATRICES, dot_soft, layer_matmul_inputs};
use qwen_sumcheck::{
    Com, Dims, MatrixProof, Pcs, Poly, Proof, Transcript, absorb_public, challenge, combine,
    eq_table, keys, verify,
};
use rayon::prelude::*;
use serde_json::Value;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn pow2(k: u32) -> Fr {
    Fr::from(2u64).pow([k as u64])
}

/// A bf16 weight as `(±mantissa, exponent)`, value `m · 2^(e - 134)`.
fn bf16_parts(b: u16) -> (i64, i32) {
    let (e, m) = ((b >> 7 & 0xff) as i32, (b & 0x7f) as i64);
    assert!(e != 0xff, "non-finite weight");
    let (m, e) = if e == 0 { (m, 1) } else { (m + 128, e) };
    (if b >> 15 == 1 { -m } else { m }, e)
}

/// An f32 as `(±mantissa, exponent)`, value `m · 2^(e - 150)`.
fn f32_parts(v: f32) -> (i64, i32) {
    let b = v.to_bits();
    let (e, m) = ((b >> 23 & 0xff) as i32, (b & 0x7f_ffff) as i64);
    assert!(e != 0xff, "non-finite activation");
    let (m, e) = if e == 0 { (m, 1) } else { (m + (1 << 23), e) };
    (if b >> 31 == 1 { -m } else { m }, e)
}

fn signed(m: i64, shift: u32) -> Fr {
    let f = Fr::from(m.unsigned_abs()) * pow2(shift);
    if m < 0 { -f } else { f }
}

/// A field element as a signed value in f64 (|v| < p/2).
fn to_f64(f: Fr) -> f64 {
    let (neg, mag) = if f.into_bigint().num_bits() >= Fr::MODULUS_BIT_SIZE - 1 {
        (true, -f)
    } else {
        (false, f)
    };
    let v = mag
        .into_bigint()
        .to_bytes_le()
        .iter()
        .rev()
        .fold(0f64, |a, &b| a * 256.0 + b as f64);
    if neg { -v } else { v }
}

struct Matrix {
    rows: usize,
    cols: usize,
    /// Row-major, padded to `2^row_vars × 2^col_vars`: index `i·2^col_vars + j`.
    w: Vec<Fr>,
    row_vars: usize,
    col_vars: usize,
    max_bits: u32,
}

impl Matrix {
    /// Each row scaled by its smallest exponent, so its integers are exact.
    fn encode(rows: usize, cols: usize, raw: &[u16]) -> (Self, Vec<i32>) {
        let row_vars = rows.next_power_of_two().ilog2() as usize;
        let col_vars = cols.next_power_of_two().ilog2() as usize;
        let mut w = vec![Fr::zero(); 1 << (row_vars + col_vars)];
        let row_exp: Vec<i32> = raw
            .chunks(cols)
            .map(|r| {
                r.iter()
                    .map(|&b| bf16_parts(b))
                    .filter(|p| p.0 != 0)
                    .map(|p| p.1)
                    .min()
                    .unwrap_or(0)
            })
            .collect();
        let max_bits = w
            .par_chunks_mut(1 << col_vars)
            .zip(raw.par_chunks(cols))
            .zip(&row_exp)
            .map(|((out, row), &e0)| {
                let mut bits = 0;
                for (o, &b) in out.iter_mut().zip(row) {
                    let (m, e) = bf16_parts(b);
                    if m != 0 {
                        let s = (e - e0) as u32;
                        bits = bits.max(64 - m.unsigned_abs().leading_zeros() + s);
                        *o = signed(m, s);
                    }
                }
                bits
            })
            .max()
            .unwrap();
        (
            Matrix {
                rows,
                cols,
                w,
                row_vars,
                col_vars,
                max_bits,
            },
            row_exp,
        )
    }
}

/// An activation vector scaled by its smallest exponent, padded to `2^vars`.
fn encode_x(x: &[f32], vars: usize) -> (Vec<Fr>, i32, u32) {
    let e0 = x
        .iter()
        .map(|&v| f32_parts(v))
        .filter(|p| p.0 != 0)
        .map(|p| p.1)
        .min()
        .unwrap_or(0);
    let mut out = vec![Fr::zero(); 1 << vars];
    let mut bits = 0;
    for (o, &v) in out.iter_mut().zip(x) {
        let (m, e) = f32_parts(v);
        if m != 0 {
            let s = (e - e0) as u32;
            bits = bits.max(64 - m.unsigned_abs().leading_zeros() + s);
            *o = signed(m, s);
        }
    }
    (out, e0, bits)
}

/// Proves `sum_t rho^t y_t = W (sum_t rho^t x_t)` at a random row point: one
/// sumcheck and one opening of W for all the given tokens.
fn prove(
    m: &Matrix,
    ck: &<Pcs as PolynomialCommitment<Fr, Poly>>::CommitterKey,
    poly: &LabeledPolynomial<Fr, Poly>,
    com: &Com,
    state: &<Pcs as PolynomialCommitment<Fr, Poly>>::CommitmentState,
    xs: &[Vec<Fr>],
    ys: &[Vec<Fr>],
    t: &mut Transcript,
) -> Proof {
    absorb_public(t, com, xs, ys);
    let r: Vec<Fr> = (0..m.row_vars).map(|_| challenge(t)).collect();
    let rho = challenge(t);
    let eq_r = eq_table(&r);
    let cn = 1 << m.col_vars;
    // a_j = W~(r, j).
    let mut a: Vec<Fr> = (0..cn)
        .into_par_iter()
        .map(|j| (0..m.rows).map(|i| eq_r[i] * m.w[i * cn + j]).sum())
        .collect();
    let mut x = combine(xs, rho);
    let mut rounds = Vec::with_capacity(m.col_vars);
    let mut s = Vec::with_capacity(m.col_vars);
    // Variable k is bit k of j, so each round folds adjacent pairs.
    for _ in 0..m.col_vars {
        let half = a.len() / 2;
        let g = (0..half)
            .into_par_iter()
            .map(|i| {
                let (a0, a1, x0, x1) = (a[2 * i], a[2 * i + 1], x[2 * i], x[2 * i + 1]);
                [a0 * x0, a1 * x1, (a1 + a1 - a0) * (x1 + x1 - x0)]
            })
            .reduce(
                || [Fr::zero(); 3],
                |p, q| [p[0] + q[0], p[1] + q[1], p[2] + q[2]],
            );
        t.absorb(&g.to_vec());
        let c = challenge(t);
        a = (0..half)
            .map(|i| a[2 * i] + c * (a[2 * i + 1] - a[2 * i]))
            .collect();
        x = (0..half)
            .map(|i| x[2 * i] + c * (x[2 * i + 1] - x[2 * i]))
            .collect();
        rounds.push(g);
        s.push(c);
    }
    let point: Vec<Fr> = s.iter().chain(&r).copied().collect();
    let opening = Pcs::open(ck, [poly], [com], &point, t, [state], None).unwrap();
    Proof {
        rounds,
        value: a[0],
        opening,
    }
}

fn bf16_tensor(model: &Path, name: &str) -> Result<Vec<u16>> {
    let index: Value =
        serde_json::from_slice(&std::fs::read(model.join("model.safetensors.index.json"))?)?;
    let shard = index["weight_map"][name].as_str().ok_or("no tensor")?;
    let mut f = File::open(model.join(shard))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut header)?;
    let h: Value = serde_json::from_slice(&header)?;
    let o = h[name]["data_offsets"].as_array().unwrap();
    let (s, e) = (o[0].as_u64().unwrap(), o[1].as_u64().unwrap());
    f.seek(SeekFrom::Start(8 + header.len() as u64 + s))?;
    let mut raw = vec![0u8; (e - s) as usize];
    f.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(&args[1]);
    let tokens: Vec<usize> = args[2]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let cfg: Value = serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
    let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let h = Header {
        kind: LAYER,
        dim: n("hidden_size"),
        hidden: n("intermediate_size"),
        heads: n("num_attention_heads"),
        kv_heads: n("num_key_value_heads"),
        head_dim: n("head_dim"),
        vocab: n("vocab_size"),
        n_tok: tokens.len(),
        rope_theta: cfg["rope_theta"].as_f64().unwrap() as f32,
        eps: cfg["rms_norm_eps"].as_f64().unwrap() as f32,
        flags: 0,
        n_out: 1,
    };

    // Layer 0's input, as in the ZisK run, and its matmul inputs.
    let emb = bf16_tensor(&model, "model.embed_tokens.weight")?;
    let mut input = h.to_bytes().to_vec();
    assert_eq!(input.len(), HEADER_BYTES);
    for &t in &tokens {
        input.extend(
            emb[t * h.dim..][..h.dim]
                .iter()
                .flat_map(|&b| (u32::from(b) << 16).to_le_bytes()),
        );
    }
    drop(emb);
    let names = [
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ];
    let mut raw: HashMap<&str, Vec<u16>> = HashMap::new();
    for name in names {
        let t = bf16_tensor(&model, &format!("model.layers.0.{name}.weight"))?;
        input.extend(t.iter().flat_map(|&b| (u32::from(b) << 16).to_le_bytes()));
        raw.insert(name, t);
    }
    let inputs = layer_matmul_inputs(&input);
    drop(input);

    let shapes = [
        (h.q_dim(), h.dim),
        (h.kv_dim(), h.dim),
        (h.kv_dim(), h.dim),
        (h.dim, h.q_dim()),
        (h.hidden, h.dim),
        (h.hidden, h.dim),
        (h.dim, h.hidden),
    ];
    let (ck, vk) = keys();
    let out_path = args.get(3).cloned();
    let mut proofs = Vec::new();
    println!(
        "BN254 Fr, Ligero sec_param 128, rho_inv 2, Blake3; {} threads; {} tokens",
        rayon::current_num_threads(),
        tokens.len()
    );
    println!(
        "matrix      rows x cols  vars    w_bits x_bits sum_bits commit_s | single: prove_s verify_s bytes | batched: prove_s verify_s bytes | max_rel_err"
    );

    let mut tot = [0f64; 7];
    for (mi, &name) in MATRICES.iter().enumerate() {
        let key = names.iter().find(|n| n.ends_with(name)).unwrap();
        let (rows, cols) = shapes[mi];
        let (m, row_exp) = Matrix::encode(rows, cols, &raw[key]);
        let dims = Dims {
            rows: rows as u64,
            cols: cols as u64,
            row_vars: m.row_vars as u64,
            col_vars: m.col_vars as u64,
        };
        let w32: Vec<u64> = raw[key]
            .chunks(2)
            .map(|p| (u64::from(p[1]) << 48) | (u64::from(p[0]) << 16))
            .collect();

        let (mut xs, mut ys, mut x_bits, mut max_err) = (Vec::new(), Vec::new(), 0, 0f64);
        for (_, _, x) in inputs.iter().filter(|(k, _, _)| *k == mi) {
            let (xe, e0, bits) = encode_x(x, m.col_vars);
            x_bits = x_bits.max(bits);
            let cn = 1 << m.col_vars;
            let y: Vec<Fr> = (0..1 << m.row_vars)
                .into_par_iter()
                .map(|i| {
                    if i < rows {
                        (0..cols).map(|j| m.w[i * cn + j] * xe[j]).sum()
                    } else {
                        Fr::zero()
                    }
                })
                .collect();
            // The exact sum, scaled back, against the FP32 ordered result FloatDot proves.
            let xw: Vec<u64> = x
                .chunks(2)
                .map(|p| u64::from(p[0].to_bits()) | u64::from(p[1].to_bits()) << 32)
                .collect();
            for i in (0..rows).step_by(97) {
                let fp = dot_soft(&w32[i * cols / 2..][..cols / 2], &xw, cols) as f64;
                let exact = to_f64(y[i]) * 2f64.powi(row_exp[i] - 134 + e0 - 150);
                if fp != 0.0 {
                    max_err = max_err.max(((exact - fp) / fp).abs());
                }
            }
            xs.push(xe);
            ys.push(y);
        }
        let sum_bits = m.max_bits + x_bits + m.col_vars as u32;
        assert!(
            sum_bits < Fr::MODULUS_BIT_SIZE - 1,
            "{name}: sums may wrap ({sum_bits} bits)"
        );

        let t0 = Instant::now();
        let poly = LabeledPolynomial::new(
            name.to_string(),
            Poly::from_evaluations_vec(m.row_vars + m.col_vars, m.w.clone()),
            None,
            None,
        );
        let (coms, states) = Pcs::commit(&ck, [&poly], None).unwrap();
        let commit_s = t0.elapsed().as_secs_f64();

        let mut single = (0f64, 0f64, 0usize);
        for k in 0..xs.len() {
            let (x1, y1) = (&xs[k..k + 1], &ys[k..k + 1]);
            let t1 = Instant::now();
            let p = prove(
                &m,
                &ck,
                &poly,
                &coms[0],
                &states[0],
                x1,
                y1,
                &mut Transcript::new(&()),
            );
            single.0 += t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            assert!(
                verify(&dims, &vk, &coms[0], x1, y1, &p, &mut Transcript::new(&())),
                "{name} token {k}"
            );
            single.1 += t2.elapsed().as_secs_f64();
            single.2 += p.bytes();
        }
        let t1 = Instant::now();
        let p = prove(
            &m,
            &ck,
            &poly,
            &coms[0],
            &states[0],
            &xs,
            &ys,
            &mut Transcript::new(&()),
        );
        let bp = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        assert!(
            verify(
                &dims,
                &vk,
                &coms[0],
                &xs,
                &ys,
                &p,
                &mut Transcript::new(&())
            ),
            "{name} batched"
        );
        let bv = t2.elapsed().as_secs_f64();
        let mut bad = ys.clone();
        let commitment = coms[0].commitment().clone();
        bad[0][0] += Fr::one();
        assert!(
            !verify(
                &dims,
                &vk,
                &coms[0],
                &xs,
                &bad,
                &p,
                &mut Transcript::new(&())
            ),
            "{name}: forged y accepted"
        );

        println!(
            "{:10} {:>5} x {:<5} {:>2}+{:<2}  {:>6} {:>6} {:>8} {:>8.2} | {:>8.2} {:>8.2} {:>8} | {:>8.2} {:>8.2} {:>8} | {:.1e}",
            name,
            m.rows,
            m.cols,
            m.row_vars,
            m.col_vars,
            m.max_bits,
            x_bits,
            sum_bits,
            commit_s,
            single.0,
            single.1,
            single.2,
            bp,
            bv,
            p.bytes(),
            max_err
        );
        let bytes = p.bytes();
        proofs.push(MatrixProof {
            dims,
            commitment,
            xs,
            ys,
            proof: p,
        });
        for (t, v) in tot.iter_mut().zip([
            commit_s,
            single.0,
            single.1,
            single.2 as f64,
            bp,
            bv,
            bytes as f64,
        ]) {
            *t += v;
        }
    }
    println!(
        "TOTAL commit {:.2} s | single x{}: prove {:.2} s, verify {:.2} s, {} bytes | batched: prove {:.2} s, verify {:.2} s, {} bytes",
        tot[0],
        tokens.len(),
        tot[1],
        tot[2],
        tot[3],
        tot[4],
        tot[5],
        tot[6]
    );
    if let Some(path) = out_path {
        let mut b = Vec::new();
        proofs.serialize_uncompressed(&mut b)?;
        std::fs::write(&path, &b)?;
        println!("wrote {path}: {} bytes", b.len());
    }
    Ok(())
}
