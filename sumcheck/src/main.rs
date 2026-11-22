//! qwen-sumcheck MODEL_DIR TOKENS OUT_DIR [N_OUT [LAYERS]]
//!
//! Runs Qwen3 on its bf16 weights, every layer then the head, summing each
//! matrix product exactly and rounding once to f32. Proves each segment's
//! products with sumcheck against Ligero commitments to its weights.
//!
//! Writes `OUT_DIR/{layerNN,head}.stdin` (the `sumcheck-guest` input) and
//! `.expected` (the 96 bytes the guest must commit). With LAYERS, stops after
//! that many layers and skips the head.

use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::Zero;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use ark_serialize::CanonicalSerialize;
use qwen_common::{HEAD, Header, LAYER, argmax, dot_soft, qdot_soft, run_with};
use qwen_sumcheck::{
    Com, Dims, MatrixProof, Pcs, Poly, Proof, Transcript, absorb_public, challenge, combine,
    encode_x, eq_table, f32_parts, keys, sha256, signed, weights_id, y_to_f32,
};
use rayon::prelude::*;
use serde_json::Value;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Most rows per proof. Matrices are proven in power-of-two row chunks
/// (9,728 = 8,192 + 1,024 + 512) so no rows are padded and prover memory stays bounded.
const CHUNK_ROWS: usize = 8192;

/// A row chunk of one of a segment's matrices.
struct Chunk {
    matrix: usize,
    rows: usize,
    cols: usize,
    raw: Vec<u16>,
}

fn chunks(matrix: usize, cols: usize, raw: &[u16]) -> Vec<Chunk> {
    let (mut at, rows) = (0, raw.len() / cols);
    let mut out = Vec::new();
    while at < rows {
        let n = (1 << (rows - at).ilog2()).min(CHUNK_ROWS);
        out.push(Chunk {
            matrix,
            rows: n,
            cols,
            raw: raw[at * cols..(at + n) * cols].to_vec(),
        });
        at += n;
    }
    out
}

/// A bf16 weight as `(±mantissa, exponent)`, value `m · 2^(e - 134)`.
fn bf16_parts(b: u16) -> (i64, i32) {
    let (e, m) = ((b >> 7 & 0xff) as i32, (b & 0x7f) as i64);
    assert!(e != 0xff, "non-finite weight");
    let (m, e) = if e == 0 { (m, 1) } else { (m + 128, e) };
    (if b >> 15 == 1 { -m } else { m }, e)
}

fn min_exp(parts: impl Iterator<Item = (i64, i32)>) -> i32 {
    parts.filter(|p| p.0 != 0).map(|p| p.1).min().unwrap_or(0)
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
            .map(|r| min_exp(r.iter().map(|&b| bf16_parts(b))))
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
    let w = &poly.polynomial().evaluations;
    // a_j = W~(r, j).
    let mut a: Vec<Fr> = (0..cn)
        .into_par_iter()
        .map(|j| (0..m.rows).map(|i| eq_r[i] * w[i * cn + j]).sum())
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

/// A bf16 matrix as integers, each row scaled by its smallest exponent like
/// `Matrix::encode`, for computing the products exactly.
struct Ints {
    cols: usize,
    w: Vec<i64>,
    row_exp: Vec<i32>,
}

fn ints(cols: usize, raw: &[u16]) -> Ints {
    let mut w = vec![0i64; raw.len()];
    let row_exp: Vec<i32> = w
        .par_chunks_mut(cols)
        .zip(raw.par_chunks(cols))
        .map(|(out, row)| {
            let parts: Vec<(i64, i32)> = row.iter().map(|&b| bf16_parts(b)).collect();
            let e0 = min_exp(parts.iter().copied());
            for (o, &(m, e)) in out.iter_mut().zip(&parts) {
                if m != 0 {
                    assert!(e - e0 < 54, "weight row spans too many exponents");
                    *o = m << (e - e0);
                }
            }
            e0
        })
        .collect();
    Ints { cols, w, row_exp }
}

/// `x` as integers scaled by its smallest exponent, like `encode_x`.
fn x_ints(x: &[f32]) -> (Vec<i64>, i32) {
    let parts: Vec<(i64, i32)> = x.iter().map(|&v| f32_parts(v)).collect();
    let e0 = min_exp(parts.iter().copied());
    let v = parts
        .iter()
        .map(|&(m, e)| {
            if m == 0 {
                return 0;
            }
            assert!(e - e0 < 38, "activation spans too many exponents");
            m << (e - e0)
        })
        .collect();
    (v, e0)
}

fn fr(v: i128) -> Fr {
    let f = Fr::from(v.unsigned_abs());
    if v < 0 { -f } else { f }
}

/// Runs one segment with exact products of `mats`, proves each chunk, and
/// writes the guest's input and expected commit. Returns the segment's output.
fn segment(name: &str, input: Vec<u8>, mats: &[Chunk], out_dir: &Path) -> Result<Vec<f32>> {
    let t0 = Instant::now();
    let w: Vec<Ints> = mats.iter().map(|c| ints(c.cols, &c.raw)).collect();
    let n = mats.iter().map(|c| c.matrix + 1).max().unwrap();
    let xs = RefCell::new(vec![Vec::<Vec<f32>>::new(); n]);
    let ys = RefCell::new(vec![Vec::<Vec<Fr>>::new(); mats.len()]);
    let mut obs = |m: usize, x: &[f32]| xs.borrow_mut()[m].push(x.to_vec());
    let mut claimed = |m: usize, out: &mut [f32]| {
        let (xi, e0) = x_ints(xs.borrow()[m].last().unwrap());
        let mut at = 0;
        for k in (0..mats.len()).filter(|&k| mats[k].matrix == m) {
            let y: Vec<Fr> = w[k]
                .w
                .par_chunks(w[k].cols)
                .map(|row| {
                    fr(row
                        .iter()
                        .zip(&xi)
                        .map(|(&a, &b)| a as i128 * b as i128)
                        .sum())
                })
                .collect();
            for (i, &v) in y.iter().enumerate() {
                out[at + i] = y_to_f32(v, w[k].row_exp[i] as u32, e0);
            }
            at += y.len();
            ys.borrow_mut()[k].push(y);
        }
        assert_eq!(at, out.len());
    };
    let out = run_with(&input, dot_soft, qdot_soft, &mut obs, Some(&mut claimed));
    let run_s = t0.elapsed().as_secs_f64();
    drop(w);

    let (ck, vk) = keys();
    let (xs, mut ys) = (xs.into_inner(), ys.into_inner());
    let (mut commit_s, mut prove_s) = (0.0, 0.0);
    let mut proofs = Vec::new();
    for (k, c) in mats.iter().enumerate() {
        let (mut m, row_exp) = Matrix::encode(c.rows, c.cols, &c.raw);
        let mut x_bits = 0;
        let xe: Vec<Vec<Fr>> = xs[c.matrix]
            .iter()
            .map(|x| {
                let (e, _, bits) = encode_x(x, m.col_vars);
                x_bits = x_bits.max(bits);
                e
            })
            .collect();
        assert!(
            m.max_bits + x_bits + (m.col_vars as u32) < 253,
            "{name} chunk {k}: sums may wrap"
        );
        let ye: Vec<Vec<Fr>> = std::mem::take(&mut ys[k])
            .into_iter()
            .map(|mut y| {
                y.resize(1 << m.row_vars, Fr::zero());
                y
            })
            .collect();
        let t1 = Instant::now();
        let poly = LabeledPolynomial::new(
            String::new(),
            // Take W instead of cloning it, to keep one copy in memory.
            Poly::from_evaluations_vec(m.row_vars + m.col_vars, std::mem::take(&mut m.w)),
            None,
            None,
        );
        let (coms, states) = Pcs::commit(&ck, [&poly], None).unwrap();
        commit_s += t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let p = prove(
            &m,
            &ck,
            &poly,
            &coms[0],
            &states[0],
            &xe,
            &ye,
            &mut Transcript::new(&()),
        );
        prove_s += t2.elapsed().as_secs_f64();
        let dims = Dims {
            rows: m.rows as u64,
            cols: m.cols as u64,
            row_vars: m.row_vars as u64,
            col_vars: m.col_vars as u64,
        };
        let mut mp = MatrixProof {
            matrix: c.matrix as u32,
            dims,
            commitment: coms[0].commitment().clone(),
            row_exp: row_exp.iter().map(|&e| e as u32).collect(),
            xs: xe,
            ys: ye,
            proof: p,
        };
        assert!(mp.verify(&vk), "{name} chunk {k}: proof rejected");
        mp.xs = Vec::new();
        proofs.push(mp);
    }

    let mut stdin = Vec::new();
    let mut proof_bytes = Vec::new();
    proofs.serialize_uncompressed(&mut proof_bytes)?;
    for rec in [&input, &proof_bytes] {
        stdin.extend((rec.len() as u64).to_le_bytes());
        stdin.extend_from_slice(rec);
        stdin.resize(stdin.len().next_multiple_of(8), 0);
    }
    std::fs::write(out_dir.join(format!("{name}.stdin")), &stdin)?;
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut expected = sha256(&input).to_vec();
    expected.extend(weights_id(&proofs));
    expected.extend(sha256(&bytes));
    std::fs::write(out_dir.join(format!("{name}.expected")), &expected)?;
    println!(
        "{name}: native {run_s:.1} s, commit {commit_s:.1} s, prove {prove_s:.1} s, guest input {} MB",
        stdin.len() >> 20
    );
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(&args[1]);
    let tokens: Vec<usize> = args[2]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let out_dir = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&out_dir)?;
    let n_out: usize = args.get(4).map_or(Ok(8), |a| a.parse())?;
    let cfg: Value = serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
    let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let layers: usize = args
        .get(5)
        .map_or(Ok(n("num_hidden_layers")), |a| a.parse())?;
    let mut h = Header {
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
        n_out,
    };
    let floats = |v: &[u16]| -> Vec<u8> {
        v.iter()
            .flat_map(|&b| (u32::from(b) << 16).to_le_bytes())
            .collect()
    };
    let emb = bf16_tensor(&model, "model.embed_tokens.weight")?;
    let mut x: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| {
            emb[t * h.dim..][..h.dim]
                .iter()
                .map(|&b| f32::from_bits(u32::from(b) << 16))
        })
        .collect();
    println!(
        "BN254 Fr, Ligero sec_param 128, rho_inv 2, SHA-256; {} threads; {} tokens",
        rayon::current_num_threads(),
        tokens.len()
    );

    // Each matrix's columns.
    let shapes = [
        ("self_attn.q_proj", h.dim),
        ("self_attn.k_proj", h.dim),
        ("self_attn.v_proj", h.dim),
        ("self_attn.o_proj", h.q_dim()),
        ("mlp.gate_proj", h.dim),
        ("mlp.up_proj", h.dim),
        ("mlp.down_proj", h.hidden),
    ];
    for l in 0..layers {
        let t = |name: &str| bf16_tensor(&model, &format!("model.layers.{l}.{name}.weight"));
        let mut input = h.to_bytes().to_vec();
        input.extend(x.iter().flat_map(|v| v.to_le_bytes()));
        for name in [
            "input_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "post_attention_layernorm",
        ] {
            input.extend(floats(&t(name)?));
        }
        let mut mats = Vec::new();
        for (m, (name, cols)) in shapes.into_iter().enumerate() {
            mats.extend(chunks(m, cols, &t(name)?));
        }
        x = segment(&format!("layer{l:02}"), input, &mats, &out_dir)?;
    }
    if layers < n("num_hidden_layers") {
        return Ok(());
    }

    h.kind = HEAD;
    let mut input = h.to_bytes().to_vec();
    input.extend(x.iter().flat_map(|v| v.to_le_bytes()));
    input.extend(floats(&bf16_tensor(&model, "model.norm.weight")?));
    let mats = chunks(0, h.dim, &emb);
    drop(emb);
    let logits = segment("head", input, &mats, &out_dir)?;
    let next: Vec<usize> = logits.chunks(h.vocab).map(argmax).collect();
    println!("next tokens at the last {n_out} positions: {next:?}");
    Ok(())
}
