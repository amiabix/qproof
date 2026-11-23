//! One Qwen3-4B decoder layer under the ordered-FP32 / original-BF16 profile,
//! with every consumed byte bound into the child receipt. The guest and the
//! native host run this same function; only the SHA-256 implementation differs.
//!
//! Record v1 (little-endian, no trailing bytes):
//! ```text
//! "PCILAYR1" | model 32 | suite 32 | request 32 | layer u32 | positions u32
//! | profile header 64 (must equal pci_receipts::layer_profile(positions))
//! | A_layer: positions x 2560 FP32 bits
//! | 11 raw BF16 checkpoint tensors in pci_receipts::LAYER_TENSORS order
//! ```
use pci_receipts::{
    DIM, Hash, LAYER_TENSORS, LAYERS, Receipt, Role, activation_prefix, layer_profile, weights_root,
};
use qwen_common::{HEADER_BYTES, run_with};
use std::cell::RefCell;

pub const MAGIC: &[u8; 8] = b"PCILAYR1";
pub const FIXED: usize = 8 + 3 * 32 + 4 + 4 + HEADER_BYTES;
pub type Sha = fn(&[u8]) -> Hash;

/// BF16 element count of each tensor in [`LAYER_TENSORS`] order.
pub const TENSOR_LEN: [usize; 11] = [
    2560,
    2560 * 9728,
    9728 * 2560,
    9728 * 2560,
    2560,
    128,
    1024 * 2560,
    2560 * 4096,
    128,
    4096 * 2560,
    1024 * 2560,
];
/// Positions in [`LAYER_TENSORS`] of the seven projections, in the order
/// `qwen_common::run_with` numbers them (q, k, v, o, gate, up, down), and
/// their (rows, columns).
const PROJECTIONS: [(usize, usize, usize); 7] = [
    (9, 4096, 2560),
    (6, 1024, 2560),
    (10, 1024, 2560),
    (7, 2560, 4096),
    (2, 9728, 2560),
    (3, 9728, 2560),
    (1, 2560, 9728),
];
const INPUT_LN: usize = 0;
const POST_LN: usize = 4;
const K_NORM: usize = 5;
const Q_NORM: usize = 8;

pub struct Outcome {
    pub receipt: Receipt,
    pub output: Vec<f32>,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// Exact BF16 to FP32 widening.
pub fn widen(bf16: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bf16.chunks_exact(2)
        .map(|p| f32::from_bits(u32::from(u16::from_le_bytes([p[0], p[1]])) << 16))
}

/// `out[r] = sum_j w[r][j] * x[j]`, each product and each addition rounded
/// to FP32 (RNE) in increasing j, starting from +0. This is
/// `qwen_common::dot_soft` on the exactly widened row.
pub fn project(x: &[f32], matrix: &[u8], out: &mut [f32]) {
    assert_eq!(matrix.len(), 2 * x.len() * out.len());
    for (o, row) in out.iter_mut().zip(matrix.chunks_exact(2 * x.len())) {
        let mut acc = 0.0f32;
        for (w, &v) in widen(row).zip(x) {
            acc += w * v;
        }
        *o = acc;
    }
}

fn activation_root(
    sha: Sha,
    model: Hash,
    request: Hash,
    boundary: u32,
    positions: u32,
    bytes: &[u8],
) -> Result<Hash, &'static str> {
    let mut pre =
        activation_prefix(model, request, boundary, positions).map_err(|_| "activation shape")?;
    pre.extend_from_slice(bytes);
    Ok(sha(&pre))
}

/// A projection kernel: `out[r] = row r of the raw BF16 matrix . x`, where `x`
/// holds FP32 bit patterns. Every kernel must be bit-identical to [`project`];
/// the receipt and digest do not identify the kernel.
pub type Project<'a> = &'a mut dyn FnMut(&[u32], &[u8], &mut [f32]);

/// [`project`] over FP32 bit patterns: the reference kernel.
pub fn project_bits(x: &[u32], matrix: &[u8], out: &mut [f32]) {
    // SAFETY: u32 and f32 have the same size, alignment and no invalid values.
    let x = unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<f32>(), x.len()) };
    project(x, matrix, out)
}

/// Authenticate, compute and commit one layer with the reference kernel.
/// Any error means no receipt.
pub fn execute(record: &[u8], sha: Sha) -> Result<Outcome, &'static str> {
    execute_with(record, sha, &mut project_bits)
}

/// As [`execute`] with a caller-supplied projection kernel.
pub fn execute_with(record: &[u8], sha: Sha, kernel: Project) -> Result<Outcome, &'static str> {
    if record.len() < FIXED || &record[..8] != MAGIC {
        return Err("magic");
    }
    let id = |at: usize| -> Hash { record[at..at + 32].try_into().unwrap() };
    let (model, request) = (id(8), id(72));
    let index = u32_at(record, 104);
    let positions = u32_at(record, 108);
    if index as usize >= LAYERS || !(1..=40_960).contains(&positions) {
        return Err("layer index or positions");
    }
    let profile = layer_profile(positions);
    if record[112..FIXED] != profile {
        return Err("arithmetic profile");
    }
    let t = positions as usize;
    let act_bytes = t * DIM as usize * 4;
    let expected = FIXED + act_bytes + TENSOR_LEN.iter().map(|n| 2 * n).sum::<usize>();
    if record.len() != expected {
        return Err("record length");
    }
    let x_bytes = &record[FIXED..FIXED + act_bytes];
    let mut tensors = [&record[..0]; 11];
    let mut at = FIXED + act_bytes;
    for (slot, n) in tensors.iter_mut().zip(TENSOR_LEN) {
        *slot = &record[at..at + 2 * n];
        at += 2 * n;
    }

    // Weights: the guest hashes exactly the bytes it consumes below.
    let digests: Vec<Hash> = tensors.iter().map(|t| sha(t)).collect();
    let role = Role::Layer(index);
    let weights = weights_root(role, &digests);
    let input = activation_root(sha, model, request, index, positions, x_bytes)?;

    // Segment for qwen_common: header, x, input_ln, q_norm, k_norm, post_ln as
    // FP32; projections are claimed from the BF16 matrices.
    let mut seg: Vec<u32> = Vec::with_capacity(16 + t * DIM as usize + 2 * 2560 + 256);
    seg.extend(
        profile
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
    );
    for b in x_bytes.chunks_exact(4) {
        let v = u32::from_le_bytes(b.try_into().unwrap());
        if (v >> 23) & 255 == 255 {
            return Err("nonfinite input activation");
        }
        seg.push(v);
    }
    for k in [INPUT_LN, Q_NORM, K_NORM, POST_LN] {
        seg.extend(widen(tensors[k]).map(f32::to_bits));
    }
    // SAFETY: u32 to bytes reinterpretation; the Vec outlives the slice.
    let seg_bytes = unsafe { std::slice::from_raw_parts(seg.as_ptr().cast::<u8>(), 4 * seg.len()) };

    let inputs = RefCell::new(core::array::from_fn::<Vec<u32>, 7, _>(|_| Vec::new()));
    let mut observe = |m: usize, v: &[f32]| {
        let x = &mut inputs.borrow_mut()[m];
        x.clear();
        x.extend(v.iter().map(|f| f.to_bits()));
    };
    let mut claim = |m: usize, out: &mut [f32]| {
        let (slot, rows, cols) = PROJECTIONS[m];
        let x = &inputs.borrow()[m];
        assert_eq!((out.len(), x.len()), (rows, cols), "projection shape");
        kernel(x, tensors[slot], out);
    };
    let dot_unused: qwen_common::QDotFn = |_, _, _, _, _| panic!("quantized path disabled");
    let output = run_with(
        seg_bytes,
        qwen_common::dot_soft,
        dot_unused,
        &mut observe,
        Some(&mut claim),
    );
    if output.len() != t * DIM as usize || output.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite or misshapen output");
    }
    let out_bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_root = activation_root(sha, model, request, index + 1, positions, &out_bytes)?;
    Ok(Outcome {
        receipt: Receipt {
            role,
            positions,
            first_prediction: 0,
            predictions: 0,
            input,
            output: out_root,
            weights,
        },
        output,
    })
}

/// Receipt digest the guest commits for a record.
pub fn digest_of(record: &[u8], receipt: &Receipt) -> Hash {
    let id = |at: usize| -> Hash { record[at..at + 32].try_into().unwrap() };
    receipt.digest(id(8), id(40), id(72))
}

/// Build a record. `tensors` are raw BF16 checkpoint bytes in [`LAYER_TENSORS`] order.
pub fn encode(
    model: Hash,
    suite: Hash,
    request: Hash,
    index: u32,
    x: &[f32],
    tensors: &[Vec<u8>],
) -> Vec<u8> {
    assert_eq!(tensors.len(), LAYER_TENSORS.len());
    let positions = (x.len() / DIM as usize) as u32;
    let mut r = MAGIC.to_vec();
    r.extend(model);
    r.extend(suite);
    r.extend(request);
    r.extend(index.to_le_bytes());
    r.extend(positions.to_le_bytes());
    r.extend(layer_profile(positions));
    r.extend(x.iter().flat_map(|v| v.to_le_bytes()));
    for t in tensors {
        r.extend_from_slice(t);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(v: &[f32]) -> Vec<u64> {
        v.chunks(2)
            .map(|p| {
                u64::from(p[0].to_bits()) | u64::from(p.get(1).map_or(0, |h| h.to_bits())) << 32
            })
            .collect()
    }

    #[test]
    fn projection_is_dot_soft_on_widened_weights() {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for cols in [1usize, 2, 7, 64, 129] {
            let rows = 5;
            let mut w = vec![0u8; 2 * rows * cols];
            for b in w.chunks_exact_mut(2) {
                let mut v = next() as u16;
                if (v >> 7) & 255 == 255 {
                    v &= !(1 << 14);
                }
                b.copy_from_slice(&v.to_le_bytes());
            }
            let x: Vec<f32> = (0..cols)
                .map(|_| {
                    let b = next() as u32;
                    let f = f32::from_bits(b);
                    if f.is_finite() { f } else { -0.0 }
                })
                .collect();
            let mut out = vec![0.0; rows];
            project(&x, &w, &mut out);
            for (r, o) in out.iter().enumerate() {
                let row: Vec<f32> = widen(&w[2 * r * cols..2 * (r + 1) * cols]).collect();
                let e = qwen_common::dot_soft(&pack(&row), &pack(&x), cols);
                assert_eq!(o.to_bits(), e.to_bits());
            }
        }
    }
}

/// A batched projection kernel: `kernel(x, cols, matrices, outs)` sets `outs[k]`
/// (`positions x rows_k`, row-major) to every row of `x` (`positions x cols` FP32 bit
/// patterns) times raw BF16 `matrices[k]` (`rows_k x cols`). Every kernel must be
/// bit-identical to [`project`] applied per position; the receipt and digest do not
/// identify the kernel.
pub type ProjectBatch<'a> = &'a mut dyn FnMut(&[u32], usize, &[&[u8]], &mut [&mut [f32]]);

/// As [`execute_with`], but each projection receives all positions at once, and the
/// projections sharing an input (q/k/v, gate/up) are claimed together
/// (`qwen_common::run_layer_batched`). Same record checks, same receipt.
pub fn execute_batched(
    record: &[u8],
    sha: Sha,
    kernel: ProjectBatch,
) -> Result<Outcome, &'static str> {
    if record.len() < FIXED || &record[..8] != MAGIC {
        return Err("magic");
    }
    let id = |at: usize| -> Hash { record[at..at + 32].try_into().unwrap() };
    let (model, request) = (id(8), id(72));
    let index = u32_at(record, 104);
    let positions = u32_at(record, 108);
    if index as usize >= LAYERS || !(1..=40_960).contains(&positions) {
        return Err("layer index or positions");
    }
    let profile = layer_profile(positions);
    if record[112..FIXED] != profile {
        return Err("arithmetic profile");
    }
    let t = positions as usize;
    let act_bytes = t * DIM as usize * 4;
    let expected = FIXED + act_bytes + TENSOR_LEN.iter().map(|n| 2 * n).sum::<usize>();
    if record.len() != expected {
        return Err("record length");
    }
    let x_bytes = &record[FIXED..FIXED + act_bytes];
    let mut tensors = [&record[..0]; 11];
    let mut at = FIXED + act_bytes;
    for (slot, n) in tensors.iter_mut().zip(TENSOR_LEN) {
        *slot = &record[at..at + 2 * n];
        at += 2 * n;
    }

    let digests: Vec<Hash> = tensors.iter().map(|t| sha(t)).collect();
    let role = Role::Layer(index);
    let weights = weights_root(role, &digests);
    let input = activation_root(sha, model, request, index, positions, x_bytes)?;

    let mut seg: Vec<u32> = Vec::with_capacity(16 + t * DIM as usize + 2 * 2560 + 256);
    seg.extend(
        profile
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
    );
    for b in x_bytes.chunks_exact(4) {
        let v = u32::from_le_bytes(b.try_into().unwrap());
        if (v >> 23) & 255 == 255 {
            return Err("nonfinite input activation");
        }
        seg.push(v);
    }
    for k in [INPUT_LN, Q_NORM, K_NORM, POST_LN] {
        seg.extend(widen(tensors[k]).map(f32::to_bits));
    }
    // SAFETY: u32 to bytes reinterpretation; the Vec outlives the slice.
    let seg_bytes = unsafe { std::slice::from_raw_parts(seg.as_ptr().cast::<u8>(), 4 * seg.len()) };

    let mut claim = |group: usize, x: &[f32], outs: &mut [&mut [f32]]| {
        let members = qwen_common::BATCH_GROUPS[group];
        assert_eq!(outs.len(), members.len(), "projection group");
        let cols = PROJECTIONS[members[0]].2;
        assert_eq!(x.len(), t * cols, "projection input shape");
        let mut matrices = [&record[..0]; 3];
        for (k, &m) in members.iter().enumerate() {
            let (slot, rows, c) = PROJECTIONS[m];
            assert_eq!((c, outs[k].len()), (cols, t * rows), "projection shape");
            matrices[k] = tensors[slot];
        }
        // SAFETY: f32 to u32 reinterpretation of the same memory.
        let bits = unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<u32>(), x.len()) };
        kernel(bits, cols, &matrices[..members.len()], outs);
    };
    let output = qwen_common::run_layer_batched(seg_bytes, qwen_common::dot_soft, &mut claim);
    if output.len() != t * DIM as usize || output.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite or misshapen output");
    }
    let out_bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out_root = activation_root(sha, model, request, index + 1, positions, &out_bytes)?;
    Ok(Outcome {
        receipt: Receipt {
            role,
            positions,
            first_prediction: 0,
            predictions: 0,
            input,
            output: out_root,
            weights,
        },
        output,
    })
}

/// [`project`] applied per position over a group: the batched reference kernel.
pub fn project_batch_reference(x: &[u32], cols: usize, matrices: &[&[u8]], outs: &mut [&mut [f32]]) {
    for (m, out) in matrices.iter().zip(outs.iter_mut()) {
        let rows = m.len() / (2 * cols);
        for (xt, ot) in x.chunks_exact(cols).zip(out.chunks_exact_mut(rows)) {
            project_bits(xt, m, ot);
        }
    }
}
