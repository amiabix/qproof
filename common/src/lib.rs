//! Qwen3 in `f32`, one segment at a time: the embedding, one decoder layer, or
//! the head. A segment is one input record:
//!
//! ```text
//! u32 header[16]: kind, dim, hidden, heads, kv_heads, head_dim, vocab, n_tok,
//!                 rope_theta (f32), eps (f32), flags, n_out
//! embed: u32 tokens[n_tok, padded to even], f32 embed[vocab][dim]
//! layer: f32 x[n_tok][dim], input_ln[dim], q[heads*hd][dim], k[kv*hd][dim],
//!        v[kv*hd][dim], o[dim][heads*hd], q_norm[hd], k_norm[hd], post_ln[dim],
//!        gate[hidden][dim], up[hidden][dim], down[dim][hidden]
//! head:  f32 x[n_tok][dim], norm[dim], embed[vocab][dim]
//! ```
//!
//! Matrices are row-major `f32`, which on an 8-byte boundary is already the
//! FloatDot operand layout: two values per `u64`, low half first.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

pub const EMBED: u32 = 0;
pub const LAYER: u32 = 1;
pub const HEAD: u32 = 2;

/// Flag: the guest also hashes the segment's whole input.
pub const HASH_INPUT: u32 = 1;
/// Flag: dot products use FloatDot.
pub const FLOATDOT: u32 = 2;
/// Flag: matrices are E4M3 in groups of [`GROUP`] with an `f32` scale per group, and matrix
/// multiplications quantize their input the same way and take exact group sums.
pub const DOTF8: u32 = 4;
/// Values per E4M3 group.
pub const GROUP: usize = 128;
/// Flag, with `DOTF8`: one group per matrix row and per activation vector.
pub const ROWSCALE: u32 = 8;
/// Flag, with `DOTF8`: attention in E4M3 too. q and k with one scale per head, v with one
/// scale per channel over all positions, the softmax weights with one per query.
pub const ATTN8: u32 = 16;

pub type DotFn = fn(&[u64], &[u64], usize) -> f32;
/// The exact sum of each group of `g` products of two E4M3 vectors of `n` values, in units of
/// 2^-18.
pub type QDotFn = fn(&[u64], &[u64], usize, usize, &mut [i64]);

/// Value of an E4M3 code in units of 2^-9 (OCP E4M3FN: NaN decodes to 0).
const fn e4m3(c: u8) -> i32 {
    let (e, m) = ((c >> 3) & 15, (c & 7) as i32);
    let mag = if e == 15 && m == 7 {
        0
    } else if e == 0 {
        m
    } else {
        (8 + m) << (e - 1)
    };
    if c & 0x80 != 0 { -mag } else { mag }
}

static E4M3: [i32; 256] = {
    let mut t = [0; 256];
    let mut c = 0;
    while c < 256 {
        t[c] = e4m3(c as u8);
        c += 1;
    }
    t
};

fn bytes(w: &[u64]) -> &[u8] {
    // SAFETY: any u64 is 8 valid bytes.
    unsafe { core::slice::from_raw_parts(w.as_ptr().cast(), 8 * w.len()) }
}

pub fn qdot_soft(w: &[u64], x: &[u64], n: usize, g: usize, sums: &mut [i64]) {
    let (w, x) = (bytes(w), bytes(x));
    for (k, s) in sums[..n / g].iter_mut().enumerate() {
        *s = (k * g..(k + 1) * g)
            .map(|i| E4M3[w[i] as usize] as i64 * E4M3[x[i] as usize] as i64)
            .sum();
    }
}

#[cfg(feature = "dotf8")]
pub fn qdot_dotf8(w: &[u64], x: &[u64], n: usize, g: usize, sums: &mut [i64]) {
    use ziskos::syscalls::{DOT_F8_E4M3, SyscallDotF8Params, dot_f8_formats, syscall_dot_f8};
    let p = SyscallDotF8Params {
        x: w.as_ptr().cast(),
        y: x.as_ptr().cast(),
        n: n as u64,
        formats: dot_f8_formats(DOT_F8_E4M3, DOT_F8_E4M3),
        block: g as u64,
        out: sums.as_mut_ptr(),
    };
    // SAFETY: both vectors hold n aligned bytes and sums n / g values.
    unsafe { syscall_dot_f8(&p) };
}

/// `x >> sh` rounded to nearest, ties to even (`sh` in 1..=63).
fn shr_round_even(x: u64, sh: u32) -> u64 {
    let (q, rem, half) = (x >> sh, x & ((1 << sh) - 1), 1 << (sh - 1));
    if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// The E4M3 code nearest to `v`, ties to even, saturating at 448; f32 subnormals become 0.
fn e4m3_encode(v: f32) -> u8 {
    let bits = v.to_bits();
    let exp = (bits >> 23) as i32 & 0xff;
    let mag = if exp == 0 {
        0
    } else {
        let sig = (bits & 0x7f_ffff) as u64 | 0x80_0000;
        let x = exp - 127;
        let code = if x >= -6 {
            ((((x + 7) as i64) << 3) + shr_round_even(sig, 20) as i64 - 8).min(0x7e)
        } else {
            let sh = 23 - x - 9;
            if sh > 40 {
                0
            } else {
                shr_round_even(sig, sh as u32) as i64
            }
        };
        code as u8
    };
    if mag == 0 || bits >> 31 == 0 {
        mag
    } else {
        mag | 0x80
    }
}

/// Quantizes one group `v` to E4M3 codes `q`; returns the scale of one unit of 2^-9, so that
/// `v[i] ≈ e4m3(q[i]) · scale`.
pub fn quantize(v: &[f32], q: &mut [u8]) -> f32 {
    let scale = v.iter().fold(0.0f32, |m, &x| m.max(libm::fabsf(x))) / 448.0;
    for (c, &x) in q.iter_mut().zip(v) {
        *c = if scale == 0.0 {
            0
        } else {
            e4m3_encode(x / scale)
        };
    }
    scale / 512.0
}

pub fn dot_soft(x: &[u64], y: &[u64], n: usize) -> f32 {
    let lo = |w: u64| f32::from_bits(w as u32);
    let hi = |w: u64| f32::from_bits((w >> 32) as u32);
    let mut acc = 0.0f32;
    for (&a, &b) in x[..n / 2].iter().zip(&y[..n / 2]) {
        acc += lo(a) * lo(b);
        acc += hi(a) * hi(b);
    }
    if n % 2 == 1 {
        acc += lo(x[n / 2]) * lo(y[n / 2]);
    }
    acc
}

/// Dot products whose exact sum did not fit `dot_exact`'s i128.
pub static EXACT_FALLBACKS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// The exact sum of the products, rounded to `f32` once.
pub fn dot_exact(x: &[u64], y: &[u64], n: usize) -> f32 {
    let lo = |w: u64| f32::from_bits(w as u32);
    let hi = |w: u64| f32::from_bits((w >> 32) as u32);
    // Each f32 product is exact in f64: `±m · 2^e` with m odd.
    let terms: Vec<(i128, i32)> = (0..n)
        .map(|i| {
            let (a, b) = if i % 2 == 0 {
                (lo(x[i / 2]), lo(y[i / 2]))
            } else {
                (hi(x[i / 2]), hi(y[i / 2]))
            };
            let bits = (f64::from(a) * f64::from(b)).to_bits();
            let (f, e) = (bits & ((1 << 52) - 1), ((bits >> 52) & 0x7ff) as i32);
            let (m, e) = if e == 0 {
                (f, -1074)
            } else {
                (f | 1 << 52, e - 1075)
            };
            if m == 0 {
                return (0, 0);
            }
            let tz = m.trailing_zeros();
            let m = (m >> tz) as i128;
            (if bits >> 63 == 1 { -m } else { m }, e + tz as i32)
        })
        .filter(|t| t.0 != 0)
        .collect();
    let Some(emin) = terms.iter().map(|t| t.1).min() else {
        return 0.0;
    };
    let top = terms
        .iter()
        .map(|&(m, e)| e + (128 - m.unsigned_abs().leading_zeros()) as i32)
        .max()
        .unwrap();
    if top - emin + (usize::BITS - n.leading_zeros()) as i32 > 126 {
        EXACT_FALLBACKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        return dot_soft(x, y, n);
    }
    let s: i128 = terms.iter().map(|&(m, e)| m << (e - emin)).sum();
    libm::ldexpf(s as f32, emin)
}

#[cfg(feature = "floatdot")]
pub fn dot_floatdot(x: &[u64], y: &[u64], n: usize) -> f32 {
    const MAX: usize = 4096;
    let words = n.div_ceil(2);
    let mut acc = 0u32;
    for (i, (xc, yc)) in x[..words]
        .chunks(MAX / 2)
        .zip(y[..words].chunks(MAX / 2))
        .enumerate()
    {
        let len = (n - i * MAX).min(MAX);
        let r =
            ziskos::syscalls::syscall_fp32_dot(xc, yc, len, acc, 0).expect("valid FloatDot call");
        acc = r as u32;
    }
    f32::from_bits(acc)
}

#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub kind: u32,
    pub dim: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub n_tok: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub flags: u32,
    /// Head only: positions whose logits are computed, the last `n_out`.
    pub n_out: usize,
}

pub const HEADER_BYTES: usize = 64;

impl Header {
    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let v = [
            self.kind,
            self.dim as u32,
            self.hidden as u32,
            self.heads as u32,
            self.kv_heads as u32,
            self.head_dim as u32,
            self.vocab as u32,
            self.n_tok as u32,
            self.rope_theta.to_bits(),
            self.eps.to_bits(),
            self.flags,
            self.n_out as u32,
        ];
        let mut b = [0; HEADER_BYTES];
        for (i, x) in v.iter().enumerate() {
            b[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
        }
        b
    }

    pub fn parse(b: &[u8]) -> Self {
        let u = |i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
        Header {
            kind: u(0),
            dim: u(1) as usize,
            hidden: u(2) as usize,
            heads: u(3) as usize,
            kv_heads: u(4) as usize,
            head_dim: u(5) as usize,
            vocab: u(6) as usize,
            n_tok: u(7) as usize,
            rope_theta: f32::from_bits(u(8)),
            eps: f32::from_bits(u(9)),
            flags: u(10),
            n_out: u(11) as usize,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Splits `bytes` into consecutive `f32` / packed-word views.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let b = &self.bytes[self.at..self.at + 4 * n];
        self.at += 4 * n;
        b
    }

    fn floats(&mut self, n: usize) -> &'a [f32] {
        // SAFETY: f32 has no invalid bit patterns.
        let (pre, v, post) = unsafe { self.take(n).align_to::<f32>() };
        assert!(pre.is_empty() && post.is_empty(), "misaligned f32 tensor");
        v
    }

    fn words(&mut self, n: usize) -> &'a [u64] {
        assert!(n % 2 == 0, "odd tensor size");
        // SAFETY: u64 has no invalid bit patterns.
        let (pre, v, post) = unsafe { self.take(n).align_to::<u64>() };
        assert!(pre.is_empty() && post.is_empty(), "misaligned matrix");
        v
    }
}

/// Quantizes `v` as one E4M3 group into the first `v.len()` bytes of `w`, zeroing the rest;
/// returns the unit scale.
fn qwords(v: &[f32], w: &mut [u64]) -> f32 {
    w.fill(0);
    // SAFETY: the words hold at least v.len() bytes.
    let b = unsafe { core::slice::from_raw_parts_mut(w.as_mut_ptr().cast(), v.len()) };
    quantize(v, b)
}

/// A matrix: packed `f32`, or with `DOTF8` E4M3 codes and each group's unit scale.
enum Mat<'a> {
    F32(&'a [u64]),
    Q(&'a [u64], &'a [f32]),
}

impl<'a> Reader<'a> {
    fn mat(&mut self, rows: usize, cols: usize, q: bool, g: usize) -> Mat<'a> {
        if q {
            Mat::Q(self.words(rows * cols / 4), self.floats(rows * cols / g))
        } else {
            Mat::F32(self.words(rows * cols))
        }
    }
}

/// An activation vector as the matrices read it: packed `f32`, or E4M3 codes and scales.
struct Act {
    w: Vec<u64>,
    s: Vec<f32>,
    q: bool,
    /// Values per group.
    g: usize,
}

impl Act {
    fn new(n: usize, q: bool, g: usize) -> Self {
        Act {
            w: vec![0; n.div_ceil(2)],
            s: vec![0.0; n / g],
            q,
            g,
        }
    }

    fn set(&mut self, x: &[f32]) {
        if self.q {
            // SAFETY: the words hold at least x.len() bytes.
            let b = unsafe { core::slice::from_raw_parts_mut(self.w.as_mut_ptr().cast(), x.len()) };
            for (k, (v, c)) in x.chunks(self.g).zip(b.chunks_mut(self.g)).enumerate() {
                self.s[k] = quantize(v, c);
            }
        } else {
            pack(x, &mut self.w);
        }
    }
}

/// Values per E4M3 group of a matrix with `cols` columns.
fn group(h: &Header, cols: usize) -> usize {
    if h.flags & ROWSCALE != 0 { cols } else { GROUP }
}

fn pack(v: &[f32], words: &mut [u64]) {
    for (w, pair) in words.iter_mut().zip(v.chunks(2)) {
        let hi = pair.get(1).map_or(0, |h| h.to_bits());
        *w = pair[0].to_bits() as u64 | (hi as u64) << 32;
    }
}

/// `out[i] = row i of w · x`, `w` with `cols` columns.
fn matmul(dot: DotFn, qdot: QDotFn, out: &mut [f32], x: &Act, w: &Mat, cols: usize) {
    match *w {
        Mat::F32(w) => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = dot(&w[i * cols / 2..][..cols / 2], &x.w, cols);
            }
        }
        Mat::Q(w, ws) => {
            let g = cols / x.g;
            let mut sums = vec![0i64; g];
            for (i, o) in out.iter_mut().enumerate() {
                qdot(&w[i * cols / 8..][..cols / 8], &x.w, cols, x.g, &mut sums);
                *o = sums
                    .iter()
                    .zip(&ws[i * g..][..g])
                    .zip(&x.s)
                    .fold(0.0, |a, ((&v, &sw), &sx)| a + v as f32 * sw * sx);
            }
        }
    }
}

fn rmsnorm(dot: DotFn, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32, xw: &mut [u64]) {
    pack(x, xw);
    let ss = dot(xw, xw, x.len()) / x.len() as f32;
    let inv = 1.0 / libm::sqrtf(ss + eps);
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (v * inv);
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(x[0], |m, &v| m.max(v));
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// HF's `rotate_half` RoPE on one head at position `pos`.
fn rope(v: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = v.len() / 2;
    for i in 0..half {
        let a = pos as f32 * inv_freq[i];
        let (c, s) = (libm::cosf(a), libm::sinf(a));
        let (x1, x2) = (v[i], v[i + half]);
        v[i] = x1 * c - x2 * s;
        v[i + half] = x2 * c + x1 * s;
    }
}

/// The segment's output: the activations, or for the head the logits of the
/// last `n_out` positions.
pub fn run(input: &[u8], dot: DotFn, qdot: QDotFn) -> Vec<f32> {
    run_with(input, dot, qdot, &mut |_, _| {}, None)
}

/// Fills `out` with the product of matrix `matrix`; called once per product, in order.
pub type Claimed<'a> = &'a mut dyn FnMut(usize, &mut [f32]);

/// Like [`run`], but passes each matrix product's input to `obs` as `(matrix, x)`,
/// numbering matrices as in [`MATRICES`] (0 for the head). With `claimed`, the
/// input has no matrices and each product comes from `claimed` right after its `obs`.
pub fn run_with(
    input: &[u8],
    dot: DotFn,
    qdot: QDotFn,
    obs: &mut dyn FnMut(usize, &[f32]),
    mut claimed: Option<Claimed>,
) -> Vec<f32> {
    let h = Header::parse(input);
    let mut r = Reader {
        bytes: input,
        at: HEADER_BYTES,
    };
    match h.kind {
        EMBED => {
            let raw = r.take(h.n_tok.next_multiple_of(2));
            let tokens: Vec<usize> = raw
                .chunks_exact(4)
                .take(h.n_tok)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                .collect();
            let emb = r.floats(h.vocab * h.dim);
            tokens
                .iter()
                .flat_map(|&t| emb[t * h.dim..][..h.dim].iter().copied())
                .collect()
        }
        LAYER => layer(&h, &mut r, dot, qdot, obs, claimed),
        HEAD => {
            let x = r.floats(h.n_tok * h.dim);
            let norm = r.floats(h.dim);
            let emb = claimed
                .is_none()
                .then(|| r.mat(h.vocab, h.dim, h.flags & DOTF8 != 0, group(&h, h.dim)));
            let (mut xn, mut xw) = (vec![0.0; h.dim], vec![0; h.dim / 2]);
            let mut a = Act::new(h.dim, h.flags & DOTF8 != 0, group(&h, h.dim));
            let mut logits = vec![0.0; h.n_out * h.vocab];
            for (i, out) in logits.chunks_mut(h.vocab).enumerate() {
                let t = h.n_tok - h.n_out + i;
                rmsnorm(dot, &mut xn, &x[t * h.dim..][..h.dim], norm, h.eps, &mut xw);
                a.set(&xn);
                obs(0, &xn);
                match (claimed.as_mut(), &emb) {
                    (Some(c), _) => c(0, out),
                    (None, Some(emb)) => matmul(dot, qdot, out, &a, emb, h.dim),
                    (None, None) => unreachable!(),
                }
            }
            logits
        }
        k => panic!("unknown segment kind {k}"),
    }
}

/// Matrices of a layer, in checkpoint order.
pub const MATRICES: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

fn layer(
    h: &Header,
    r: &mut Reader,
    dot: DotFn,
    qdot: QDotFn,
    obs: &mut dyn FnMut(usize, &[f32]),
    mut claimed: Option<Claimed>,
) -> Vec<f32> {
    let q8 = h.flags & DOTF8 != 0;
    let own = claimed.is_none();
    let (dim, hd, n) = (h.dim, h.head_dim, h.n_tok);
    let (qd, kvd) = (h.q_dim(), h.kv_dim());
    let mut x = r.floats(n * dim).to_vec();
    let input_ln = r.floats(dim);
    let wq = own.then(|| r.mat(qd, dim, q8, group(h, dim)));
    let wk = own.then(|| r.mat(kvd, dim, q8, group(h, dim)));
    let wv = own.then(|| r.mat(kvd, dim, q8, group(h, dim)));
    let wo = own.then(|| r.mat(dim, qd, q8, group(h, qd)));
    let q_norm = r.floats(hd);
    let k_norm = r.floats(hd);
    let post_ln = r.floats(dim);
    let wg = own.then(|| r.mat(h.hidden, dim, q8, group(h, dim)));
    let wu = own.then(|| r.mat(h.hidden, dim, q8, group(h, dim)));
    let wd = own.then(|| r.mat(dim, h.hidden, q8, group(h, h.hidden)));

    let inv_freq: Vec<f32> = (0..hd / 2)
        .map(|i| 1.0 / libm::powf(h.rope_theta, (2 * i) as f32 / hd as f32))
        .collect();
    let scale = 1.0 / libm::sqrtf(hd as f32);
    let mut xw = vec![0u64; dim.max(qd).max(h.hidden) / 2];
    let mut xn = vec![0.0f32; dim];
    let mut q = vec![0.0f32; n * qd];
    let mut k = vec![0.0f32; n * kvd];
    let mut v = vec![0.0f32; n * kvd];
    let mut tmp = vec![0.0f32; hd];
    let (mut ad, mut aq, mut ah) = (
        Act::new(dim, q8, group(h, dim)),
        Act::new(qd, q8, group(h, qd)),
        Act::new(h.hidden, q8, group(h, h.hidden)),
    );
    let mut hw = vec![0u64; hd / 2];
    let mut mm = |m: usize, out: &mut [f32], x: &Act, w: &Option<Mat>, cols: usize| match (
        claimed.as_mut(),
        w,
    ) {
        (Some(c), _) => c(m, out),
        (None, Some(w)) => matmul(dot, qdot, out, x, w, cols),
        (None, None) => unreachable!(),
    };

    for t in 0..n {
        rmsnorm(dot, &mut xn, &x[t * dim..][..dim], input_ln, h.eps, &mut xw);
        ad.set(&xn);
        for m in 0..3 {
            obs(m, &xn);
        }
        mm(0, &mut q[t * qd..][..qd], &ad, &wq, dim);
        mm(1, &mut k[t * kvd..][..kvd], &ad, &wk, dim);
        mm(2, &mut v[t * kvd..][..kvd], &ad, &wv, dim);
        for (heads, weight, buf) in [
            (h.heads, q_norm, &mut q[t * qd..][..qd]),
            (h.kv_heads, k_norm, &mut k[t * kvd..][..kvd]),
        ] {
            for j in 0..heads {
                let head = &mut buf[j * hd..][..hd];
                tmp.copy_from_slice(head);
                rmsnorm(dot, head, &tmp, weight, h.eps, &mut hw);
                rope(head, t, &inv_freq);
            }
        }
    }

    let kw: Vec<u64> = {
        let mut w = vec![0u64; n * kvd / 2];
        pack(&k, &mut w);
        w
    };
    let group = h.heads / h.kv_heads;
    let mut att = vec![0.0f32; n];
    let (mut aw, mut cw) = (vec![0u64; n.div_ceil(2)], vec![0u64; n.div_ceil(2)]);
    let mut qw = vec![0u64; qd / 2];
    let mut attn = vec![0.0f32; qd];
    let mut o = vec![0.0f32; dim];
    let mut g = vec![0.0f32; h.hidden];
    let mut u = vec![0.0f32; h.hidden];
    let a8 = q8 && h.flags & ATTN8 != 0;
    let np = n.next_multiple_of(8);
    let (mut kq, mut ks, mut vq, mut vs) = (vec![], vec![], vec![], vec![]);
    if a8 {
        kq = vec![0u64; n * kvd / 8];
        ks = vec![0.0f32; n * h.kv_heads];
        for (m, head) in k.chunks(hd).enumerate() {
            ks[m] = qwords(head, &mut kq[m * hd / 8..][..hd / 8]);
        }
        vq = vec![0u64; kvd * np / 8];
        vs = vec![0.0f32; kvd];
        let mut col = vec![0.0f32; n];
        for c in 0..kvd {
            for (s, x) in col.iter_mut().enumerate() {
                *x = v[s * kvd + c];
            }
            vs[c] = qwords(&col, &mut vq[c * np / 8..][..np / 8]);
        }
    }
    let (mut qh, mut awq, mut sum) = (vec![0u64; hd / 8], vec![0u64; np / 8], [0i64]);
    for t in 0..n {
        if a8 {
            for j in 0..h.heads {
                let kvh = j / group;
                let qs = qwords(&q[t * qd + j * hd..][..hd], &mut qh);
                let a = &mut att[..t + 1];
                for (s, score) in a.iter_mut().enumerate() {
                    let m = s * h.kv_heads + kvh;
                    qdot(&qh, &kq[m * hd / 8..][..hd / 8], hd, hd, &mut sum);
                    *score = sum[0] as f32 * qs * ks[m] * scale;
                }
                softmax(a);
                let sa = qwords(a, &mut awq);
                for i in 0..hd {
                    let c = kvh * hd + i;
                    qdot(&vq[c * np / 8..][..np / 8], &awq, np, np, &mut sum);
                    attn[j * hd + i] = sum[0] as f32 * vs[c] * sa;
                }
            }
        } else {
            pack(&q[t * qd..][..qd], &mut qw);
            for j in 0..h.heads {
                let kv = (j / group) * hd;
                let a = &mut att[..t + 1];
                for (s, score) in a.iter_mut().enumerate() {
                    *score = dot(
                        &qw[j * hd / 2..][..hd / 2],
                        &kw[(s * kvd + kv) / 2..][..hd / 2],
                        hd,
                    ) * scale;
                }
                softmax(a);
                pack(a, &mut aw);
                for i in 0..hd {
                    let col = |s: usize| match s <= t {
                        true => v[s * kvd + kv + i].to_bits() as u64,
                        false => 0,
                    };
                    for (m, w) in cw[..(t + 1).div_ceil(2)].iter_mut().enumerate() {
                        *w = col(2 * m) | col(2 * m + 1) << 32;
                    }
                    attn[j * hd + i] = dot(&aw, &cw, t + 1);
                }
            }
        }
        aq.set(&attn);
        obs(3, &attn);
        mm(3, &mut o, &aq, &wo, qd);
        let xt = &mut x[t * dim..][..dim];
        for (a, &b) in xt.iter_mut().zip(&o) {
            *a += b;
        }

        rmsnorm(dot, &mut xn, xt, post_ln, h.eps, &mut xw);
        ad.set(&xn);
        obs(4, &xn);
        obs(5, &xn);
        mm(4, &mut g, &ad, &wg, dim);
        mm(5, &mut u, &ad, &wu, dim);
        for (a, &b) in g.iter_mut().zip(&u) {
            *a = *a / (1.0 + libm::expf(-*a)) * b;
        }
        ah.set(&g);
        obs(6, &g);
        mm(6, &mut o, &ah, &wd, h.hidden);
        for (a, &b) in xt.iter_mut().zip(&o) {
            *a += b;
        }
    }
    x
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack2(v: &[f32]) -> Vec<u64> {
        v.chunks(2)
            .map(|p| {
                u64::from(p[0].to_bits()) | u64::from(p.get(1).unwrap_or(&0.0).to_bits()) << 32
            })
            .collect()
    }

    #[test]
    fn e4m3_codes_round_trip() {
        for c in 0..=255u8 {
            if c == 0x80 || c & 0x7f == 0x7f {
                continue; // negative zero, NaN
            }
            let v = e4m3(c) as f32 / 512.0;
            assert_eq!(e4m3_encode(v), c, "code {c:#x}");
        }
        assert_eq!(e4m3_encode(1000.0), 0x7e); // saturates at 448
        assert_eq!(e4m3_encode(1.0 + 1.0 / 16.0), 0x38); // tie to even: 1.0
    }

    #[test]
    fn quantized_dot_is_exact_on_the_codes() {
        let x: Vec<f32> = (0..GROUP).map(|i| (i as f32 - 60.0) / 7.0).collect();
        let y: Vec<f32> = (0..GROUP).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let (mut a, mut b) = (Act::new(GROUP, true, GROUP), Act::new(GROUP, true, GROUP));
        a.set(&x);
        b.set(&y);
        let deq = |act: &Act, i: usize| e4m3(bytes(&act.w)[i]) as f64 * act.s[0] as f64;
        for (i, &v) in x.iter().enumerate() {
            // Half an E4M3 step: 2^-4 relative, or half the subnormal step.
            let bound = (v.abs() / 16.0).max(a.s[0] / 2.0) as f64;
            assert!((deq(&a, i) - v as f64).abs() <= bound, "value {i}");
        }
        let mut s = [0i64];
        qdot_soft(&a.w, &b.w, GROUP, GROUP, &mut s);
        let exact: f64 = (0..GROUP).map(|i| deq(&a, i) * deq(&b, i)).sum();
        let mag: f64 = (0..GROUP).map(|i| (deq(&a, i) * deq(&b, i)).abs()).sum();
        let got = s[0] as f64 * a.s[0] as f64 * b.s[0] as f64;
        assert!((got - exact).abs() <= 1e-12 * mag, "{got} vs {exact}");
    }

    #[test]
    fn exact_rounds_once() {
        let ones = pack2(&[1.0; 4]);
        // Ordered f32 loses the 1 to cancellation; the exact sum keeps it.
        let x = pack2(&[1e8, 1.0, -1e8]);
        assert_eq!(dot_soft(&x, &ones, 3), 0.0);
        assert_eq!(dot_exact(&x, &ones, 3), 1.0);
        // 1 + 2^-24 + 2^-24 is 1 + 2^-23 exactly, but each ordered add ties to 1.
        let x = pack2(&[1.0, 2f32.powi(-24), 2f32.powi(-24)]);
        assert_eq!(dot_soft(&x, &ones, 3), 1.0);
        assert_eq!(dot_exact(&x, &ones, 3), 1.0 + 2f32.powi(-23));
        // Products: 3·0.5 + (-0.25)·4 = 0.5.
        assert_eq!(
            dot_exact(&pack2(&[3.0, -0.25]), &pack2(&[0.5, 4.0]), 2),
            0.5
        );
        assert_eq!(dot_exact(&pack2(&[0.0, 0.0]), &ones, 2), 0.0);
        assert_eq!(
            EXACT_FALLBACKS.load(core::sync::atomic::Ordering::Relaxed),
            0
        );
    }
}

/// Projections that share one input, in the order [`run_layer_batched`] claims them:
/// 0 = q, k, v (input: input-normed x); 1 = o (attention output); 2 = gate, up
/// (post-attention-normed x); 3 = down (SiLU(gate) * up).
pub const BATCH_GROUPS: [&[usize]; 4] = [&[0, 1, 2], &[3], &[4, 5], &[6]];

/// Fills `outs[k]` (`n_tok x rows`, row-major) with the product of matrix
/// `BATCH_GROUPS[group][k]` and every row of `x` (`n_tok x cols`); called once per group,
/// in order.
pub type ClaimedBatch<'a> = &'a mut dyn FnMut(usize, &[f32], &mut [&mut [f32]]);

/// The lossless layer (`flags` without `DOTF8`) of [`run_with`] with claimed products,
/// reordered so each projection sees all positions at once. Every non-projection
/// operation is the same code, per position, in the same order, so for bit-identical
/// claimed products the output equals `run_with`'s bit for bit.
pub fn run_layer_batched(input: &[u8], dot: DotFn, claimed: ClaimedBatch) -> Vec<f32> {
    let h = Header::parse(input);
    assert!(h.kind == LAYER, "batched run is for layers");
    assert!(h.flags & DOTF8 == 0, "batched run is for the lossless profile");
    let mut r = Reader {
        bytes: input,
        at: HEADER_BYTES,
    };
    let (dim, hd, n) = (h.dim, h.head_dim, h.n_tok);
    let (qd, kvd) = (h.q_dim(), h.kv_dim());
    let mut x = r.floats(n * dim).to_vec();
    let input_ln = r.floats(dim);
    let q_norm = r.floats(hd);
    let k_norm = r.floats(hd);
    let post_ln = r.floats(dim);

    let inv_freq: Vec<f32> = (0..hd / 2)
        .map(|i| 1.0 / libm::powf(h.rope_theta, (2 * i) as f32 / hd as f32))
        .collect();
    let scale = 1.0 / libm::sqrtf(hd as f32);
    let mut xw = vec![0u64; dim.max(qd).max(h.hidden) / 2];
    let mut xn = vec![0.0f32; n * dim];
    let mut q = vec![0.0f32; n * qd];
    let mut k = vec![0.0f32; n * kvd];
    let mut v = vec![0.0f32; n * kvd];
    let mut tmp = vec![0.0f32; hd];
    let mut hw = vec![0u64; hd / 2];

    for t in 0..n {
        rmsnorm(dot, &mut xn[t * dim..][..dim], &x[t * dim..][..dim], input_ln, h.eps, &mut xw);
    }
    claimed(0, &xn, &mut [&mut q, &mut k, &mut v]);
    for t in 0..n {
        for (heads, weight, buf) in [
            (h.heads, q_norm, &mut q[t * qd..][..qd]),
            (h.kv_heads, k_norm, &mut k[t * kvd..][..kvd]),
        ] {
            for j in 0..heads {
                let head = &mut buf[j * hd..][..hd];
                tmp.copy_from_slice(head);
                rmsnorm(dot, head, &tmp, weight, h.eps, &mut hw);
                rope(head, t, &inv_freq);
            }
        }
    }

    let kw: Vec<u64> = {
        let mut w = vec![0u64; n * kvd / 2];
        pack(&k, &mut w);
        w
    };
    let group = h.heads / h.kv_heads;
    let mut att = vec![0.0f32; n];
    let (mut aw, mut cw) = (vec![0u64; n.div_ceil(2)], vec![0u64; n.div_ceil(2)]);
    let mut qw = vec![0u64; qd / 2];
    let mut attn = vec![0.0f32; n * qd];
    for t in 0..n {
        pack(&q[t * qd..][..qd], &mut qw);
        for j in 0..h.heads {
            let kv = (j / group) * hd;
            let a = &mut att[..t + 1];
            for (s, score) in a.iter_mut().enumerate() {
                *score = dot(
                    &qw[j * hd / 2..][..hd / 2],
                    &kw[(s * kvd + kv) / 2..][..hd / 2],
                    hd,
                ) * scale;
            }
            softmax(a);
            pack(a, &mut aw);
            for i in 0..hd {
                let col = |s: usize| match s <= t {
                    true => v[s * kvd + kv + i].to_bits() as u64,
                    false => 0,
                };
                for (m, w) in cw[..(t + 1).div_ceil(2)].iter_mut().enumerate() {
                    *w = col(2 * m) | col(2 * m + 1) << 32;
                }
                attn[t * qd + j * hd + i] = dot(&aw, &cw, t + 1);
            }
        }
    }
    drop((q, k, v, kw));

    let mut o = vec![0.0f32; n * dim];
    claimed(1, &attn, &mut [&mut o]);
    drop(attn);
    for t in 0..n {
        let xt = &mut x[t * dim..][..dim];
        for (a, &b) in xt.iter_mut().zip(&o[t * dim..][..dim]) {
            *a += b;
        }
        rmsnorm(dot, &mut xn[t * dim..][..dim], xt, post_ln, h.eps, &mut xw);
    }
    let mut g = vec![0.0f32; n * h.hidden];
    let mut u = vec![0.0f32; n * h.hidden];
    claimed(2, &xn, &mut [&mut g, &mut u]);
    for (a, &b) in g.iter_mut().zip(&u) {
        *a = *a / (1.0 + libm::expf(-*a)) * b;
    }
    drop(u);
    claimed(3, &g, &mut [&mut o]);
    for (a, &b) in x.iter_mut().zip(&o) {
        *a += b;
    }
    x
}
