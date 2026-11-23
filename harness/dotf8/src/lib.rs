//! Canonical batched lossless BF16 x FP32 projection on the `dot_f8` precompile.
//!
//! Function (identical to `pci_layer::project` for every input, bit for bit):
//! `out[t][r] = (((+0 + w[r][0]*x[t][0]) + w[r][1]*x[t][1]) + ...)`, where `w` is the
//! exactly widened BF16 weight, `x` an FP32 activation, and every product and every
//! addition is rounded to FP32 (RNE) in increasing column order. No fusion.
//!
//! How:
//! - The exact integer product of the two significands comes from `dot_f8` on FP8 E4M3
//!   lanes: the 8-bit weight significand is two nibbles, the 24-bit activation
//!   significand is two 12-bit chunks of three nibbles; every lane holds `nibble * 2^k`
//!   exactly, so each 8-lane block sum is `wm * chunk * 2^10` exactly (units 2^-18).
//! - Sign and biased exponent of each operand live in one "metadata" word
//!   (`sign << 63 | exponent`); adding a weight word and an activation word gives the
//!   product's sign (bit 63, carry-free XOR) and exponent sum in one addition.
//!   Zero, subnormal, infinite and NaN operands carry a poison exponent that forces the
//!   exact fallback.
//! - Each weight row is encoded once and reused for every position ("batched");
//!   activations are prepared once per group of matrices sharing an input.
//! - [`Kernel::project_group_bf16`] is the pinned Mac HF BF16 function instead (identical to
//!   `pci_layer::project_bf16`): `x` rounded to BF16, 32 lanes each accumulating
//!   `fma(w, x, acc)` from +0 over columns `l, l + 32, ...`, the lanes reduced at offsets
//!   16, 8, 4 then `(0 + 1) + (2 + 3)`, the sum rounded to BF16. A BF16 x BF16 product is
//!   exact in 24 bits, so the integer path below (product rounded once, then the sum once)
//!   is the FMA; only the fallback differs (`fmaf`).
//! - The accumulator is kept as magnitude bits plus a sign mask. A term is added in
//!   integer arithmetic only when the result provably stays in the accumulator's binade
//!   and the accumulator is finite and normal (exponent field in [GAP+1, 254]; the 254
//!   bound is the overflow guard: an infinite or NaN accumulator never takes the integer
//!   path). Anything else is one ordinary FP32 `acc + w * x`.
#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

use alloc::vec;
use alloc::vec::Vec;

/// Exact block sums of two E4M3 vectors (OCP FP8, values in units of 2^-9):
/// `sums[i] = sum_{j<8} x[8i+j] * y[8i+j]` in units of 2^-18, for `i < sums.len()`;
/// `x` and `y` hold `sums.len()` 64-bit words. This is `dot_f8` with both formats
/// E4M3 and block 8.
pub trait DotF8 {
    fn block_sums(&mut self, x: &[u64], y: &[u64], sums: &mut [i64]);
}

/// Software `dot_f8` (native runs and tests).
pub struct SoftDotF8;

/// E4M3 code to value in units of 2^-9. The encoders below never emit NaN codes.
const fn e4m3(c: u8) -> i64 {
    let (e, m) = ((c >> 3) & 15, (c & 7) as i64);
    let mag = if e == 0 { m } else { (8 + m) << (e - 1) };
    if c & 0x80 != 0 { -mag } else { mag }
}

impl DotF8 for SoftDotF8 {
    fn block_sums(&mut self, x: &[u64], y: &[u64], sums: &mut [i64]) {
        assert!(x.len() == sums.len() && y.len() == sums.len());
        for ((&a, &b), s) in x.iter().zip(y).zip(sums.iter_mut()) {
            *s = (0..8)
                .map(|k| e4m3((a >> (8 * k)) as u8) * e4m3((b >> (8 * k)) as u8))
                .sum();
        }
    }
}

/// E4M3 code of `n * 2^power` for a nibble `n` (exact: at most 4 significant bits).
const fn code(n: u32, power: i32) -> u64 {
    if n == 0 {
        return 0;
    }
    let log = 31 - n.leading_zeros();
    ((((log as i32) + power + 7) as u64) << 3) | (((n << 3) >> log) & 7) as u64
}

/// Weight word for an 8-bit significand: lanes (low nibble * 2^-4, high nibble) at bytes
/// 0-1, 2-3 and 4-5, one pair per activation nibble.
const fn weight_word(m: u32) -> u64 {
    let pair = code(m & 15, -4) | (code(m >> 4, 0) << 8);
    pair | (pair << 16) | (pair << 32)
}

/// Weight words of the 128 normal BF16 significands `128 | f`.
static WEIGHT: [u64; 128] = {
    let mut t = [0; 128];
    let mut f = 0;
    while f < 128 {
        t[f] = weight_word(128 | f as u32);
        f += 1;
    }
    t
};

/// Activation word for a 12-bit chunk: nibbles at 2^-4, 2^0, 2^4, each in both lanes of
/// its byte pair.
const fn activation_word(chunk: u32) -> u64 {
    let w = code(chunk & 15, -4) | (code((chunk >> 4) & 15, 0) << 16) | (code(chunk >> 8, 4) << 32);
    w | (w << 8)
}

/// Exponent added to a zero, subnormal, infinite or NaN operand's metadata. Any product
/// involving it has an exponent sum far above every finite accumulator's window.
const POISON: u64 = 1 << 20;
/// Largest exponent gap between accumulator and product taken by the integer path.
const GAP: u64 = 32;

const fn metadata(sign: u32, exponent: u32) -> u64 {
    let poison = if exponent == 0 || exponent == 255 { POISON } else { 0 };
    ((sign as u64) << 63) | (exponent as u64 + poison)
}

/// Weight metadata indexed by the BF16 sign and exponent (`bits >> 7`).
static WEIGHT_META: [u64; 512] = {
    let mut t = [0; 512];
    let mut i = 0;
    while i < 512 {
        t[i] = metadata(i as u32 >> 8, i as u32 & 255);
        i += 1;
    }
    t
};

/// BF16 bits of column `c` of a raw little-endian row.
#[inline(always)]
fn bf16_at(row: &[u8], c: usize) -> u32 {
    let p = &row[2 * c..2 * c + 2];
    // SAFETY: p holds 2 bytes; the read is unaligned-safe.
    u32::from(u16::from_le(unsafe { core::ptr::read_unaligned(p.as_ptr().cast::<u16>()) }))
}

/// FP32 activations of `positions x cols`, prepared once and shared by every matrix
/// that reads them (q/k/v, gate/up).
pub struct Prepared<'a> {
    x: &'a [u32],
    cols: usize,
    positions: usize,
    lo: Vec<u64>,
    hi: Vec<u64>,
    meta: Vec<u64>,
}

impl<'a> Prepared<'a> {
    /// `x`: FP32 bit patterns, row-major `[positions][cols]`.
    pub fn new(x: &'a [u32], cols: usize) -> Self {
        Self::prepare::<true>(x, cols)
    }

    fn prepare<const LOW_CHUNK: bool>(x: &'a [u32], cols: usize) -> Self {
        assert!(cols > 0 && x.len() % cols == 0, "activation shape");
        let mut lo = Vec::with_capacity(if LOW_CHUNK { x.len() } else { 0 });
        let mut hi = Vec::with_capacity(x.len());
        let mut meta = Vec::with_capacity(x.len());
        for &v in x {
            let mant = (v & 0x7f_ffff) | 0x80_0000;
            if LOW_CHUNK {
                lo.push(activation_word(mant & 4095));
            }
            hi.push(activation_word(mant >> 12));
            meta.push(metadata(v >> 31, (v >> 23) & 255));
        }
        Prepared { x, cols, positions: x.len() / cols, lo, hi, meta }
    }

    pub fn positions(&self) -> usize {
        self.positions
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// The kernel: a `dot_f8` backend and per-row scratch, reused across matrices.
pub struct Kernel<D: DotF8> {
    dot: D,
    wd: Vec<u64>,
    wmeta: Vec<u64>,
    s_lo: Vec<i64>,
    s_hi: Vec<i64>,
}

impl<D: DotF8> Kernel<D> {
    pub fn new(dot: D) -> Self {
        Kernel { dot, wd: vec![], wmeta: vec![], s_lo: vec![], s_hi: vec![] }
    }

    /// `out[t * rows + r] = row r of the raw BF16 matrix . x[t]` for every prepared
    /// position `t` (the layout of a `[positions][rows]` activation), written in place.
    pub fn project(&mut self, x: &Prepared, matrix: &[u8], out: &mut [f32]) {
        let cols = x.cols;
        assert!(matrix.len() % (2 * cols) == 0, "matrix shape");
        let rows = matrix.len() / (2 * cols);
        assert_eq!(out.len(), x.positions * rows, "output shape");
        for v in [&mut self.wd, &mut self.wmeta] {
            v.resize(cols, 0);
        }
        for v in [&mut self.s_lo, &mut self.s_hi] {
            v.resize(cols, 0);
        }
        for (r, row) in matrix.chunks_exact(2 * cols).enumerate() {
            encode_row(row, &mut self.wd, &mut self.wmeta);
            for t in 0..x.positions {
                let at = t * cols..(t + 1) * cols;
                self.dot.block_sums(&self.wd, &x.lo[at.clone()], &mut self.s_lo);
                self.dot.block_sums(&self.wd, &x.hi[at.clone()], &mut self.s_hi);
                out[t * rows + r] = accumulate::<1, false>(
                    0,
                    &self.s_lo,
                    &self.s_hi,
                    &self.wmeta,
                    &x.meta[at.clone()],
                    row,
                    &x.x[at],
                );
            }
        }
    }

    /// As [`Kernel::project`], with the Mac BF16 lanes; `x` must hold BF16 values.
    fn project_lanes(&mut self, x: &Prepared, matrix: &[u8], out: &mut [f32]) {
        let cols = x.cols;
        assert!(cols % LANES == 0 && matrix.len() % (2 * cols) == 0, "matrix shape");
        let rows = matrix.len() / (2 * cols);
        assert_eq!(out.len(), x.positions * rows, "output shape");
        for v in [&mut self.wd, &mut self.wmeta] {
            v.resize(cols, 0);
        }
        self.s_hi.resize(cols, 0);
        for (r, row) in matrix.chunks_exact(2 * cols).enumerate() {
            encode_row(row, &mut self.wd, &mut self.wmeta);
            for t in 0..x.positions {
                let at = t * cols..(t + 1) * cols;
                self.dot.block_sums(&self.wd, &x.hi[at.clone()], &mut self.s_hi);
                let mut acc = [0f32; LANES];
                for (l, a) in acc.iter_mut().enumerate() {
                    *a = accumulate::<LANES, true>(
                        l,
                        &[],
                        &self.s_hi,
                        &self.wmeta,
                        &x.meta[at.clone()],
                        row,
                        &x.x[at.clone()],
                    );
                }
                out[t * rows + r] = qwen_common::finish_bf16_dot(acc);
            }
        }
    }

    /// The Mac BF16 function on every matrix of a group sharing one input: `x` (FP32
    /// bits) is rounded to BF16 and prepared once.
    pub fn project_group_bf16(&mut self, x: &[u32], cols: usize, matrices: &[&[u8]], outs: &mut [&mut [f32]]) {
        assert_eq!(matrices.len(), outs.len(), "group shape");
        let xb: Vec<u32> = x.iter().map(|&v| qwen_common::bf16_value(f32::from_bits(v)).to_bits()).collect();
        let p = Prepared::prepare::<false>(&xb, cols);
        for (m, o) in matrices.iter().zip(outs.iter_mut()) {
            self.project_lanes(&p, m, o);
        }
    }

    /// Every matrix of a group sharing one input: prepare `x` once, then project each
    /// matrix straight into its output.
    pub fn project_group(&mut self, x: &[u32], cols: usize, matrices: &[&[u8]], outs: &mut [&mut [f32]]) {
        assert_eq!(matrices.len(), outs.len(), "group shape");
        let p = Prepared::new(x, cols);
        for (m, o) in matrices.iter().zip(outs.iter_mut()) {
            self.project(&p, m, o);
        }
    }
}

use qwen_common::BF16_DOT_LANES as LANES;

#[inline(always)]
fn encode_row(row: &[u8], wd: &mut [u64], wmeta: &mut [u64]) {
    let n = wd.len().min(wmeta.len()).min(row.len() / 2);
    for c in 0..n {
        let w = bf16_at(row, c);
        wd[c] = WEIGHT[(w & 127) as usize];
        wmeta[c] = WEIGHT_META[(w >> 7) as usize];
    }
}

#[inline(always)]
fn product_significand<const BF16: bool>(mag: u64, top: u64) -> u64 {
    let shift = 7 + top;
    if BF16 {
        // Two 8-bit significands multiply exactly in FP32.
        mag >> shift
    } else {
        (mag + 63 + (top << 6) + ((mag >> shift) & 1)) >> shift
    }
}

/// One ordered row-by-position dot product from the block sums and metadata, over
/// columns `start, start + STEP, ...`; `BF16` uses only the high chunk and a fused fallback.
#[inline(never)]
fn accumulate<const STEP: usize, const BF16: bool>(
    start: usize,
    s_lo: &[i64],
    s_hi: &[i64],
    wmeta: &[u64],
    xmeta: &[u64],
    row: &[u8],
    x: &[u32],
) -> f32 {
    let n = s_hi.len();
    let (wmeta, xmeta, x) = (&wmeta[..n], &xmeta[..n], &x[..n]);
    let magnitude = |c: usize| {
        if BF16 {
            // BF16 inputs have no low chunk.
            (s_hi[c] << 2) as u64
        } else {
            ((s_lo[c] + (s_hi[c] << 12)) >> 10) as u64
        }
    };
    // Accumulator: FP32 magnitude bits and a sign mask (0 or all ones). Starts at +0.
    let (mut am, mut sa) = (0u64, 0u64);
    let mut c = start;
    while c < n {
        let ae = am >> 23;
        // Integer path only while the accumulator is normal, finite and at least 2^-94:
        // then a product with gap <= GAP is a normal FP32 number (exponent >= 1) and,
        // because the accumulator's exponent is at most 254, never infinite.
        if ae.wrapping_sub(GAP + 1) <= 254 - (GAP + 1) {
            let base = ae + 127;
            let lo = am & 0x7f80_0000;
            while c < n {
                // Exact significand product wm * xm in [2^30, 2^32) for normal operands.
                let mag = magnitude(c);
                let meta = wmeta[c].wrapping_add(xmeta[c]);
                let top = mag >> 31;
                // Product exponent field (before a rounding carry) is we + xe - 127 + top;
                // gap = accumulator exponent - product exponent.
                let gap = base.wrapping_sub(meta & 0xffff_ffff).wrapping_sub(top);
                if gap.wrapping_sub(1) >= GAP {
                    break;
                }
                // Product rounded to 24 bits, ties to even (a carry to 2^24 keeps its value).
                let s = product_significand::<BF16>(mag, top);
                // That term in units of the accumulator's ulp, rounded to nearest with ties
                // to the even result: the result's parity is (am ^ whole) & 1.
                let whole = s >> gap;
                let inc = if BF16 && gap <= 8 {
                    // A normalized BF16 product has at least eight trailing zero bits.
                    whole
                } else {
                    (s + (1 << (gap - 1)) - 1 + ((am ^ whole) & 1)) >> gap
                };
                let dm = ((meta as i64 >> 63) as u64) ^ sa;
                let cand = am.wrapping_add((inc ^ dm).wrapping_sub(dm));
                // Same binade with a nonzero fraction: the exact sum lies strictly inside
                // the binade, so rounding at the accumulator's ulp is the FP32 rounding.
                if cand.wrapping_sub(lo).wrapping_sub(1) >= 0x7f_ffff {
                    break;
                }
                am = cand;
                c += STEP;
            }
            if c >= n {
                break;
            }
        }
        // Binade change, cancellation, a larger term or a zero accumulator with normal
        // operands: one exact integer addition of the two FP32 values.
        #[cfg(test)]
        tests::EXITS.with(|k| k.set(k.get() + 1));
        let mag = magnitude(c);
        let meta = wmeta[c].wrapping_add(xmeta[c]);
        if let Some((m, s)) = add_normal::<BF16>(am, sa, (meta as i64 >> 63) as u64, mag, meta & 0xffff_ffff) {
            (am, sa) = (m, s);
            c += STEP;
            continue;
        }
        // Exact fallback: one ordinary FP32 multiply and add.
        #[cfg(test)]
        tests::SLOW.with(|k| k.set(k.get() + 1));
        let acc = f32::from_bits((sa as u32 & 0x8000_0000) | am as u32);
        let w = f32::from_bits(bf16_at(row, c) << 16);
        let v = f32::from_bits(x[c]);
        let r = if BF16 { libm::fmaf(w, v, acc) } else { acc + w * v }.to_bits();
        am = u64::from(r & 0x7fff_ffff);
        sa = if r >> 31 != 0 { !0 } else { 0 };
        c += STEP;
    }
    f32::from_bits((sa as u32 & 0x8000_0000) | am as u32)
}

/// `acc + product` rounded once to FP32 (RNE), in integers, for an accumulator that is +0
/// or normal and a product of two normal operands that is itself a normal FP32 number;
/// `None` (use the FP32 fallback) otherwise. The accumulator is magnitude bits `am` and
/// sign mask `sa`; the product is the exact significand product `mag` (in [2^30, 2^32))
/// with sign mask `sp` and operand exponent sum `esum`.
#[inline(never)]
fn add_normal<const BF16: bool>(am: u64, sa: u64, sp: u64, mag: u64, esum: u64) -> Option<(u64, u64)> {
    // Product: 24-bit RNE significand `s` (2^24 after a carry) at exponent field `be`.
    let top = mag >> 31;
    let be = (esum + top).wrapping_sub(127);
    if be.wrapping_sub(1) >= 254 {
        return None; // poisoned operand, subnormal or infinite product
    }
    let s = product_significand::<BF16>(mag, top);
    let ae = am >> 23;
    if am == 0 {
        // +0 + p = p (the product is nonzero). A carry to field 255 is the correct
        // overflow to infinity.
        return Some((((be - 1) << 23) + s, sp));
    }
    if ae == 0 || ae >= 255 {
        return None; // subnormal, infinite or NaN accumulator
    }
    let a_sig = (am & 0x7f_ffff) | 0x80_0000;
    let (e1, s1, sg1, e2, s2) = if ae >= be { (ae, a_sig, sa, be, s) } else { (be, s, sp, ae, a_sig) };
    let d = e1 - e2;
    if d > 38 {
        return None;
    }
    // Both significands are below 2^25; aligned at bit 38 their exact sum is below 2^64.
    let (x1, x2) = (s1 << 38, s2 << (38 - d));
    let (r, sign) = if sa == sp {
        (x1 + x2, sg1)
    } else if x1 >= x2 {
        (x1 - x2, sg1)
    } else {
        (x2 - x1, !sg1)
    };
    if r == 0 {
        return Some((0, 0)); // exact cancellation is +0 under RNE
    }
    let p = 63 - u64::from(r.leading_zeros());
    // A 24-bit significand at bit 61 has the field e1.
    let ef = (e1 + p).wrapping_sub(61);
    if ef.wrapping_sub(1) >= 254 {
        return None; // subnormal result (or below)
    }
    let sig = if p > 23 {
        let sh = p - 23;
        (r + (1 << (sh - 1)) - 1 + ((r >> sh) & 1)) >> sh
    } else {
        r << (23 - p)
    };
    // A rounding carry moves into the exponent; from field 254 that is infinity.
    Some((((ef - 1) << 23) + sig, sign))
}

/// The reference function on the same layout: plain ordered FP32 from +0.
pub fn reference(x: &[u32], cols: usize, matrix: &[u8], out: &mut [f32]) {
    let rows = matrix.len() / (2 * cols);
    assert_eq!(matrix.len(), 2 * rows * cols, "matrix shape");
    assert_eq!(out.len() * cols, x.len() * rows, "output shape");
    for (t, xt) in x.chunks_exact(cols).enumerate() {
        for (r, row) in matrix.chunks_exact(2 * cols).enumerate() {
            let mut acc = 0.0f32;
            for (p, &v) in row.chunks_exact(2).zip(xt) {
                let w = f32::from_bits(u32::from(u16::from_le_bytes([p[0], p[1]])) << 16);
                acc += w * f32::from_bits(v);
            }
            out[t * rows + r] = acc;
        }
    }
}

#[cfg(test)]
mod tests;
