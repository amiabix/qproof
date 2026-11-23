use super::*;
use std::cell::Cell;
use std::vec::Vec;

std::thread_local! {
    /// Elements that took the FP32 fallback in this thread.
    pub static SLOW: Cell<usize> = const { Cell::new(0) };
    /// Elements that left the integer fast loop (then integer add or FP32 fallback).
    pub static EXITS: Cell<usize> = const { Cell::new(0) };
}

// NaN payload propagation varies between optimized ARM and zkVM operations.
// The layer rejects nonfinite outputs. Every non-NaN value, including signed
// zero and infinity, still has to match bit for bit.
fn result_bits(v: f32) -> u32 {
    if v.is_nan() { f32::NAN.to_bits() } else { v.to_bits() }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn lanes_give_exact_significand_products() {
    for chunk in 0..4096u32 {
        let y = [activation_word(chunk)];
        for f in 0..128u32 {
            let mut s = [0i64];
            SoftDotF8.block_sums(&[WEIGHT[f as usize]], &y, &mut s);
            assert_eq!(s[0], i64::from((128 | f) * chunk) << 10, "chunk {chunk} f {f}");
        }
    }
}

#[test]
fn bf16_projection_calls_dot_f8_once_per_row_and_position() {
    struct CountedDot(usize);
    impl DotF8 for CountedDot {
        fn block_sums(&mut self, x: &[u64], y: &[u64], sums: &mut [i64]) {
            self.0 += 1;
            SoftDotF8.block_sums(x, y, sums);
        }
    }

    let mut kernel = Kernel::new(CountedDot(0));
    let inputs = [1f32.to_bits(); 2 * LANES];
    let two_rows = 0x3f80u16.to_le_bytes().repeat(2 * LANES);
    let one_row = &two_rows[..2 * LANES];
    let mut first = [0.0; 4];
    let mut second = [0.0; 2];
    kernel.project_group_bf16(&inputs, LANES, &[&two_rows, one_row], &mut [&mut first, &mut second]);
    assert_eq!(kernel.dot.0, 2 * (2 + 1));
    assert_eq!(first, [LANES as f32; 4]);
    assert_eq!(second, [LANES as f32; 2]);
}

/// Weight and activation generators for one regime.
#[derive(Clone, Copy, Debug)]
enum Regime {
    /// Checkpoint-like weights, activations of order 1.
    Realistic,
    /// Any bits, finite or not.
    AnyBits,
    /// Any finite bits, subnormals included.
    AnyFinite,
    /// Few significant bits: exact ties and exact cancellation are frequent.
    Ties,
    /// Products near FLT_MAX: the accumulator overflows to +-inf and keeps going.
    Overflow,
    /// Tiny products: subnormal products and accumulators near the integer window edge.
    Tiny,
    /// Zeros of both signs mixed into realistic data.
    Zeros,
}

fn weight(r: &mut Rng, g: Regime) -> u16 {
    let any = r.next() as u16;
    let sign = (r.below(2) as u16) << 15;
    match g {
        Regime::Realistic => sign | ((112 + r.below(14) as u16) << 7) | (any & 127),
        Regime::AnyBits => any,
        Regime::AnyFinite => {
            if (any >> 7) & 255 == 255 { any & !(1 << 14) } else { any }
        }
        Regime::Ties => sign | ((120 + r.below(8) as u16) << 7) | ((any & 3) << 5),
        Regime::Overflow => sign | ((240 + r.below(15) as u16) << 7) | (any & 127),
        Regime::Tiny => sign | ((1 + r.below(30) as u16) << 7) | (any & 127),
        Regime::Zeros => {
            if r.below(4) == 0 { sign } else { sign | ((112 + r.below(14) as u16) << 7) | (any & 127) }
        }
    }
}

fn activation(r: &mut Rng, g: Regime) -> u32 {
    let any = r.next() as u32;
    let sign = (r.below(2) as u32) << 31;
    match g {
        Regime::Realistic | Regime::Zeros => {
            if matches!(g, Regime::Zeros) && r.below(4) == 0 {
                sign
            } else {
                sign | ((118 + r.below(14) as u32) << 23) | (any & 0x7f_ffff)
            }
        }
        Regime::AnyBits => any,
        Regime::AnyFinite => {
            if (any >> 23) & 255 == 255 { any & !(1 << 30) } else { any }
        }
        Regime::Ties => sign | ((124 + r.below(8) as u32) << 23) | ((any & 7) << 20),
        Regime::Overflow => sign | ((200 + r.below(54) as u32) << 23) | (any & 0x7f_ffff),
        Regime::Tiny => sign | ((r.below(60) as u32) << 23) | (any & 0x7f_ffff),
    }
}

fn check(seed: u64, g: Regime, positions: usize, rows: usize, cols: usize) -> (usize, usize) {
    let mut r = Rng(seed | 1);
    let matrix: Vec<u8> = (0..rows * cols).flat_map(|_| weight(&mut r, g).to_le_bytes()).collect();
    let x: Vec<u32> = (0..positions * cols).map(|_| activation(&mut r, g)).collect();
    let mut want = vec![0f32; positions * rows];
    reference(&x, cols, &matrix, &mut want);
    let mut k = Kernel::new(SoftDotF8);
    let before = SLOW.with(Cell::get);
    let mut got = vec![0f32; positions * rows];
    k.project(&Prepared::new(&x, cols), &matrix, &mut got);
    let slow = SLOW.with(Cell::get) - before;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(result_bits(*a), result_bits(*b), "{g:?} seed {seed} t {} r {}", i / rows, i % rows);
    }
    // The same kernel one position at a time.
    for t in 0..positions {
        let mut one = vec![0f32; rows];
        k.project(&Prepared::new(&x[t * cols..][..cols], cols), &matrix, &mut one);
        for (a, b) in one.iter().zip(&want[t * rows..][..rows]) {
            assert_eq!(result_bits(*a), result_bits(*b), "{g:?} per-position t {t}");
        }
    }
    (slow, positions * rows * cols)
}

#[test]
fn bit_identical_to_ordered_fp32_in_every_regime() {
    use Regime::*;
    let mut total = 0;
    for g in [Realistic, AnyBits, AnyFinite, Ties, Overflow, Tiny, Zeros] {
        for (i, &(t, rows, cols)) in [(1, 7, 1), (2, 5, 2), (3, 4, 7), (4, 6, 64), (2, 9, 129), (5, 8, 300)]
            .iter()
            .enumerate()
        {
            for s in 0..40u64 {
                total += check(s * 1000 + i as u64 * 7 + g as u64, g, t, rows, cols).1;
            }
        }
    }
    std::eprintln!("{total} MACs matched");
}

#[test]
fn realistic_rows_take_the_integer_path() {
    let (slow, macs) = check(42, Regime::Realistic, 3, 16, 2560);
    // One fallback per row start (+0 accumulator) plus rare binade crossings.
    assert!(slow * 20 < macs, "{slow} fallbacks of {macs}");
    std::eprintln!("realistic: {slow} fallbacks of {macs} MACs");
}

#[test]
fn ties_carries_and_binade_edges_exhaustively() {
    // Accumulator a (one row, two columns: w0*x0 = a exactly, then w1*x1 the term).
    let mut n = 0;
    let mut r = Rng(7);
    for ae in [33u32, 34, 60, 127, 200, 253, 254] {
        for am in [0u32, 1, 2, 3, 0x40_0000, 0x7f_fffe, 0x7f_ffff] {
            for gap in 0..=34u32 {
                for _ in 0..24 {
                    let a = (ae << 23) | am;
                    // Term: exponent ae - gap, few or many mantissa bits, either sign.
                    let te = ae.saturating_sub(gap).max(1);
                    let tm = match r.below(3) {
                        0 => (r.next() as u32) & 0x7f_ffff,
                        1 => 0x40_0000,
                        _ => ((r.next() as u32) & 0xff) << 15,
                    };
                    let tsign = (r.below(2) as u32) << 31;
                    // a = 1.0 (bf16 0x3f80) * a; term = bf16 2^(te-127) * x with x = 1.tm.
                    let wexp = te as i32 - 127 + 127;
                    if !(1..=254).contains(&wexp) {
                        continue;
                    }
                    let matrix: Vec<u8> = [0x3f80u16, (wexp as u16) << 7]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect();
                    let x = [a | (((r.below(2)) as u32) << 31), tsign | (127 << 23) | tm];
                    let (mut got, mut want) = ([0f32], [0f32]);
                    reference(&x, 2, &matrix, &mut want);
                    Kernel::new(SoftDotF8).project(&Prepared::new(&x, 2), &matrix, &mut got);
                    assert_eq!(got[0].to_bits(), want[0].to_bits(), "a {a:#x} gap {gap} x {x:x?}");
                    n += 1;
                }
            }
        }
    }
    std::eprintln!("{n} edge cases matched");
}

#[test]
fn infinite_accumulator_absorbs_finite_terms_and_nan_propagates() {
    // Two FLT_MAX-sized products overflow to +inf. The next terms (2^127 * 2^-1.., exponent
    // sum within the integer window's gap of an exponent-255 accumulator) must leave +inf
    // unchanged; a final -inf makes NaN.
    let w = |e: u16, s: u16| (s << 15) | (e << 7);
    let x = |e: u32, s: u32| (s << 31) | (e << 23) | 0x12345;
    for (tail_w, tail_x) in [(w(254, 0), x(126, 0)), (w(254, 1), x(120, 0)), (w(250, 0), x(100, 1))] {
        let mut row = vec![w(254, 0), w(254, 0)];
        let mut xs = vec![x(253, 0), x(253, 0)];
        for _ in 0..6 {
            row.push(tail_w);
            xs.push(tail_x);
        }
        let matrix: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (mut got, mut want) = ([0f32], [0f32]);
        reference(&xs, row.len(), &matrix, &mut want);
        Kernel::new(SoftDotF8).project(&Prepared::new(&xs, row.len()), &matrix, &mut got);
        assert_eq!(want[0], f32::INFINITY);
        assert_eq!(got[0].to_bits(), want[0].to_bits());
        // Append -inf: NaN, bit-identical.
        row.extend([w(254, 1), w(254, 1), w(254, 1)]);
        xs.extend([x(253, 0), x(253, 0), x(253, 0)]);
        let matrix: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        reference(&xs, row.len(), &matrix, &mut want);
        Kernel::new(SoftDotF8).project(&Prepared::new(&xs, row.len()), &matrix, &mut got);
        assert!(want[0].is_nan());
        assert_eq!(got[0].to_bits(), want[0].to_bits());
    }
}

/// Fast-loop exit rates on real projection samples (`PCI_SAMPLES=<dir with *.t39.m2.stdin>`).
#[test]
#[ignore]
fn exit_rates_on_real_samples() {
    let dir = std::env::var("PCI_SAMPLES").expect("PCI_SAMPLES");
    for p in ["q", "k", "v", "o", "gate", "up", "down"] {
        let b = std::fs::read(std::format!("{dir}/{p}.t39.m2.stdin")).unwrap();
        let b = &b[8..];
        let u = |i: usize| u32::from_le_bytes(b[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
        let (n, rows, cols) = (u(1), u(2), u(3));
        let x: Vec<u32> = b[24..24 + 4 * n * cols].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let m = &b[24 + 4 * n * cols..24 + 4 * n * cols + 2 * rows * cols];
        let (e0, s0) = (EXITS.with(Cell::get), SLOW.with(Cell::get));
        let mut out = vec![0f32; n * rows];
        Kernel::new(SoftDotF8).project(&Prepared::new(&x, cols), m, &mut out);
        let mut want = vec![0f32; n * rows];
        reference(&x, cols, m, &mut want);
        assert!(out.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
        let macs = n * rows * cols;
        let (e, s) = (EXITS.with(Cell::get) - e0, SLOW.with(Cell::get) - s0);
        std::eprintln!("{p}: {macs} MACs, fast-loop exits {e} ({:.2}%), FP32 fallbacks {s}", 100.0 * e as f64 / macs as f64);
    }
}

/// The Mac BF16 lanes against `pci_layer::project_bf16`, batched and per position.
fn check_bf16(seed: u64, g: Regime, positions: usize, rows: usize, cols: usize) -> usize {
    let mut r = Rng(seed | 1);
    let matrix: Vec<u8> = (0..rows * cols).flat_map(|_| weight(&mut r, g).to_le_bytes()).collect();
    let x: Vec<u32> = (0..positions * cols).map(|_| activation(&mut r, g)).collect();
    let mut want = vec![0f32; positions * rows];
    for (xt, ot) in x.chunks_exact(cols).zip(want.chunks_exact_mut(rows)) {
        pci_layer::project_bf16_bits(xt, &matrix, ot);
    }
    let mut k = Kernel::new(SoftDotF8);
    let before = SLOW.with(Cell::get);
    let mut got = vec![0f32; positions * rows];
    k.project_group_bf16(&x, cols, &[&matrix], &mut [&mut got]);
    let slow = SLOW.with(Cell::get) - before;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(result_bits(*a), result_bits(*b), "{g:?} seed {seed} t {} r {}", i / rows, i % rows);
    }
    for t in 0..positions {
        let mut one = vec![0f32; rows];
        k.project_group_bf16(&x[t * cols..][..cols], cols, &[&matrix], &mut [&mut one]);
        for (a, b) in one.iter().zip(&want[t * rows..][..rows]) {
            assert_eq!(result_bits(*a), result_bits(*b), "{g:?} per-position t {t}");
        }
    }
    slow
}

#[test]
fn bf16_lanes_bit_identical_to_mac_reference_in_every_regime() {
    use Regime::*;
    for g in [Realistic, AnyBits, AnyFinite, Ties, Overflow, Tiny, Zeros] {
        for (i, &(t, rows, cols)) in [(1, 7, 32), (2, 5, 64), (3, 4, 96), (2, 6, 320)].iter().enumerate() {
            for s in 0..40u64 {
                check_bf16(s * 1000 + i as u64 * 7 + g as u64, g, t, rows, cols);
            }
        }
    }
}

#[test]
fn bf16_realistic_rows_take_the_integer_path() {
    let slow = check_bf16(42, Regime::Realistic, 3, 16, 2560);
    // One fallback per lane start (+0 accumulator) plus rare binade crossings.
    let macs = 3 * 16 * 2560;
    assert!(slow * 20 < macs, "{slow} fallbacks of {macs}");
}
