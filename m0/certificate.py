"""Exact fixed-point certificate for the 32-lane pinned BF16 dot product."""

import math
from fractions import Fraction

import numpy as np


GOLDILOCKS = (1 << 64) - (1 << 32) + 1
MIN_NORMAL = Fraction(1, 1 << 126)
MIN_SUBNORMAL = MIN_NORMAL / 128


def _pow2(exponent):
    return Fraction(2) ** exponent


def _magnitude_cell(y):
    """Adjacent midpoints for a positive finite BF16 magnitude."""
    exponent = math.frexp(float(y))[1] - 1  # y itself is exactly representable
    up = max(MIN_SUBNORMAL, _pow2(exponent - 7))
    down = up / 2 if y > MIN_NORMAL and y == _pow2(exponent) else up
    return y - down / 2, y + up / 2


def lower_boundary(y):
    """Exact lower midpoint for numeric BF16 comparisons (the zeros tie)."""
    y = Fraction(float(y))
    if not y:
        return -MIN_SUBNORMAL / 2
    lo, hi = _magnitude_cell(abs(y))
    return lo if y > 0 else -hi


def _nearest_magnitude(a):
    # a is positive and dyadic. Its denominator is a power of two.
    exponent = a.numerator.bit_length() - a.denominator.bit_length()
    step = max(MIN_SUBNORMAL, _pow2(exponent - 7))
    return round(a / step) * step


def _rows(a):
    a = np.asarray(a)
    if a.ndim != 2 or a.dtype != np.float32:
        raise ValueError("expected a float32 row matrix")
    if not np.isfinite(a).all():
        raise ValueError("E1 requires finite inputs")
    if np.any(a.view(np.uint32) & 0xffff):
        raise ValueError("expected BF16-exact inputs")
    return a


def _encode(a, signed_bits, magnitude_bits):
    # Use binary64 for the exact power-of-two shifts, even for BF16 extremes.
    a = a.astype(np.float64)
    peak = np.max(np.abs(a), axis=1)
    _, exponent = np.frexp(peak)
    s = np.where(peak == 0, 0, signed_bits - exponent)
    b = np.where(peak == 0, 0, magnitude_bits - exponent)
    shifted = np.ldexp(a, s[:, None])
    q = shifted.astype(np.int64)
    flags = (q != shifted).astype(np.int64)
    up = np.ceil(np.ldexp(np.abs(a), b[:, None])).astype(np.int64)
    return q, flags, up, [_pow2(-int(v)) for v in s], [_pow2(-int(v)) for v in b]


def assess(w, x, maxima=None):
    """Yield (weight row, activation row, status, signed y, signed upper bound).

    Accumulate absolute integer maxima across calls in the supplied dictionary.
    E1 input errors reject the entire call before any outputs are yielded.
    """
    w, x = _rows(w), _rows(x)
    k = w.shape[1]
    if x.shape[1] != k or not 0 < k <= 9728 or k % 32:
        raise ValueError("expected equal widths divisible by 32 in [32, 9728]")

    n = k // 32
    depth = n + 4
    counts = n - np.maximum(np.arange(k, dtype=np.int64) // 32, 1) + 5
    gamma = Fraction(1, (1 << 24) - depth)
    # (K+31)*(1+M*gamma) <= K+32 over the supported widths.
    eta = Fraction(k + 32, 1 << 150)

    # Absolute term-sum bounds also protect every intermediate int64 MAC.
    limits = dict(S=k << 47, Bs=(k * depth) << 34, E1=k << 19, E2=k << 15, E3=k)
    if any(2 * limit >= GOLDILOCKS for limit in limits.values()):
        raise ArithmeticError("integer bound exceeds the signed Goldilocks range")

    W, tw, wu, sw, bw = _encode(w, 20, 15)
    X, tx, xu, sx, bx = _encode(x, 27, 19)
    S = W @ X.T
    Bs = (wu * counts) @ xu.T
    E1, E2, E3 = tw @ xu.T, wu @ tx.T, tw @ tx.T
    for name, matrix in zip(("S", "Bs", "E1", "E2", "E3"), (S, Bs, E1, E2, E3)):
        peak = max(abs(int(matrix.min(initial=0))), int(matrix.max(initial=0)))
        if peak > limits[name]:
            raise ArithmeticError(f"{name} exceeds its integer bound")
        if maxima is not None:
            maxima[name] = max(maxima.get(name, 0), peak)

    for r in range(len(w)):
        for t in range(len(x)):
            weighted = int(Bs[r, t]) * bw[r] * bx[t]
            if (1 + depth * gamma) * weighted / 5 + eta >= 1 << 127:
                yield r, t, "E2", None, None
                continue
            sf = int(S[r, t]) * sw[r] * sx[t]
            error = (int(E1[r, t]) * sw[r] * bx[t]
                     + int(E2[r, t]) * bw[r] * sx[t]
                     + int(E3[r, t]) * sw[r] * sx[t])
            total = gamma * weighted + error + eta
            upper = sf + total
            a = abs(sf)
            if a - total <= MIN_NORMAL:
                yield r, t, "E3", None, upper
                continue
            y = _nearest_magnitude(a)
            lo, hi = _magnitude_cell(y)
            if lo < a - total and a + total < hi:
                yield r, t, "certified", float(y if sf > 0 else -y), upper
            else:
                yield r, t, "E4", None, upper
