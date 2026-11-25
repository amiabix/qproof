# M0 reference for QPROOF_LAYERED_PLAN.md sections 3.1 and 3.3.
# The certificate is the integer routine the guest runs (no floats, no rationals); the oracle is an
# exact model of the pinned kernel (fma from +0 in 32 lanes, 31 reduction adds, BF16 RNE, with
# subnormals). Any certified-but-wrong output blocks the lemma.
# Usage: python3 -I m0_certificate_reference.py
import numpy as np

SH = 400                      # oracle fixed point: integer values at 2^-SH (covers every BF16 product)
U_DEN = 1 << 24               # u = 1/U_DEN
LIMIT = 127                   # guest works in u128: any intermediate wider than this is ineligible

def bf16(a):
    b = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7fff + ((b >> 16) & 1)) >> 16) << 16
    return (b.astype(np.uint32) >> 16).astype(np.uint16)          # BF16 bit patterns

def split(bits):
    """BF16 bits -> (sign, M, e) with |value| = M * 2^e, M an integer < 256. Finite inputs only."""
    b = bits.astype(np.int64); s = b >> 15; E = (b >> 7) & 255; f = b & 127
    assert not (E == 255).any(), "non-finite input (E1)"
    M = np.where(E == 0, f, f + 128); e = np.where(E == 0, -133, E - 134)
    return s, M, e

# ---- exact oracle (pinned kernel) --------------------------------------------------------------
def rne_shift(mag, sh):
    if sh <= 0: return mag << -sh
    q, r = mag >> sh, mag & ((1 << sh) - 1); h = 1 << (sh - 1)
    if r > h or (r == h and q & 1): q += 1
    return q << sh

def round_to(v, mant_bits, min_exp):          # v at 2^-SH; RNE to a binary format (with subnormals)
    if v == 0: return 0
    mag = abs(v); E = mag.bit_length() - 1 - SH
    keep_lsb = max(E - (mant_bits - 1), min_exp)      # exponent of the last kept bit
    r = rne_shift(mag, keep_lsb + SH)
    assert (r.bit_length() - 1 - SH) <= 127, "overflow in oracle"
    return r if v > 0 else -r

fl32 = lambda v: round_to(v, 24, -149)
fl_bf16 = lambda v: round_to(v, 8, -133)

def oracle(wb, xb):           # wb: R x K bits, xb: T x K bits -> R x T exact values at 2^-SH
    sw, Mw, ew = split(wb); sx, Mx, ex = split(xb)
    R, K = wb.shape; T = xb.shape[0]; out = [[0] * T for _ in range(R)]
    for r in range(R):
        for t in range(T):
            lanes = [0] * 32                                    # explicit +0
            for c in range(K):
                p = int(Mw[r, c]) * int(Mx[t, c]) << (int(ew[r, c]) + int(ex[t, c]) + SH)
                if sw[r, c] ^ sx[t, c]: p = -p
                lanes[c % 32] = fl32(lanes[c % 32] + p)          # fma: one rounding
            for off in (16, 8, 4):                              # 28 adds
                for i in range(off): lanes[i] = fl32(lanes[i] + lanes[i + off])
            s = fl32(fl32(lanes[0] + lanes[1]) + fl32(lanes[2] + lanes[3]))   # 3 adds
            out[r][t] = fl_bf16(s)
    return out

# ---- encodings (3.1), all derived from BF16 bits -----------------------------------------------
def scales(M, e, bits):       # per row/position: scale so that |value| * 2^s < 2^bits
    lead = np.where(M > 0, np.floor(np.log2(np.maximum(M, 1))).astype(np.int64) + e, -10**6)
    maximum = lead.max(axis=1)
    return np.where(maximum == -10**6, 0, bits - 1 - maximum)

def enc_trunc(s, M, e, sc):   # trunc(value * 2^sc) and the dropped-bits flag
    sh = e + sc[:, None]; out = np.zeros_like(M); flag = np.zeros_like(M)
    pos = sh >= 0; out[pos] = M[pos] << sh[pos]
    neg = ~pos; k = np.minimum(-sh[neg], 62)
    out[neg] = M[neg] >> k; flag[neg] = ((M[neg] & ((1 << k) - 1)) != 0) | (-sh[neg] > 62) & (M[neg] != 0)
    return np.where(s == 1, -out, out), flag.astype(np.int64)

def enc_up(M, e, sc):         # ceil(|value| * 2^sc)
    sh = e + sc[:, None]; out = np.zeros_like(M)
    pos = sh >= 0; out[pos] = M[pos] << sh[pos]
    neg = ~pos; k = np.minimum(-sh[neg], 62)
    out[neg] = (M[neg] + (1 << k) - 1) >> k
    return out

def encode(wb, xb, weight_hints=None):
    sw, Mw, ew = split(wb); sx, Mx, ex = split(xb); K = wb.shape[1]; n = K // 32
    sig, bet = (scales(Mw, ew, 20), scales(Mw, ew, 15)) if weight_hints is None else weight_hints
    tau, chi = scales(Mx, ex, 27), scales(Mx, ex, 19)          # activation scales: deterministic
    # Validate host hints before a fixed-width shift can discard high bits.
    for hint, ceiling in ((sig, (1 << 20) - 1), (bet, 1 << 15)):
        hint = np.asarray(hint)
        if hint.shape != (wb.shape[0],) or hint.dtype != np.int64:
            raise ValueError("invalid weight-scale encoding")
        if np.any(hint < -512) or np.any(hint > 512):
            raise ValueError("weight scale outside supported exponent range")
        shift = ew + hint[:, None]
        # For negative shifts even ceil(M / 2**k) fits, since M <= 255.
        allowed = ceiling >> np.clip(shift, 0, ceiling.bit_length())
        if np.any(Mw > allowed):
            raise ValueError("weight scale exceeds encoding bound")
    W, tW = enc_trunc(sw, Mw, ew, sig); Wu = enc_up(Mw, ew, bet)
    X, tX = enc_trunc(sx, Mx, ex, tau); Xu = enc_up(Mx, ex, chi)
    assert np.abs(W).max() < 2**20 and Wu.max() <= 2**15 and np.abs(X).max() < 2**27 and Xu.max() <= 2**19
    m = (n - np.maximum(np.arange(K) // 32, 1) + 5).astype(np.int64)
    return dict(W=W, tW=tW, Wu=Wu, X=X, tX=tX, Xu=Xu, m=m, sig=sig, bet=bet, tau=tau, chi=chi, K=K, M=n + 4)

def host_sums(en):            # exactly four products
    return (en['W'] @ en['X'].T, en['Wu'] @ (en['Xu'] * en['m']).T,
            en['tW'] @ en['Xu'].T, en['Wu'] @ en['tX'].T)

# ---- certificate (3.3), integer routine --------------------------------------------------------
class Ineligible(Exception): pass

def scaled_up(v, num_shift, den):     # ceil(v * 2^num_shift / den), v >= 0, exact, u128-bounded
    if num_shift >= 0:
        if v.bit_length() + num_shift > LIMIT: raise Ineligible   # guest cannot represent it
        return -((-(v << num_shift)) // den)
    if den.bit_length() - num_shift > LIMIT: return 1 if v else 0     # tiny term: one unit covers it
    return -((-v) // (den << -num_shift))

def certify(S, Bs, E1, E2, sig, bet, tau, chi, K, M):
    """Returns (certified, value at 2^-Z, Z) or (False, None, None). Pure integers."""
    Z = sig + tau; D = U_DEN - M                       # gamma = 1/D exactly
    # E2: (1+M gamma) * Bs * 2^-(bet+chi) / 5 < 2^126, i.e. Bs * 2^24 < 5 * D * 2^(126+bet+chi).
    # The eta term is below 2^-100 and fits in the slack between 2^126 and 2^127.
    e2 = 126 + bet + chi
    lhs, rhs = (Bs * U_DEN, (5 * D) << e2) if e2 >= 0 else ((Bs * U_DEN) << -e2, 5 * D)
    if lhs >= rhs: return False, None, None
    try:
        tot = (scaled_up(Bs, Z - bet - chi, D)          # gamma * Bsum
               + scaled_up(E1, tau - chi, 1)            # 2^-(sig+chi) E1
               + scaled_up(E2, sig - bet, 1)            # 2^-(bet+tau) E2
               + scaled_up((K + 31) * U_DEN, Z - 150, D))   # B_eta = (K+31) 2^-150 (1+M gamma) at this scale
    except Ineligible:
        return False, None, None
    A = abs(S)
    if max(tot.bit_length(), A.bit_length()) + 3 > LIMIT: return False, None, None   # not representable
    thr = (1 << (Z - 126)) if Z >= 126 else 0           # E3: A - tot > 2^-126
    if A - tot <= thr: return False, None, None
    L = A.bit_length()
    if L <= 8: return False, None, None
    sh = L - 8; q, rem = A >> sh, A & ((1 << sh) - 1); h = 1 << (sh - 1)
    if rem > h or (rem == h and q & 1): q += 1
    if q == 256: q, sh = 128, sh + 1
    y = q << sh; up = 1 << sh; down = (1 << (sh - 1)) if q == 128 else up
    if not (4 * A - 4 * tot > 4 * y - 2 * down and 4 * A + 4 * tot < 4 * y + 2 * up): return False, None, None   # E4
    return True, (y if S > 0 else -y), Z

def run(wb, xb, label, hints=None):
    en = encode(wb, xb, hints); S, Bs, E1, E2 = host_sums(en); ref = oracle(wb, xb)
    R, T = S.shape; ok = bad = 0
    for r in range(R):
        for t in range(T):
            c, y, Z = certify(int(S[r, t]), int(Bs[r, t]), int(E1[r, t]), int(E2[r, t]),
                              int(en['sig'][r]), int(en['bet'][r]), int(en['tau'][t]), int(en['chi'][t]), en['K'], en['M'])
            if c:
                ok += 1
                if (y << (SH - Z) if SH >= Z else y >> (Z - SH)) != ref[r][t]: bad += 1
    print(f"{label}: certified {ok / (R * T):.1%} of {R * T}; certified-but-wrong: {bad}; "
          f"truncations W/X: {int(en['tW'].sum())}/{int(en['tX'].sum())}; max Z={int((en['sig'][:, None] + en['tau'][None, :]).max())}")
    return ok, bad

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    def case(K, R, T, wmul=1.0, xmul=1.0, massive=1.0):
        w = rng.standard_t(5, size=(R, K)) * 0.02 * wmul
        x = rng.standard_t(3, size=(T, K)) * xmul; x[:, ::97] *= massive
        x[:, 5] = 1e-9 * xmul                                     # forces activation truncation (tX, E2)
        return bf16(w), bf16(x)
    tot_bad = 0
    for K, R in [(2560, 120), (4096, 80), (9728, 30)]:
        tot_bad += run(*case(K, R, 3), f"K={K} ordinary")[1]
        tot_bad += run(*case(K, R, 3, massive=200.0), f"K={K} massive channels")[1]
    tot_bad += run(*case(2560, 60, 2, wmul=2.0**-50, xmul=2.0**-62), "K=2560 fine scale (Z>150: eta worth many units)")[1]
    tot_bad += run(*case(2560, 60, 2, wmul=2.0**-60, xmul=2.0**-70), "K=2560 subnormal partial sums")[1]
    print("TOTAL certified-but-wrong:", tot_bad)
