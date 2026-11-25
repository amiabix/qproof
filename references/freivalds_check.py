# L2 reference for QPROOF_LAYERED_PLAN.md section 3.2: host-supplied exact sums (S, Bs, E1, E2)
# verified in the guest by a random-vector check over Goldilocks.
# - The seed binds the domain, request, layer, projection, dimensions, the authenticated BF16 weight and
#   activation bytes, the weight-scale hints and all four sum matrices. Activation scales are
#   deterministic; every encoding is a function of bound data.
# - Field elements are sampled uniformly (rejection sampling).
# Needs m0_certificate_reference.py in the same directory. Usage: python3 -I freivalds_check.py
import hashlib, importlib.util, os, struct
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m0", os.path.join(here, "m0_certificate_reference.py"))
m0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m0)

P = 2**64 - 2**32 + 1
LIM = 2**61

def ints(a): return [[int(v) for v in row] for row in a]

def seed_for(ctx, wb, xb, hints, sums):
    h = hashlib.sha256()
    h.update(b"qproof/L2/v1"); h.update(ctx["request_id"])
    h.update(struct.pack("<IIIII", ctx["layer"], ctx["projection"], wb.shape[0], xb.shape[0], wb.shape[1]))
    h.update(hashlib.sha256(wb.tobytes()).digest()); h.update(hashlib.sha256(xb.tobytes()).digest())
    for s in hints: h.update(np.asarray(s, np.int64).tobytes())
    for mtx in sums:                                       # signed values, fixed-width encoding
        h.update(b"".join(int(v).to_bytes(16, "little", signed=True) for row in mtx for v in row))
    return h.digest()

class Stream:                                             # uniform elements of F_p by rejection
    def __init__(self, seed): self.seed, self.ctr, self.buf = seed, 0, b""
    def field(self):
        while True:
            if len(self.buf) < 8:
                self.buf += hashlib.sha256(self.seed + self.ctr.to_bytes(8, "little")).digest(); self.ctr += 1
            v, self.buf = int.from_bytes(self.buf[:8], "little"), self.buf[8:]
            if v < P: return v

def guest_verify(ctx, wb, xb, hints, sums, range_check=True, instances=2):
    S, Bs, E1, E2 = sums
    if range_check:
        for mtx, signed in ((S, True), (Bs, False), (E1, False), (E2, False)):
            for row in mtx:
                for v in row:
                    if (abs(v) >= LIM) if signed else not (0 <= v < LIM): return False
    en = m0.encode(wb, xb, hints)                          # guest-side encodings from bound data
    W, tW, Wu, X, tX, Xu, m = en["W"], en["tW"], en["Wu"], en["X"], en["tX"], en["Xu"], en["m"]
    R, K = W.shape; T = X.shape[0]; XM = Xu * m
    st = Stream(seed_for(ctx, wb, xb, hints, sums))
    for _ in range(instances):
        al = st.field(); v = [st.field() for _ in range(T)]
        a2, a3 = al * al % P, al * al * al % P
        a = [sum(int(X[t, c]) * v[t] for t in range(T)) % P for c in range(K)]
        b = [(al * sum(int(XM[t, c]) * v[t] for t in range(T)) + a3 * sum(int(tX[t, c]) * v[t] for t in range(T))) % P for c in range(K)]
        cc = [a2 * sum(int(Xu[t, c]) * v[t] for t in range(T)) % P for c in range(K)]
        for r in range(R):
            lhs = sum(v[t] * ((S[r][t] + al * Bs[r][t] + a2 * E1[r][t] + a3 * E2[r][t]) % P) for t in range(T)) % P
            rhs = sum(int(W[r, c]) * a[c] + int(Wu[r, c]) * b[c] + int(tW[r, c]) * cc[c] for c in range(K)) % P
            if lhs != rhs: return False
    return True

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    K, R, T = 2560, 12, 3
    w = rng.standard_t(5, size=(R, K)) * 0.02
    x = rng.standard_t(3, size=(T, K)); x[:, 7] = 1e-9                # forces activation truncation
    wb, xb = m0.bf16(w), m0.bf16(x)
    ctx = dict(request_id=b"\x11" * 32, layer=0, projection=1)
    en = m0.encode(wb, xb); hints = (en["sig"], en["bet"])
    honest = [ints(s) for s in m0.host_sums(en)]
    assert any(v for row in honest[3] for v in row), "E2 must be nonzero for its mutation test"

    def mutate(i, fn):
        s = [[row[:] for row in mtx] for mtx in honest]; fn(s[i]); return s
    def setv(r, t, f): return lambda mtx: mtx[r].__setitem__(t, f(mtx[r][t]))
    nz = lambda mtx: next((r, t) for r in range(R) for t in range(T) if mtx[r][t] != 0)
    r2, t2 = nz(honest[3]); r1, t1 = nz(honest[2])

    cases = {
        "honest (must pass)":                 (honest, True),
        "S + 1 at one output":                (mutate(0, setv(4, 1, lambda v: v + 1)), True),
        "S + p (same residue mod p)":         (mutate(0, setv(4, 1, lambda v: v + P)), True),
        "S + 2^62":                           (mutate(0, setv(4, 1, lambda v: v + 2**62)), True),
        "Bs - 1 (bound too small)":           (mutate(1, setv(2, 0, lambda v: v - 1)), True),
        "E1 zeroed (hide weight truncation)": (mutate(2, setv(r1, t1, lambda v: 0)), True),
        "E2 - 1 (hide activation truncation)":(mutate(3, setv(r2, t2, lambda v: v - 1)), True),
        "rows 0 and 1 swapped in S":          ([honest[0][1], honest[0][0]] + honest[0][2:], True),
        "positions 0 and 1 swapped in S":     ([[row[1], row[0]] + row[2:] for row in honest[0]], True),
    }
    for name, (sums, rc) in cases.items():
        if name.startswith("rows") or name.startswith("positions"): sums = [sums] + honest[1:]
        ok = guest_verify(ctx, wb, xb, hints, sums, range_check=rc)
        expect = name.startswith("honest")
        print(f"{name:40s} {'accepted' if ok else 'rejected'}  {'OK' if ok == expect else 'FAIL'}")
    # Why the range check exists: without it, S + p is the same field element and passes.
    s_p = mutate(0, setv(4, 1, lambda v: v + P))
    print(f"{'S + p with range check disabled':40s} {'accepted' if guest_verify(ctx, wb, xb, hints, s_p, range_check=False) else 'rejected'}  (expected: accepted)")
    # Binding: a different weight-scale hint changes the seed, and the honest sums for the old hint fail.
    alt = (hints[0] - 1, hints[1])
    print(f"{'weight-scale hint changed, old sums':40s} {'accepted' if guest_verify(ctx, wb, xb, alt, honest) else 'rejected'}  (expected: rejected)")
    # A consistent alternative hint is allowed: sums recomputed under it pass, and it can never certify
    # a wrong value (m0 falsification covers this).
    en2 = m0.encode(wb, xb, alt); honest2 = [ints(s) for s in m0.host_sums(en2)]
    print(f"{'alternative hint, matching sums':40s} {'accepted' if guest_verify(ctx, wb, xb, alt, honest2) else 'rejected'}  (expected: accepted)")
    # Binding: a different request or projection changes the challenges; honest sums still pass.
    ctx2 = dict(ctx, projection=2)
    print(f"{'different projection id, honest sums':40s} {'accepted' if guest_verify(ctx2, wb, xb, hints, honest) else 'rejected'}  (expected: accepted)")
