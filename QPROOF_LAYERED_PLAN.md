# qproof: layered proving plan

Owner: Ayush. Implementer: Codex. Specs: this file.
Paths are as on `amiabix/qproof` main @ b235c8e; map them to the worktree layout (`guests/src/bin/...`) before starting.

## 0. Where we are

| Part | State |
|---|---|
| Model correctness | Native harness matches the pinned BF16 reference on the saved request: 37 activation boundaries, 1,823,232 logits, 12 reply tokens. |
| Proofs | 2/112 verified (embedding, head shard 74). 0/36 decoder layers. |
| Blocker | Layer 0 (39 positions) runs in ASM, then the prover's memory-access counter exhausts its per-thread slots. The cause by address range is not yet measured. |
| Cost | Today's layer guest re-executes FP32 rounding in software: ~70 guest steps per FMA (measured at 1 position), 3.94B FMAs per layer at 39 positions. |

## 1. Frozen rules

1. The statement never changes: given the prompt token IDs, the pinned Qwen3-4B-Instruct-2507 BF16 checkpoint, the pinned arithmetic and greedy decoding, this is the reply. The pinned projection is `common/src/lib.rs` `bf16_linear_dot` + `finish_bf16_dot` (32 lanes, fma from +0 over columns l, l+32, ...; reduction at offsets 16, 8, 4 then (0+1)+(2+3); BF16 RNE). Any change to an output bit is a rejected change.
2. No quantization, no exact-sum-rounded-once profile, no tolerance checks.
3. Claims are limited to transcripts of at most 512 positions until the 500/250 divergence (layer 0, context 574) is root-caused.
4. Evidence rules (BRIEF.md §7): cite file:line, label every number measured or estimate, no stubs presented as progress, nothing is done without passing output.

## 2. Architecture: seven layers, one contract each

Each layer is testable on its own against L0. A layer above never reaches into a layer below except through its contract.

| Layer | Responsibility | Contract | Status |
|---|---|---|---|
| L0 Oracle | Frozen inputs and outputs of every projection on the saved request; the pinned kernel as reference | `golden/` files + manifest; native replay reproduces every output bit | verified (WP0) |
| L1 Identity | Authenticate weights and activations; bind request, model and program | Today's SHA-256 roots and receipts, unchanged | exists |
| L2 Exact algebra | Verify host-supplied exact integer sums S, Bs, E1, E2 for a matrix | §3.2 Freivalds check: accept iff all four matrices are the true integer products | new (WP3) |
| L3 Decision | Per output: certified (value from S) or exact path | §3.3 certificate; never certifies a wrong value | new (WP3) |
| L4 Exact path | Pinned value for every output L3 does not certify | Today's kernel (`harness/dotf8` `project_lanes`), restricted to the listed (row, position) pairs. The checker state machine (WP8) can replace it behind the same function | exists, restrict |
| L5 Layer composition | Norms, RoPE, attention, softmax, SiLU, residuals; splitting a layer into jobs | Today's `run_layer_batched` with projections routed through L2-L4 | adapt (WP6) |
| L6 Request composition | 112 children, join, aggregation, client verify | Unchanged | exists |

Nothing outside the guest changes for the client. The release keeps its SHA-256 weight roots, because the guest derives every encoding from the authenticated BF16 bytes. Only the program keys change.

## 3. Core specs

### 3.1 Encodings (all computed by the guest from authenticated BF16 values)

For a matrix with R rows, T positions and K columns (n = K/32 terms per lane, M = n + 4):

- `W'[r][c] = trunc(w·2^σ_r)`, required `|W'| < 2^20`; `tW[r][c] = 1` iff the truncation dropped bits.
- `Ŵ[r][c] = ceil(|w|·2^β_r)`, required `Ŵ <= 2^15`.
- `X'[t][c] = trunc(x·2^τ_t)`, required `|X'| < 2^27`; `tX[t][c]` likewise.
- `X̂[t][c] = ceil(|x|·2^χ_t)`, required `X̂ <= 2^19`.
- `m_c = (n - max(j_c, 1)) + 5` with `j_c = c / 32`. This is the number of relative roundings term c passes through: lane steps max(j_c,1)..n-1, then 5 reduction levels. Step 0 adds to +0 and can only round by underflow.

Scales:
- Activation scales τ, χ are deterministic: the guest derives them from the leading bit of each position's largest authenticated BF16 value, so that the bounds above hold with no slack lost.
- Weight scales σ, β are host hints, because deriving them would cost an extra pass over the weights. They are bound into the challenge seed (§3.2) before any challenge is drawn. The guest range-checks every encoded value while decoding. A bad hint can only fail a range check or lower the certification rate; it can never produce a wrong output, and it cannot be changed after the challenges are known.

### 3.2 L2: exact sums, verified in the guest

The host supplies, per matrix, `S = W'·X'ᵀ`, `Bs = Ŵ·(X̂⊙m)ᵀ`, `E1 = tW·X̂ᵀ`, `E2 = Ŵ·tXᵀ` (R×T integers).

Guest procedure:

1. Range checks: `|S| < 2^61`; `0 <= Bs, E1, E2 < 2^61`. The true values are below 2^60.3, so any accepted difference is below 2^63 < p and equality mod p implies equality over the integers. Without this check, S + p passes (it is the same field element); the reference demonstrates both cases.
2. Challenges: `seed = SHA256("qproof/L2/v1" || RequestID || layer || projection || R || T || K || SHA256(BF16 weight bytes) || SHA256(BF16 activation bytes) || weight-scale hints || S || Bs || E1 || E2)`, with every sum value encoded as a fixed-width 16-byte signed integer. Every encoding is a deterministic function of these bound inputs, so nothing the check depends on can change after the challenges are drawn.
   Derive two independent instances `(α_i, v_i)`, `α_i ∈ F_p`, `v_i ∈ F_p^T`, p = 2^64 - 2^32 + 1. Sample uniformly by rejection: read 64-bit words from a SHA-256 counter stream and discard any word >= p. Never reduce mod p (that is biased).
3. Per instance: `a = X'ᵀv`, `b = α·(X̂⊙m)ᵀv + α³·tXᵀv`, `c = α²·X̂ᵀv` (K-vectors, O(T·K)).
4. One pass over the weights, per row r:
   `Σ_t v_t (S + αBs + α²E1 + α³E2)[r][t]  ==  Σ_c (W'[r][c]·a_c + Ŵ[r][c]·b_c + tW[r][c]·c_c)`.
   W' and Ŵ are small integers: accumulate the product of small integer × field element in 128 bits and reduce once per row.
5. Soundness: if any entry of any of the four matrices is wrong, a fixed row identity becomes a nonzero polynomial of degree <= 4 in (α, v). It passes with probability <= 4/p per instance, so <= 2^-123 per attempt with two instances. Grinding costs the prover one hash per attempt.

Cost (estimate, measure in WP4): about 2.4B guest steps per decoder layer for the weight pass and about 1B for decoding, independent of T. This covers the weight pass only. It does not cover hashing, reading the four sum matrices, the certificate, the exact path or the nonlinear ops; WP4 measures the whole guest. Today's layer is about 158-276B steps (estimate from the 1-position measurement).

Memory: the four sum matrices are 4 × 1,198,080 values × 8 bytes = 36.6 MiB per decoder layer at 39 positions, read as job input. WP4 and WP5 count planner slots by region (weights, activations, sums, exact-path operands) for both guests on the same tile.

Reference: `freivalds_check.py`. The honest case passes. These are rejected: S+1, S+p, S+2^62, Bs-1, E1 zeroed, E2-1 (activation truncation forced), swapped rows, swapped positions, and a changed weight-scale hint with the old sums. Demonstrated as accepted: S+p with the range check disabled, which is why the check exists, and an alternative hint with matching sums, which is allowed and sound.

### 3.3 L3: certificate (lemma)

Rounding model per RNE32 result: `fl(z) = z(1+ε) + η`, `|ε| <= u = 2^-24`, `|η| <= 2^-150`, εη = 0, valid while nothing overflows.

Lemma 1: `|ŝ - S| <= γ·Σ m_c|p_c| + B_η`, where `γ = u/(1 - M·u)` and `B_η = (K+31)·2^-150·(1 + M·γ)`.
Lemma 2: `|S - S_fixed| <= E = 2^(-σ-χ)·E1 + 2^(-β-τ)·E2`, where `S_fixed = 2^(-σ-τ)·S`. This uses `wx - w'x' = (w-w')x + w'(x-x')` and `|w'| <= |w|`.
Bound on the weighted absolute sum: `Σ m_c|p_c| <= 2^(-β-χ)·Bs`.

`tot = γ·2^(-β-χ)·Bs + E + B_η`. The output is certified when all four hold:

- E1: every input is finite. The guest already rejects non-finite activations, and the release asserts every weight is finite.
- E2: no overflow: `(1 + M·γ)·2^(-β-χ)·Bs/5 + B_η < 2^127`. Every m_c >= 5, so this bounds every partial sum without reading the trace.
- E3: `|S_fixed| - tot > 2^-126`.
- E4: let `y = RNE_bf16(S_fixed)` with open rounding interval `(lo, hi)` (the spacing below a power of two is halved). Require `lo < |S_fixed| - tot` and `|S_fixed| + tot < hi`. Strict, so no tie can occur.

When certified, the output is `sign(S)·y`. Otherwise the output goes to L4.

Integer procedure for the guest:
- Work at S's scale `Z = σ + τ` in u128. γ = 1/(2^24 - M) exactly, and (1 + M·γ) = 2^24/(2^24 - M).
- Each tot term is converted to scale Z with an exact ceiling:
  - γ·Bsum: `ceil(Bs·2^(Z-β-χ) / (2^24-M))`;
  - E1 term: `ceil(E1·2^(τ-χ))`;
  - E2 term: `ceil(E2·2^(σ-β))`;
  - B_η: `ceil((K+31)·2^24·2^(Z-150) / (2^24-M))`. This is computed at the actual scale; it is not one unit. When Z > 150 it is many units.
- If any converted term, |S|, or 4·(|S| + tot) needs more than 127 bits, the output is ineligible and goes to L4. A right shift of a tiny term rounds up to one unit.
- E2 in integers: `Bs·2^24 < 5·(2^24-M)·2^(126+β+χ)`. The η term is below 2^-100 and fits in the slack between 2^126 and 2^127.
- E3 in integers: `|S| - tot > 2^(Z-126)` when Z >= 126; otherwise `|S| - tot >= 1`.
- `y`: with `L = bitlen(|S|)`, if `L <= 8` the output is ineligible. Otherwise `shift = L - 8`, then RNE on the dropped bits, renormalizing if q reaches 256.
- `ulp_up = 2^shift`; `ulp_down = 2^(shift-1)` if q == 128, else 2^shift.
- Test `4|S| - 4tot > 4y - 2·ulp_down` and `4|S| + 4tot < 4y + 2·ulp_up`, which stays in integers when ulp_down = 1.

Reference: `m0_certificate_reference.py`.
- It runs exactly this integer routine and computes exactly four products.
- It compares every certified output with an exact model of the pinned kernel: fma from +0 in 32 lanes, 31 reduction adds, BF16 RNE, with FP32 and BF16 subnormals.
- Test cases: ordinary and very-large-channel activations at K = 2,560 / 4,096 / 9,728; a fine-scale case with Z up to 158, where B_η is worth many units; and a case whose partial sums are subnormal. Activation truncation is forced in every case.
- Result: 0 certified-but-wrong across all 1,620 synthetic outputs. The subnormal case certifies 0% and routes everything to L4, as intended.

### 3.4 Head (greedy only needs the winner)

- The prover names the winner v*. The guest proves y* exactly, by certificate or L4.
- Each other token v is eliminated when E2 holds and `S_fixed,v + tot_v < lo(y*)`, compared at a common scale with the left side rounded up. E3 and E4 are not needed for elimination.
- Every token not eliminated goes through L4 and must satisfy `y_v < y*`, or `y_v == y*` and `v > v*`.

### 3.5 Integer bounds (worst case K = 9,728, M = 308)

| Integer | Bound |
|---|---|
| S | < 2^20 · 2^27 · K < 2^60.3 |
| Bs | <= 308 · 2^15 · 2^19 · K < 2^55.6 |
| E1 | < 2^19 · K < 2^33 |
| E2 | < 2^15 · K < 2^29 |

All are below p/2, where p = 2^64 − 2^32 + 1.

## 4. Work packages

### WP0: golden data (L0)
- Do: add an export hook around the `claim` closure in `pci_layer::execute_batched` (`harness/layer/src/lib.rs`).
  - Per projection group, write x (FP32 bits as passed), cols, matrix id and output bits.
  - Write the layer input and output.
  - For the head, write the final-norm output, per-shard logits and winners.
  - Layers 0, 19, 35 + head on the frozen 39-position request.
- Format: `golden/L<layer>/<matrix>.{x,out}.npy` + `manifest.json` (shape, dtype, sha256; weights by checkpoint path, tensor name, byte offset; never copy weights).
- Accept: a native replay with `project_group_bf16` reproduces every output bit from x + checkpoint; two exports have identical hashes.
- Don't: touch kernel code.

### WP1: M0, real certification rates (L3)
- Do: create `harness/cert` (no_std, shared by host and guest) with §3.1 encodings, the §3.3 integer procedure and §3.4 elimination. Port `certify` from `m0_certificate_reference.py` line for line, so M0 measures the exact code the guest will run. The falsification oracle on real data is the native pinned kernel (`project_group_bf16`), not the Python model.
- Run on WP0.
- Report per matrix: certified share by outputs and by FMA weight; ineligible counts by E2/E3/E4; certified-but-wrong; max log2 of every integer.
- Report for the head: the winner's path, the share eliminated, the candidates sent to L4.
- Report the combined FMA-weighted residual f.
- Accept: certified-but-wrong = 0, every §3.5 bound asserted, the report regenerates byte-identically.

### WP2: planner slot attribution (L5 diagnosis, ZisK fork)
- Do: in `state-machines/mem-cpp/cpp/mem_counter.cpp` (constants in `mem_config.hpp`), dump used slots per thread and per 64 MB page (ROM/INPUT/RAM), plus the distinct chunks per hot address range. Do it at the end of counting, or at the failure point.
- Run layer 0 at T = 4, 8, 16 (expected to fit) and at T = 39 (dump at failure). Fit slots(T).
- Report used slots per thread. Each counter thread owns the addresses whose bits 3-4 match its mask (`mem_counter.cpp`, `ADDR_MASK` in `mem_config.hpp`), so addresses are interleaved word by word. The saved failure names thread 2. Compare peak allocations across threads. If counts are skewed, measure which ranges drive the skew and test padding on the same input.
- Accept: a table that explains the layer-0 failure by region and by thread, with the predicted need at T = 39 against capacity. No data-layout change anywhere until this table exists.

### WP3: L2 + L3 in code
- Do:
  - `harness/cert`: the §3.2 Freivalds verifier (Goldilocks, two instances, challenge derivation, range checks).
  - `harness/cert-host`: S, Bs, E1, E2 (int64 matmuls), scale hints, and the cert record for a tile or a layer.
- Tests:
  - golden: on WP0, every certified output equals the oracle;
  - adversarial: S±1, S+p, S+2^62, Bs-1, E1 zeroed, E2-1 with activation truncation forced, swapped rows, swapped positions, a wrong lane index in m_c, a weight-scale hint changed after the sums were computed. Each must be rejected, or must fall to L4; never a wrong output;
  - binding: changing the RequestID, layer, projection, any dimension, one weight byte, one activation byte or one hint changes the seed;
  - sampling: challenges come only from the rejection sampler; a test asserts no reduction mod p exists in the path;
  - property: appending a position does not change any earlier output.
- Accept: all tests pass; the adversarial suite reports 100% rejection.

### WP4: one tile end to end (L1-L4)
- Tile: layer 0, K projection, rows 0-63, all 39 positions, K = 2,560 (6,389,760 FMAs).
- Do:
  - `harness/f8-guests/src/bin/tile_cert.rs` does five things: authenticate the weight slice and activations; read the cert record; run L2; run L3 per output; send the rest to L4 (today's kernel, listed (r,t) only). It commits a receipt with the input roots, output root, status-bitmap root and counts.
  - `tile_baseline.rs`: the same tile on today's kernel.
- Accept:
  - both outputs equal the WP0 slice bit for bit;
  - ASM replay passes;
  - the WP3 adversarial cases, run through the guest, trap or fail the receipt;
  - steps and emulator cost recorded for both.

### WP5: tile proofs on GPU
- Do: prove both tiles, twice each; reload; verify under the approved keys. Also prove a 128-row baseline tile, to separate fixed cost from per-FMA cost.
- Record: steps, emulator cost, planner slots by region and by thread, witness generation, GPU prove, recursion, peak VRAM, peak host RAM, wall time.
- Accept: verified proofs and one comparison table.
- Gate to WP6: the certified tile's output is bit-identical to golden, its verified wall time is lower than baseline, and its measured slot use extrapolates within capacity for a full layer. Any one missing means no WP6.

### WP6: one verified decoder layer (L5)
- Do: `layer_cert.rs` is today's layer guest with all seven projections through L2-L4; norms, RoPE, attention, softmax, SiLU and residuals are unchanged.
- Job policy: if WP2 shows the planner overflows, split the layer into chained jobs at group boundaries: QKV | attention | O + residual | gate/up/SiLU | down + residual. Commit the intermediate tensors and chain the receipts like layers do.
- Accept: layer 0 output equals golden bit for bit; ASM passes; the planner stays within capacity; the proof generates, reloads and verifies; the client links it to the embedding output and to layer 1's input.

### WP7: full request
- Do: head with §3.4 elimination, all 36 layers, 75 shards, aggregate, client verify.
- Accept (done criteria):
  - the saved transcript reproduces all 37 boundaries, all 1,823,232 logits and all 12 IDs bit for bit;
  - every child and the aggregate verify under pinned keys;
  - changing the nonce, one prompt token or one weight is rejected;
  - cost per position and per prediction reported at T = 39 and 512.

### WP8: checker state machine (replaces L4; scope-gated)
- Do now, outside the fork:
  - a native reference + constraint simulator for the row predicate, in integer operations only: claimed a_j inside the rounding interval of a_{j-1} + w·x; 32n lane steps from an explicit +0; 31 reduction adds; the final BF16 rounding; explicit rules for -0, subnormal, overflow, NaN; predecessor and operand-index links;
  - a mutation suite: tie, sign, predecessor, operand index, reduction step, special-value flags, ±1 ulp;
  - a cost model: columns and lookups per row.
- In the fork only after Francesc and Jordi agree on scope.
- Priority rule: this becomes the critical path if f > 30% after WP1, or if L4 exceeds half of the layer's guest steps in WP6.

## 5. Gates

| After | Rule |
|---|---|
| WP1 | Any certified-but-wrong output: stop and fix the lemma. If f > 70%, run layers on L4 alone (skip L2/L3) and make WP8 the main lever. |
| WP2 | No layout change before the slot table exists. |
| WP5 | WP6 starts only with all three: bit-identical output, lower verified wall time than baseline, and slot use within capacity for a full layer. |
| WP6 | Layer 0 does not fit: apply the job split before any other change. |

## 6. Ownership

| Item | Owner |
|---|---|
| Everything | Ayush, accountable |
| WP0-WP7 implementation | Codex |
| WP8 | Ayush accountable; Francesc invited to co-own (pending; Jordi's directive was FP8-only); Codex builds the native reference |
| Specs §3, review of WP1/WP3/WP4 outputs | Claude |

## 7. Files that come with this plan

- `m0_certificate_reference.py`: the §3.1 encodings and the §3.3 integer routine, with exactly four products, checked against an exact model of the pinned kernel (subnormals included). Synthetic data. Runs in about 7 s.
- `freivalds_check.py`: §3.2 with full binding and rejection sampling; it runs every mutation listed in WP3 that applies to the sums. Synthetic data. Imports the M0 reference from the same directory.
