# Qwen3-4B pinned dot product experiment

## Decision to make

Can a dedicated ZisK state machine prove the **same ordered BF16-input, FP32-accumulating dot product** at least 10× faster than the current guest on a matched GPU workload? Measure a complete 64-row projection block before changing the decoder, publishing a release or scheduling the full request. This is a candidate extension beyond the current dot_f8-only upstream direction. Keep it experimental pending review.

## Computation the verifier must recognize

Use the saved 28-prompt-token, 12-reply-token request for Qwen3-4B-Instruct-2507. All 39 consumed positions use the original BF16 checkpoint. Keep the greedy tie and stop rules, order of tokens, model identity and arithmetic profile. For each projection, round the activation to BF16 as today. Start 32 FP32 lanes at positive zero. Column i updates lane i modulo 32 by a fused multiply-add rounded once to FP32. Combine lanes at offsets 16, 8 and 4, then compute (lane0 + lane1) + (lane2 + lane3) in FP32 and round to BF16. Preserve the reference handling of exceptional inputs, including rejection where the guest rejects. Every job owns complete dots.

## What the existing evidence establishes

| Quantity | Evidence and scope |
|---|---|
| Required projection and head arithmetic | From dimensions: 100,925,440 projection weights × 39 positions = 3,936,092,160 logical FMAs per layer. Across 36 layers plus 151,936 × 2,560 × 12 in the head, the total is **146,366,791,680 FMAs**. Attention and lane reductions are additional work. FMAs are distinct from emulator instructions, circuit rows and GPU time. |
| Arithmetic hotspot | In the one-position layer-19 profile, projections consumed 95.49% of variable emulator cost, accumulation 67.29% and the dot_f8 opcode 2.12%. Each percentage has the same layer denominator. The remaining work and the relative prover cost require their own measurements. |
| First full decoder failure | Layer 0 passed ASM replay. Proof generation exhausted per-thread MemCounter slots, including on the rebuilt host. The failure identifies the resource. The contribution of activation rereads, the proposed 12B records and the peak slots needed are hypotheses. |
| Current proof timing | Shard 74 covered 384 rows × 2,560 columns × 12 positions = 11,796,480 FMAs. Its GPU proof took 2,809.187 s. A shard includes authentication, normalization, scoring, winner selection and recursion. One size gives neither marginal FMA time nor a whole-model ETA. |
| Current correctness | Native capture matched all 1,823,232 reference logits and all 12 IDs. The saved embedding and shard 74 proofs verified. Zero decoder proofs and zero full-request aggregates have verified. |

Ordered rounding prevents substitution of an ordinary exact matrix-product identity. It does not establish a universal lower bound of one proof row per addition. Treat one row per lane step as a design to price. Charge every lookup, transition, memory argument, reduction, fallback and recursive proof.

## Candidate instruction and data flow

1. The host captures the pinned native forward pass and packs operands in a versioned, canonical layout. Initially preserve the existing tensor digests. Either prove that the packed view is derived from the authenticated bytes or have enrollment bind a reproducible packed-layout commitment to those tensors. A request must never choose which weights the verifier trusts.
2. The decoder calls the candidate instruction with one matrix tile, up to 64 output rows, all positions and the full input width. Bind element format, dimensions, strides, address ranges, output shape and call identity. Check bounds, aliasing policy, ordering and final partial tiles. Reuse authenticated operands internally across rows and positions. A 480-call decoder layer is a call-count target. Record the bytes and memory events at both the guest boundary and inside the state machine. Reducing VM loads can move costs to a lookup or permutation argument.
3. Constrain BF16 operand bits, signs, exponent classes and significands. For ordinary finite values, constrain the exact significand product, exponent alignment, sign choice, normalization and round-to-nearest-even. Quotient, remainder and leading-bit witnesses require integer bounds strong enough to prevent field wraparound. Handle gaps beyond the alignment table with a proved sticky-bit rule or checked fallback. Verify sign on zero, ties with both parities, cancellation, rounding carry, normal/subnormal transitions and fused semantics at extreme exponents.
4. Bind the previous accumulator to the next one for the same call, row, position and lane. Enforce initial +0, column coverage, padded rows, terminal lane values, every ordered reduction and final BF16 rounding. Input and output memory events must be tied to those cells. A correct single-row arithmetic relation alone does not prove a dot.
5. For unsupported arithmetic, the guest recomputes the *entire dot* on authenticated operands with the existing implementation. Prove normal-path eligibility before accepting a hardware result. A spuriously requested fallback may increase work. A suppressed required fallback or unchecked status must never admit an incorrect result. Count real fallbacks and the cost of adversarial all-fallback inputs.
6. Keep an independent software oracle based on the pinned fused operation and reduction. Share the ABI and serialization; validate witness generation against the oracle. Common implementation code is insufficient evidence of circuit correctness. Keep the existing guest and proof suite available for comparisons.
7. Retain the existing embedding, 36 logical layers, 75 head shards, receipt links and request verifier. A future release must pin the new guest and setup keys, instruction semantics and layout version. Cross-request batching and changing the commitment hash are separate experiments.

## Experiments and gates

| Gate | Work | Pass condition |
|---|---|---|
| 1. Baseline and capacity | Prove a 2,048-row head shard and, if necessary, repeat shard 74 on the same GPU, release, input, flags and cache policy. Instrument the old layer planner for slot capacity, highest occupancy per thread, failing chunk and access category. | Reload verified proofs. Report both head timings, stage breakdown, fixed setup versus timed work, and measured slot use. A two-point slope describes these shards only. |
| 2. Executable specification | Exercise all exponent gaps, both signs, zero, ties, sticky bits, carry, cancellation, subnormals, overflow, NaNs, reduction order and rejection. Differential-test against an independent fused reference. | Bit equality on accepted inputs, matching rejection on rejected inputs; adversarial cases at both lane and final reduction boundaries. |
| 3. Constraint prototype | Replay the actual layer-0 K projection, 39 × 1,024 × 2,560 = 102,236,160 FMAs. Mutate a lane by one ULP, remainder, swap, leading bit, terminal reduction, operand address, lane identifier, coverage, fallback status or output bit. Test mutations that change the claimed relation. | Full native parity and rejection of false claims by the constraints. Reload a real GPU proof, then reject a tampered proof. |
| **4. Investment decision** | Prove **the identical 39 × 64 × 2,560 block (6,389,760 FMAs)** with the existing kernel and candidate. Use the same original tensors, values, integrity checks, public result, GPU, backend, witness threads and measurement boundaries. Derive the proper program/setup keys separately for each arm. Record cold setup separately. Repeat timed runs and report spread. | **At least 10× in verified proof generation time** on the matched block, unchanged bits, acceptable peak host RAM/VRAM and complete memory binding. Also report witness time, basic proof time, recursion, rows, lookups, memory events and total wall time. If it misses, stop before guest migration. |
| 5. Integration | Wire the winning kernel into all projections and head shards. Authenticate the packed layout. Check the changed keys and program selection. | On the saved request, compare the actual 37 boundary arrays, all 1,823,232 logits and all 12 token IDs against the pinned reference. Receipts alone do not expose every logit. |
| 6. Full acceptance | First prove and reload layer 0 at 39 positions. Then prove all 112 children and aggregate. | Full client acceptance under independently pinned keys. Reject changed nonce, prompt token, weight, missing or swapped child, wrong program key and edited reply. Publish per-child timings, peak resources, proof sizes and final verification. |

## Source and artifact map for the handoff

The design branch starts at b235c8e; the active packaged files are local working-tree inputs. Record ELF hashes and the exact source snapshot before claiming reproducibility.

- Pinned oracle and reduction: `Qproof/common/src/lib.rs`, functions `bf16_linear_dot` and `finish_bf16_dot`.
- Lossless guest kernel: `Qproof/harness/dotf8/src/lib.rs`, functions `project_group_bf16` and `accumulate`. Layer execution: `Qproof/harness/layer/src/lib.rs`, function `execute_batched`.
- Guest entry points: `Qproof/guests/src/lib.rs` and `Qproof/guests/src/bin/{embed,layer_batched,head_batched}.rs`.
- Release pins and acceptance: `Qproof/release/release.json`, `Qproof/harness/client/cli`, `Qproof/tools/prove_prepared.py`.
- Historical profile and parity tests are in the *outer* worktree: `harness/measurements/QWEN_FULL_MODEL_COST_1X1.md`, `harness/measurements/QWEN_RENTAL_REVIEW.md`, `harness/layer/tests/mac_parity.rs`.
- Frozen run: `Qproof/target/latency.aBb7uA/run/`. Use its `reference.json`, `roster.json`, `logs/layer00.capacity.log`, `logs/head-00074.prover.log` and saved proofs.
