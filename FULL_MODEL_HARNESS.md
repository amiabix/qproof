# Proof-Carrying Inference — full-model harness

Design for review. The target is a client-verifiable reply from the original BF16 checkpoint under a specified arithmetic profile. This worktree implements checkpoint enrollment, the child receipt digest and the client-side transcript and layer join checks. The inference guests, verifier and composition described below are delivery requirements. A successful emulator comparison is evidence of arithmetic agreement on its fixture.

## What the client accepts

Given independently installed release pins and a client-created request, accepting a complete reply means that greedy decoding the specified function on the client's token IDs yields exactly the supplied reply IDs, ending at the first allowed stop or the requested length limit. This establishes a computation's answer. It does not attest to the server process that originally served the bytes.

Version 1 starts at token IDs, with a nonempty prompt, positive generation limit and at least one reply token. Require valid vocabulary IDs, checked lengths and context bounds. Text handling belongs to the client with a separately pinned tokenizer, template and rendering policy. Sampling, empty replies and incremental prefix proofs require distinct policies. For P prompt and R output tokens, consume all P prompt IDs and the first R−1 output IDs. T=P+R−1. Predict exactly R times at positions P−1 through T−1. Under the 500/250 workload this gives 749 positions. A benchmark that merely predicts at eight positions in a supplied sequence does not establish that the sequence was generated greedily.

## Program and data flow

```text
Independent release ── checkpoint + arithmetic profile + program/key pins ── client
Client request ── prompt IDs, count, policy, nonce ─── request commitment
                          │
       finished transcript and original weights (untrusted witnesses)
                          ↓
                 embedding guest → A0
                          ↓
       same layer guest, index 0 → A1 … index 35 → A36
                          ↓
                 head guest → R predicted IDs
                          ↓
        verify 38 child proofs + check 37 activation links
                          ↓
          request, reply, policy and termination checks
                          ↓
                   client accepts or rejects
```

**Enrollment and pins.** The release builder maps named BF16 records to tensor digests and a checkpoint ID. The client or an independent approver must recompute this inventory against the intended publisher revision. An untrusted server's fresh inventory alone authenticates nothing. The release file in harness/qwen3-4b-release.json pins Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c. Its three publisher shard digests match the recorded download verification on Omarchy. The local config and index hashes match the immutable release. The Instruct-2507 shard digests differ. Enrollment streams all files again and rejects any mismatch. The tokenizer is outside the token-level inventory. Use separate pins for checkpoint, model semantics, and suite. ModelID binds checkpoint ID plus graph, numeric rules, context and stop IDs. SuiteID binds the ABI, relation versions, backend parameters and approved child programs/keys. Place the final key and expected SuiteID together in the external release record to avoid a self-referential key derivation.

**Arithmetic.** Use exact BF16 weight bits, widening losslessly where specified. Pin activation types, order of multiplications and additions, rounding mode, treatment of nonfinite and subnormal values, signed zero, RMSNorm epsilon, grouped-query mapping, RoPE, causal mask, softmax and SiLU implementations. Specify fused versus separately rounded operations. The matched lossless guest currently uses dot_f8 for the seven learned projections and software FP32 for attention. Pin the numerical relation independently of precompile optimizations. Require bit-level conformance from the serving engine. No claim of equivalence to arbitrary GPU kernels.

| Program | Private inputs and checks | Public result |
|---|---|---|
| Embedding | Reconstruct consumed IDs from prompt and reply prefix. Open the indexed rows of the tied matrix under the approved model root, enforce ID bounds and canonical row decoding. | Common header, consumed-ID commitment, T, A0 |
| Layer | Check index and shapes against the pinned graph. Authenticate that layer's seven matrices and four norms; recompute the input root, apply the full causal block and commit every output position. Force the lossless profile. | Header, layer index, T, input and output activation roots |
| Head | Open A36, use the pinned final norm and same tied matrix as the embedding. Cover the whole vocabulary for each required position and break ties by lowest ID. Stream logits and maintain the winner if this preserves the specified arithmetic. | Header, prediction start and count, predicted-ID commitment |
| Request verifier / recursive join | Verify each child under the independently approved key for its role before accepting public values. Enforce exactly one embedding, layers 0 through 35, one head, every boundary equality, and the manifest and transcript checks. | ModelID, SuiteID, RequestID, prompt and reply commitments/counts, policy and termination |

Every child header binds ABI version, kind, ModelID, SuiteID and RequestID. The client computes RequestID from a canonical manifest containing its nonce, prompt commitment/count, decoding policy and limits. The join reconstructs both the embedding's consumed-token commitment and the head's prediction commitment from the same transcript. Check early stops, exact length and the stop-at-limit tie rule. The client checks the verified public fields against locally retained pins, request and response. Define versioned binary encodings with lengths and fixed-width little-endian fields, domain-separated hashes, raw float bit encodings, and strict rejection of trailing data. Each activation root binds model, request, boundary index, positions, shape and values.

**Weight access.** For full scans, a guest can stream tensor bytes and compare the digest with an authenticated inventory entry. The current inventory records whole-tensor hashes. Efficient indexed embedding reads need an additional independently validated row commitment whose leaves bind row index and bytes. Until implemented, a full authenticated scan is the simpler baseline. Tiled or prepacked weights must be derived in-guest from authenticated bytes, or authenticated by an enrollment relation and opened under the resulting root. Treat off-guest cached activation and weight buffers as untrusted.

## Execution, scheduling and optimization

1. Start serving under the pinned profile. After the transcript is fixed, generate the native witness in layer order across all positions, export typed boundaries, and submit proof jobs as soon as each witness is ready. Native witness production can overlap earlier-layer proving. Once boundary witnesses exist all 38 child proofs can be scheduled independently. Their output links still require verification.
2. Run Q/K/V projections across token tiles, sharing the activation decomposition. Produce masked attention, O and the first residual. Share normalized inputs for gate/up, then run SiLU, down and the second residual. Decode one weight tile and reuse it across positions. Keep each dot product's input-column order and each causal reduction's rounding order. Start with layer-sized proof jobs; add tile proofs only after measuring their authentication, duplicate input and join costs.
3. Retain the simple guest as an oracle. Measure data preparation, ELF execution, weight authentication, trace rows by machine, padding, key setup, witness generation, basic proving, recursion, verification, peak RAM/VRAM, scratch bytes and end-to-end wall time. Profile embedding, layer, head and join separately. Include one-time and per-request work separately. Compute throughput under load independently of time-to-first-verified-reply.
4. Make jobs restartable. Address artifacts by release, request, program, segment, range and input/output digests. Publish atomically after verifying a reloaded proof and its public bytes. Duplicate workers may compute the same job; validate both and promote one. Retain inputs or a reproducible regeneration recipe until verification completes. Admit jobs using measured peak memory and disk; use the existing Omarchy capped queue.
5. Initially accept the full proof bundle via an independent client verifier that performs all joins. This provides a real end-to-end result before a compact proof exists. Then benchmark recursion on real child proofs, including proof/public-output binding, before selecting arity or committing to one compact proof. Expose bundle verification and compact verification as distinct formats. Ordinary backend aggregation of an execution does not automatically join independent guest programs.

## Delivery order and evidence

| Parallel track | First artifact | Acceptance evidence |
|---|---|---|
| Checkpoint and arithmetic | Inventory builder, independent release pins, serialized profile, golden layer boundaries | Validate on the actual checkpoint; same result from an independent parser; release review binds profile to intended reference |
| Programs | Embedding, one reusable lossless layer ELF and head ELF, with typed receipts | Each proof verifies after reload. Tests compare bitwise intermediates and reject modified weights, norms, profile flags, positions and public bytes |
| Verification | Strict manifest/receipt codecs, pinned-key bundle verifier, request join, client CLI | Skip/duplicate/swap children, mix two requests, alter one boundary, replace a key, omit a winning logit, tie or stop incorrectly: each case rejects at the appropriate check |
| Scheduling and cost | Manifest-driven job graph; immutable input fixtures, resource caps and durable results | Kill and resume a worker; retrieve a verified result from cache; report complete wall time and separate component counters on fixed hardware |

First vertical slice: one prompt token and one generated token through **all 36 Qwen layers and the full head**. Produce 38 real proofs, reload them from disk and verify them with the independent bundle verifier. Then run a multi-token transcript to exercise cross-position attention, stop handling and reply feedback. Finally target the 500/250 workload. The layer-0/39 matched benchmark informs tuning and can run throughout. It is not a gate on building the other programs.

## Claude independent review prompt

Review this design and the referenced actual sources. For each claim mark **in code**, **proposed**, **assumption**, or **unsupported**. Return a table with severity, file or design section, smallest counterexample, client-visible consequence, minimal fix and falsifying test. End with the smallest independently verifiable full-model demo and your top three changes to implementation order.

1. Can a provider substitute a checkpoint, norm, numeric profile, child ELF, key or alternate input format while retaining every expected public field? Which party pins each value?
2. Follow one returned token backward through the head, layer 35, layer 0, embedding and the client's exact prompt. Where might an untrusted host value enter without authentication?
3. For P=1,R=1 and P=500,R=250, draw consumed and predicted positions. Try a future-token leak, off-by-one mask, GQA head mismatch, incorrect tie, early stop, or shortened vocabulary.
4. Challenge the optimized product and ordered adder at rounding ties, exponent changes, cancellation, signed zero, exceptional values and tile boundaries. Identify the evidence required to trust the dot_f8 state machine and GPU key used in a real proof.
5. Which work may be moved outside the guest or reused across requests without weakening the relation? For each proposed cache, name the commitment, its owner, and the equality checked when reopening it.
6. Compare one long backend-segmented execution against the 38-child bundle and recursive join. Identify the conditions under which tiling or parallelism makes total proving cost worse. Request a matched measurement that could overturn the preferred design.

## Source map

The original branch contains the native segment builder in `host/src/main.rs` and transformer operators in `common/src/lib.rs`. The ongoing matched experiment is in the separate sibling worktree `qwen-lossless-f8-projection/bench-projection`: `src/layer.rs` currently commits only the output hash; `src/lossless.rs` implements the optimized exact projection; `layer_input.py` is a host-side input adapter. This harness worktree does not contain an integrated copy of that branch. A reviewer outside this workspace needs both source snapshots plus the ELF, emulator and configuration hashes. In this worktree `harness/enroll.py` inventories the checkpoint. The Rust crate in `harness/receipts` defines canonical SHA-256 inputs, activation commitments and the bundle join. Its ProofBackend boundary requires an independently pinned proof verifier; only the tests supply a fixture backend so far. The ABI commits a 32-byte receipt digest, which binds the typed receipt carried with the proof. Executable join tests cover P=1,R=1 and P=500,R=250 with synthetic child receipts.

## First independent desk review

Claude identified three launch blockers from this specification. First, a layer must hash the actual input activation bytes inside the guest; test by changing A17 before generating a child and requiring the chain to fail. Second, publish an executable, serialized numeric profile with independent bitwise vectors before pinning a production ELF. Third, benchmark embedding and head at one prediction early: the tied BF16 tensor contains 777,912,320 bytes, already larger than the 532 MB guest heap observed on the projection benchmark. Use streaming or authenticated tiles and measure both memory and proving time. Run this track concurrently with layer work. This was a design review; backend constraints and deployed verifier keys still require independent review.

## Layer guest and binding (in code, emulator-tested, no proofs yet)

`harness/layer` is the layer relation shared by the guest and the native host: it parses record v1, hashes the 11 raw BF16 tensors and both activations in-guest, enforces the ordered-FP32 profile header byte-for-byte, runs `qwen_common::run_with` with projections claimed from the BF16 bytes (bitwise `dot_soft` on exact widening) and returns the receipt. `harness/layer-guest` commits only its 32-byte digest. `harness/layer-host` builds records from the checkpoint, runs ziskemu, writes a bundle, and verifies bundles with release values derived only from the client's enrollment inventory (stage weight roots, ModelID) and its own request. Receipt v2 adds the stage weight root; `Release` derives ModelID from the checkpoint ID and the 38 stage roots, and `verify_bundle` rejects a child whose weights differ from the pinned stage root. The verifier's backend reads emulator public outputs; it is not a proof verifier, and the embedding and head are still native host claims. Progress and measurements: the binding agent's notes (binding.md) and `~/scale-runs/pci-layer-binding/`.
