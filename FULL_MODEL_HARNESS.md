# How the Qwen3-4B proof harness works

qproof is designed to check whether a greedy reply follows from the client's prompt tokens, the original Qwen3-4B-Instruct-2507 BF16 weights, and a specified arithmetic profile. The harness takes a finished transcript and prepares separate ZisK jobs for the embedding, decoder layers and output head. A verified end-to-end Qwen GPU proof remains outstanding.

## How closely does the harness match Qwen?

| Native comparison with the pinned Hugging Face BF16 reference | Result |
|---|---|
| 32 prompt tokens, 8 predictions | All 37 activation boundaries and all 1,215,488 logits matched bit for bit. |
| 500 prompt tokens, 250 predictions | Every layer matched across the prompt. All 250 greedy token IDs agreed, but 26,908,135 of 37,984,000 logits differed across the predictions. |

On the longer transcript, decoder layer 0 first diverged at context length 574. At that position, 665 of its 2,560 output values differed. The operation responsible is still under investigation. A verified guest execution would certify the guest’s own arithmetic. The native comparison checks whether that arithmetic reproduces the reference. [Reference profile and measurements](harness/measurements/QWEN_BF16_PARITY.md)

## From request to child jobs

Let P be the number of prompt tokens and R the number of reply tokens. The model consumes the prompt followed by the first R−1 reply IDs, giving T = P + R − 1 positions. The head predicts at positions P−1 through T−1. For 500/250, these are 749 consumed positions and 250 predictions.

```text
Client's release pins + nonce + prompt IDs + decoding limits → RequestID

Prompt and reply prefix
        ↓
Embedding → A0 → layer 0 → A1 → … → layer 35 → A36
                                                  ├─ head shard 0 → local winners
                                                  ├─ head shard 1 → local winners
                                                  └─ … last shard → local winners

Client verifies the evidence, links the receipts, combines winners
across the entire vocabulary, and checks the reply and its termination.
```

| Guest | What it computes | What links it to the request |
|---|---|---|
| Embedding | Opens indexed rows of the tied BF16 matrix and produces A0 for the consumed IDs. | The receipt binds the consumed-ID commitment, embedding weights and output root. |
| Decoder, reused for layers 0–35 | Reads the preceding activation tensor and this layer’s seven weight matrices and four norm tensors. Runs normalization, Q/K/V projections, rotary positions, causal grouped-query attention, output projection, residual additions and the gated MLP. | The guest hashes the weight bytes it uses and both activation tensors. Its receipt binds the layer index, positions, prediction window and roots. |
| Vocabulary shard, reused for every range | Authenticates its rows under the same tied-matrix root, hashes A36, applies final normalization and scores the shard at every prediction position. | Each shard binds its range, head weights, input root and winning token ID and score per prediction. The client checks coverage and resolves ties by lowest token ID. |

A layer hands the next layer **one table of numbers**, called the residual stream. Each of its T rows represents one token position and holds 2,560 values. For 749 positions the handoff contains 7,669,760 bytes in the harness's four-byte activation encoding. The host passes the actual values. The client checks the corresponding 32-byte commitments. Keys and values for attention are computed within each layer over the transcript. [Layer host](harness/layer-host/src/main.rs) · [Decoder guest](harness/f8-guests/src/bin/layer_batched.rs) · [Layer relation](harness/layer/src/lib.rs)

The lossless `dot_f8` kernel uses the precompile for partial products and reconstructs the serving profile’s BF16 projection results with ordered accumulation. Q/K/V share one prepared input, as do gate/up. Attention uses the software implementation of the pinned Mac BF16 profile. The batched head also calls the lossless kernel. [Projection kernel](harness/dotf8/src/lib.rs) · [Attention](common/src/attention_bf16.rs) · [Head guest](harness/f8-guests/src/bin/head_batched.rs)

### How many proofs?

Each request has one embedding child, 36 decoder children and one head child per vocabulary shard. The vocabulary has 151,936 entries. With 2,048 rows per shard, that gives **112 children** (1 + 36 + 75). The 512-row configuration has **334 children** (1 + 36 + 297). At a fixed shard width, a longer transcript increases the work and input sizes inside the children while the child count stays constant. Native witness generation follows the decoder in order. Once their inputs exist, individual child jobs can run independently. [Embedding and head records](harness/client/stages/src/lib.rs) · [Job preparation and replay](harness/f8-guests/preflight.py)

## What the client pins and checks

- **ModelID** hashes the enrolled checkpoint ID, the layer-profile template and ordered stage weight roots. The measured Instruct-2507 checkpoint and reference versions are recorded in the parity report. The checked-in base-Qwen release describes a different checkpoint. [Model identifier](harness/receipts/src/lib.rs) · [Measured checkpoint](harness/measurements/QWEN_BF16_PARITY.md)
- **SuiteID** hashes the suite description and vocabulary shard width. The client’s independent release separately supplies the setup verification key, approved program keys per role, tied-matrix row root and final norm digest. The client can derive the row tree from the enrolled weights. [Release builder](harness/client/cli/src/main.rs) · [Release pins](harness/client/cli/src/lib.rs)
- **RequestID** binds ModelID, SuiteID, the client’s nonce, the counted prompt IDs, maximum reply length, context limit and stop IDs under the greedy decoding policy. The client starts with token IDs. Tokenizing a conversation and rendering text are outside the guest. [Request encoding](harness/receipts/src/lib.rs)

Each guest produces a receipt describing what it read and computed. Its public output is a 32-byte hash of that receipt followed by zero padding. With a proof bundle, the client verifies each proof under the approved setup and program key and checks its public bytes against the receipt. The join checks the exact roster, weight commitments, prediction windows, prompt and feedback IDs, and every activation link. It combines the shard winners and requires the supplied reply to end at the first permitted stop or the requested length limit. [Bundle verifier and join](harness/client/cli/src/lib.rs)

The compact path folds child proofs in canonical roster order. The client verifies the aggregate under an independently pinned recurser key, recomputes the ordered accumulator from the typed receipts and performs the same request-level join. Tests cover the fold order. Verification of a complete Qwen aggregate on GPU remains outstanding. [Aggregation host](harness/client/aggregate-host/src/main.rs) · [Recursion circuits](harness/client/recursion/circuits) · [Compact verifier](harness/client/cli/src/lib.rs)

## How to interpret a result

| Check | Evidence it produces |
|---|---|
| Native reference comparison | Bitwise agreement of the model intermediates and logits on a named transcript. |
| Prepared-receipt check and ZisK emulator replay | Agreement with the client’s request and expected guest public bytes, plus measured guest steps and emulator cost. |
| Proof generation and verification | Cryptographic evidence for the program and public bytes. Full-model proof time, memory and compact verification require a completed proving run. |

The one-position benchmark covered the embedding, all decoder layers and the entire vocabulary. Using 297 head shards, the candidate configuration measured 293,320,487,209 guest steps and 27,730,939,867 emulator cost units. These figures apply to one consumed position and one prediction. The 500/250 decode mismatch remains the blocker for a reference-matching full-size run. [Matched emulator results](harness/measurements/QWEN_FULL_MODEL_COST_1X1.md) · [Preflight and proving commands](harness/measurements/RENTAL_READINESS.md)
