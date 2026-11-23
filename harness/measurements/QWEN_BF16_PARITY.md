# Qwen3-4B BF16 serving parity

The reference is the original Qwen3-4B-Instruct-2507 checkpoint on an Apple M4 Pro, macOS 27.2.1, Transformers 5.15.1 and PyTorch 2.13.0 at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`. It uses six CPU threads, BF16 tensors, SDPA, a single unpadded prompt and greedy decoding with a dynamic KV cache. The enrollment inventory checkpoint ID is `e34a1f836d5d1f2ba33497f4802da098f952683ab8a807c938baf023a0cf6b3b`.

## Numerical checks

The parity checker runs the prompt through the reference once, then feeds back each reference token with its cache. It compares the reference embedding and all 36 layer outputs against the harness commitments, and compares every logit as raw bits.

| Prompt tokens | Predictions | Exact activation commitments | Exact logits |
|---:|---:|---:|---:|
| 1 | 1 | 37 / 37 | 151,936 / 151,936 |
| 3 | 2 | 37 / 37 | 303,872 / 303,872 |
| 32 | 8 | 37 / 37 | 1,215,488 / 1,215,488 |
| 500 | 250 | 1 / 37 | 11,075,865 / 37,984,000 |

The 500-token prefill matched at every layer. Across the complete transcript, 26,908,135 logits differed even though all 250 greedy token IDs agreed. The first decoder block first differed at context length 574, with 665 of 2,560 values differing in that row. The operation responsible is still under investigation.

The three-token prompt is `[2,36518,25]` and its predictions are `[9645,264]`. Before the fix, none of the logits at either position matched. The 32-token prompt and its eight greedy predictions are in `qwen39-execution.json`.

Use Python from an environment with the pinned reference dependencies. Pass the original prompt IDs and the directory produced by `pci-layer-host run`. A multi-token test needs the supplied transcript in the host's `--reply` argument.

```sh
python harness/layer/check_bf16_reference.py "$model_dir" "$host_output" --prompt 2,36518,25
```

The regression covers rounding at normalization, projection, rotary embedding, softmax and value mixing, both residual additions, SiLU and the output head. The BF16 linear kernel uses the reference's 32 FP32 accumulation lanes. The attention kernel reproduces the prompt and cached execution shapes using the prediction count bound by every layer receipt. Real weights remain BF16 throughout enrollment.

## Complete emulator transcript

For the one-token prompt `[2]`, ZisK emulated the embedding, all 36 full-size transformer layers and all 297 vocabulary shards. The client join checked all 334 guest output digests, their request and layer links, and the greedy prediction `[82]`. The proof verifier rejected the same bundle when supplied without proofs.

| Stage | Measured guest steps |
|---|---:|
| Embedding | 81,980 |
| 36 decoder layers | 296,114,448,539 |
| 297 head shards | 88,978,034,734 |
| Total | 385,092,565,253 |

These counts include the computation and authentication inside each guest. They exclude any STARK proving or recursive verification. The longer cases in the numerical table used the native harness.

## Emulator measurement

For the layer-0 query projection sample (16 checkpoint rows, three positions, 2,560 inputs per row), skipping the zero low significand of widened BF16 activations reduced `dot_f8` calls from 96 to 48. On the same emulator and inputs, steps changed from 9,951,734 to 9,910,173 and total emulator cost from 1,196,798,572 to 1,170,605,297. The output digests matched the reference. The seven projection samples covered 336 real outputs in total.

These runs measure numerical parity and guest execution. They do not measure proving time. A different backend, device or arithmetic profile needs its own comparison.

## Lossless BF16 guest cleanup

The batched projection now skips the empty low chunk of each BF16 input during preparation and accumulation. A normal BF16 product has at most 16 significant bits, so its FP32 significand needs no rounding. The guest still rounds each accumulation in the reference order and retains the fused fallback for exceptional inputs.

The matched comparison used the same ZisK cost-v2 emulator, original checkpoint records and public output digest for both ELFs. For each of the seven layer-0 matrices, 16 rows at one and three positions matched the reference digest (448 output values across 14 samples). Across those samples guest steps fell 13.27–17.16% and total emulator cost fell 6.52–14.01%.

| Query sample, three positions | Guest steps | Total emulator cost |
|---|---:|---:|
| Previous lossless kernel | 9,903,497 | 1,170,065,841 |
| Cleaned lossless kernel | 8,349,686 | 1,043,584,295 |
| Software BF16 reference in the same sampling guest | 28,014,424 | 2,891,337,417 |

In the cleaned query sample, the guest function that reconstructs products and accumulates them in reference order accounts for 599,715,237 of 758,371,623 variable cost units (79.08%). This includes 162,103,172 in its integer addition helper. The 48 dot_f8 calls account for 18,323,712 (2.42%). The percentages exclude the fixed emulator base cost and describe this projection sample.

The deployed head guest calls the software reference kernel. A local experimental head ELF instead passed the cleaned kernel through the existing head stage's `execute_with` interface. Both variants ran shard authentication, the full shard scan and receipt generation against the same input on the same SDK. The first, middle and final shards of the one-position transcript produced the archived receipt bytes in both variants.

| Head shard | Reference steps | Candidate steps | Reference emulator cost | Candidate emulator cost |
|---|---:|---:|---:|---:|
| 0 | 299,484,300 | 98,235,727 | 28,563,698,294 | 9,567,724,693 |
| 148 | 299,836,993 | 97,454,055 | 28,604,377,666 | 9,508,114,793 |
| 296 | 227,201,900 | 68,503,306 | 21,751,033,535 | 6,835,071,458 |

Adopting this head ELF for proof generation requires updating the approved program key. The three shard measurements are an experiment. Full-head and full-request timings remain open.

On the complete Qwen layer 0 at one position, including weight and activation authentication, the previous guest took 8,422,383,214 steps and 780,098,620,885 total emulator cost units. The cleaned guest took 7,523,548,519 steps and 706,419,255,695 cost units. Both 256-byte public outputs matched the archived expected layer receipt with zero padding. The input was `target/parity-emu/layer00.v1.stdin`. Run `ziskemu -X --max-input-mem 268435456` with each layer ELF and this same input to reproduce the comparison. Raw reports are under `target/cost-review/` locally.
