# Full-depth harness benchmark, 22 November 2026

## Layer boundaries and execution cost

Same Mac, guest ELF (SHA-256 862d280352dc1fc4020daf4d1804447e80d9e3b853fad6d4e927ffe16f4142c8), emulator, six supplied positions and original synthetic BF16 checkpoint in all four runs. Each run traverses an embedding, 36 layers and the full output head. Width 64, MLP 192, head dimension 16, vocabulary 512. Each segment hashes both its input and its output. Single execution checks all 37 activation links inside the guest. The host independently checks stage outputs and the combined transcript digest. Quantized and lossless are different arithmetic profiles.

| Arithmetic | Emulator executions | Guest steps | Emulator cost |
|---|---:|---:|---:|
| Quantized FP8, causal scaling | 38 | 168,457,913 | 27,313,782,341 |
| Quantized FP8, causal scaling | 1 | 168,442,176 | 16,759,056,899 |
| Lossless dot_f8 | 38 | 1,584,942,703 | 166,640,976,837 |
| Lossless dot_f8 | 1 | 1,584,926,966 | 156,086,270,763 |

The emulator's BASE charge is 285,212,672 per execution. Running 37 additional executions adds 10,552,868,864 from that line alone. The combined fixture still supplies intermediate activations and checks them in the guest. A streaming production guest could have different memory and serialization costs. Neither layout timed a prover or verified recursive child proofs.

## Correctness checks

Appending a sixth token previously changed earlier quantized predictions because the attention value scale read the entire column. Only the embedding passed prefix replay. Selecting the scale from the causal prefix fixes all 38 stages. On the fixture, five of six argmaxes changed after the fix. Lossless passed the prefix replay before the fix. The host also exercises greedy feedback and prefix lengths around eight-element block boundaries. A one-bit mutation to the first layer's input trapped in the combined guest at step 43,394,074.

The receipt join tests cover all 38 stage slots. In a separate 500-prompt/250-reply fixture, checking the receipt and transcript links consumed 770,272 steps and 364,031,833 cost units. Its fixture verifier takes supplied child digests. This measures linkage only. Production acceptance requires a pinned key, authenticated child proofs and, for continued sessions, authenticated prior state.

## Real checkpoint coverage

The adjacent qwen39-execution.json contains the audit of all 38 archived Qwen3-4B stages for 32 prompt tokens and eight next-token choices. Cost was 809,902,301,109,087 for ordered FloatDot and 7,677,350,849,543 for historical quantized FP8. Saved logits give the same eight greedy IDs. The archived quantized run predates the causal-scaling fix. Neither archive measures lossless dot_f8. For fresh matched full-checkpoint runs use the dedicated benchmark worktree's bench-projection/full_matched.sh under the Omarchy cap. Its 1+1 workload measures all weights and layers but cannot measure growth with context length.

## Reproduction

The benchmark worktree contains bench-projection/src/smoke.rs and bench-projection/smoke-host. Run the release smoke-host with the guest ELF, dot_f8-enabled ziskemu binary and a fresh output directory as its three arguments. The report contains individual segment counts, native comparisons, prefix replay and combined execution measurements. In this worktree, harness/audit_execution.py audits the retained full-size segment and logits artifacts. The join measurements in this directory are independent of the arithmetic benchmark.
