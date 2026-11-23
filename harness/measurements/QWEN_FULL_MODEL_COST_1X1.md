# Full Qwen3-4B emulator comparison

One prompt token (ID 2), one greedy prediction (ID 82), original BF16 checkpoint, pinned Mac CPU arithmetic. Each measured variant covers the embedding, 36 transformer layers and the complete 151,936-entry output head over 297 shards. All public outputs matched the archived transcript.

| Configuration | Guest steps | Sum of emulator costs | dot_f8 calls |
|---|---:|---:|---:|
| Original layer kernel, software head | 385,092,565,253 | 35,995,104,700,156 | 1,105,920 |
| Cleaned layer kernel, software head | 353,324,117,122 | 33,393,382,661,462 | 1,105,920 |
| Cleaned layer kernel, candidate head | 293,320,487,209 | 27,730,939,867,692 | 1,257,856 |

| Stage | Guest steps | Sum of emulator costs |
|---|---:|---:|
| embed | 81,980 | 297,508,159 |
| baseline layers | 296,114,448,539 | 27,506,449,055,842 |
| cleaned layers | 264,346,000,408 | 24,904,727,017,148 |
| software head | 88,978,034,734 | 8,488,358,136,155 |
| candidate head | 28,974,404,821 | 2,825,915,342,385 |

The pinned Hugging Face reference matched all 37 activation boundaries. A diagnostic head guest independently committed the raw logits of each candidate shard. All 151,936 scores matched bit for bit. That diagnostic ran separately and is excluded from the cost table.

Costs sum each invocation’s ZisK cost-v2 TOTAL, including the fixed base once per guest. The two head implementations were measured on identical inputs. These results describe one consumed position and one prediction. Longer contexts change the balance between attention, the projections and the head. No recursive verification or proof generation was timed.

Local raw measurements are in `target/full-cost-1x1/metrics.jsonl` and the individual emulator logs in the same directory.

## GPU proving handoff

The one-position request has 334 child records. The embedding, decoder and head each have a distinct ELF. The checked-in head entry point is `harness/f8-guests/src/bin/head_batched.rs`. Its locked release build returned the expected receipt on all 297 vocabulary shards. The emulator costs above used the earlier experimental head ELF.

| Program | SHA-256 of staged ELF |
|---|---|
| Embedding | f33a05b76307982456090d183001ed90646af0f3d837023f13453135d7357cca |
| Decoder | 298d27628584331837557e1588866b5cd3daf252c388d3b3e84be3812e7322ef |
| Output head | d694451e49dd3c15b2f2403f98484b28f9ea9c3606236db9a8162725640f8ff9 |

The Omarchy run directory is `~/scale-runs/qwen-bf16-clean-1x1-20261123`. The pilot queues the embedding, head shard 0 and decoder layer 19 through the shared capped job slot. Layer 19 had the highest measured cleaned-layer emulator cost. The pilot checks the pinned inputs and ELF, runs program setup, generates a GPU proof, reloads the saved artifact and compares the complete public output with the archived receipt. Each verified artifact is published under `proofs/CHILD.proof.bin`. Monitor `pilot.log` and `logs/CHILD.prove`.

The Mac verifier needs the BLAKE3 verifier generated for Omarchy's GPU setup. Its `verifier/src/blake3/vadcop_final.rs` SHA-256 is `8549abba5164a26ccc6b507c13197971ce5dfdfaa706e9232e0a7b5ad468ab25`. Verified against a previously saved dot_f8 GPU proof: the matching verifier accepted that proof, while a changed proof byte, wrong program role and wrong digest each failed. The resulting Mac client binary is `target/proof-control/build/release/pci-client`. The control release is test-only. For the Qwen release, derive the three program keys from the staged ELFs, then use the existing client release, assemble and verify commands.

## Decoder rounding experiment

A full-function profile of the staged layer 19 guest attributed 95.49% of variable cost to the seven projections, including 67.29% of layer variable cost in their accumulation loop. Percentages are inclusive. The dot_f8 syscall accounted for 2.12%. Across all 36 staged layers, the dot_f8 opcode accounted for 541,433,733,120 units, or 2.17% of total emulator cost.

The experimental guest skips rounding when aligning a normal BF16 product discards only zero bits. Two eight-bit significands produce at most 16 significant bits, leaving at least eight trailing zeros in the 24-bit normalized FP32 significand. For exponent gaps from one to eight, the aligned increment is exact in the existing fast path. The accumulator order and exceptional-value paths are unchanged.

| Full layer, one position | Staged cost | Experimental cost | Reduction |
|---|---:|---:|---:|
| 02 | 648,674,714,181 | 614,772,614,309 | 5.23% |
| 18 | 703,319,459,193 | 658,379,832,003 | 6.39% |
| 19 | 709,618,992,522 | 664,688,368,859 | 6.33% |

Each guest made 30,720 dot_f8 calls and returned the archived 256-byte public output. All seven three-position projection samples matched their reference digests. This experiment used a separate ELF. Full-decoder savings and GPU proof time for the experiment remain unmeasured.
