# Qwen3-4B GPU rental preflight

The candidate uses the original Qwen3-4B-Instruct-2507 BF16 checkpoint and the lossless dot_f8 guest. The first measured transcript consumes 39 positions from 32 prompt tokens and eight greedy predictions. A separate capacity run uses the original 500-token benchmark prompt and permits up to 250 outputs, stopping at the model's stop IDs.

## What has been checked

- The 39-position native comparison against pinned Mac arm64 Hugging Face BF16 matched all 37 activation boundaries and all 1,215,488 logits. The eight output IDs were 13, 3776, 1899, 11, 1340, 1730, 264 and 25382.
- Embedding, decoder and output-head RISC-V executables were built. The decoder calculates rotary angles once per position and reuses them across all heads. The native test for the pinned rotary values and the full-width batched decoder test both passed.
- Same-input decoder measurement, layer 19 at one position: 7,092,624,615 to 7,091,973,637 guest steps; 664,688,368,859 to 664,634,201,928 emulator cost units (0.00815% lower). Both emitted the same public bytes. ELF SHA-256 changed from a9b285dc1ab5d5a2461a331e8aded0d26a9cc2b42f8b2098eae9c7af3be0f177 to d0abdd444d64a6ae4d1c15f6157657785ed6d227ca8601cbe3bceec7ac21fe54.
- With 2,048 vocabulary rows per head shard, the client assembled 112 receipts (one embedding, 36 decoder blocks and 75 vocabulary shards). The production client checked the prepared digests for the entire roster using its own request. Changing only the nonce was rejected at the first receipt. Proof verification rejected the resulting bundle with all proofs absent. An independent end-to-end replay of each RISC-V executable against its expected public receipt is required before export.
- A Linux test with a mock prover exercised two concurrent layers, one shared setup, saved-proof verification and tampered-proof rejection. A synthetic 112-child export checked the last shard, its transfer checksums, a changed input and an incomplete roster.

## Reproduce the local checks

From the task worktree with Python 3.11 or later:

```sh
python3 harness/f8-guests/test-preflight.py
# Linux, using flock and GNU coreutils
bash harness/f8-guests/test-prove-child.sh
```

For a prepared transcript, RUN contains elf/embed, elf/layer, elf/head, request.json, the layer host's inputs directory, and the client's shards directory. Prepare both with the same release, shard width, nonce, prompt and reply. Keep the input directory immutable. Run the reference checker with the pinned Torch and Transformers environment before exporting.

```sh
python3 harness/layer/check_bf16_reference.py "$MODEL" "$RUN/inputs" --prompt "$PROMPT_IDS"
python3 harness/f8-guests/preflight.py "$RUN" "$RELEASE" "$ZISKEMU" \
    --client "$BUILD/release/pci-client" --workers 3 --setup "$SETUP_JSON" --package "$RUN/rental-work"
```

Each completed child writes its steps, emulator cost and hashes to RUN/preflight. The runner checks the guest's 256 public bytes, including padding. Repeat the command to resume. Use a new export destination when a package already exists. A final total requires every child.

## GPU execution and verification

Build the client against the generated verifier for the intended setup. The portable builder was tested on Mac against an archived GPU control proof and a tampered copy. The control verified and the tampered copy failed STARK verification.

```sh
python3 harness/client/build-tools.py client "$BACKEND" "$BUILD"
# On the Linux prover:
python3 harness/client/build-tools.py aggregate-host "$BACKEND" "$BUILD"
export PCI_RECURSION_CIRCUITS="$PWD/harness/client/recursion/circuits"
```

The 112 guest inputs occupy 8,089,899,472 bytes at 39 positions. The largest is 202,261,176 bytes. The pinned Omarchy proving-key directory has 23,103,484,891 apparent bytes. These sizes exclude build caches, proof outputs and working memory.

Transfer the package, the pinned Linux prover, matching dot_f8 setup and proving key, and the trusted verification binaries. On the rental, the package's prove-child.sh accepts the package directory and a child name from roster. Set PROVER, PROVING_KEY and PCI_CLIENT (the release client built above). Verify the setup provenance and approve the three exact ELF-derived program keys independently before trusting any result. Pass the independent release through CLIENT_RELEASE. The runner checks that the generated program key is approved for the role before proving. The fixture release carries empty role lists and will reject proof generation until supplied with approved keys.

Once all child proofs verify, assemble and verify them with the client.

```sh
"$BUILD/release/pci-client" assemble --expected "$WORK/expected.json" \
    --layers "$WORK/layers.json" --proofs "$WORK/proofs" --out "$WORK/bundle"
"$BUILD/release/pci-client" verify --release "$INDEPENDENT_RELEASE" \
    --request "$CLIENT_REQUEST" --bundle "$WORK/bundle"
```
To produce one compact proof, the aggregation host accepts EMBED.elf LAYER.elf HEAD.elf --work WORK OUT_DIR. Its binary folds the leaves in client roster order. Verify the aggregate using the independent recurser key and the same request. Until a real aggregate passes that final check, compact GPU execution remains unverified.

## What determines when to rent

1. Exact native reference agreement for the selected rental transcript.
2. Complete emulator replay and a resumable package for the same input and ELFs. Check the longest input fits the guest heap.
3. The Linux aggregate host builds against the exact SDK. The matching setup, verifier, GPU driver and proving key are available on the destination.
4. The first paid run measures proof time, peak VRAM and peak host RAM. Guest steps and emulator cost measure different quantities.

Current raw evidence and ongoing logs are under target/rental-readiness. The separate target/full-cost-39-round run uses the previous decoder binary and 512-row shards. Keep its counters separate.

The 500/250 workload fails the reference-agreement check during decode. GPU proof timing for that workload is pending. See the [BF16 parity measurements](QWEN_BF16_PARITY.md).
