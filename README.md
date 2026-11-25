# qproof

Proof Carrying Inference for Qwen3-4B on the ZisK zkVM. The harness uses the original BF16 weights and the `dot_f8` precompile to check inference against a pinned Hugging Face reference.

The 32-token prompt with eight predictions matched all 37 activation boundaries and every logit in native execution. On the 500-token prompt with 250 predictions, all greedy token IDs matched, but decode scores diverged. Full-model GPU proof generation remains unverified.

- [Harness architecture and request verification](FULL_MODEL_HARNESS.md)
- [Reference configuration and numerical parity](harness/measurements/QWEN_BF16_PARITY.md)
- [Guest execution costs](harness/measurements/QWEN_FULL_MODEL_COST_1X1.md)
- [Build the client and run the preflight](harness/measurements/RENTAL_READINESS.md)

## Decoder proof experiment

Start with the [layered plan](QPROOF_LAYERED_PLAN.md). The frozen 39-position request has two verified children (embedding and head shard 74) and zero verified decoder layers. Layer 0 ran in ASM and exhausted one planner worker’s memory-access slots during proving. Slot use by address range remains unmeasured.

The [golden-data tools](wp0/README.md) reproduced 5,417,472 native projection and head outputs bit for bit. The [current certificate](references/m0_certificate_reference.py) covered 2,312 of 2,496 outputs on the first 64 rows of layer 0’s key projection. All certified bits matched the native oracle. The 184 remaining outputs require exact replay. The tile measurement covers certificate eligibility. Guest integration, verified proof time and full-layer memory use are still open.

For review, trace weight-hint validation through the encodings, four integer sums, [random-vector verifier](references/freivalds_check.py), BF16 rounding interval and exact fallback. Compare the proposed guest against the same tile in the original guest. The five-product prototype under `m0/` and the earlier state-machine design brief are historical experiments.

Run the synthetic checks with Python 3 and NumPy:

```sh
python3 -I references/m0_certificate_reference.py
python3 -I references/freivalds_check.py
python3 -m unittest discover -s references
```

With the frozen export and referenced checkpoint available locally, reproduce the measured tile with `python3 references/check_tile.py path/to/golden`.
