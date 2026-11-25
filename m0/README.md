# Native Qwen M0 coverage measurement

Run from the design worktree. The exporter links the active Qproof crates read-only. It executes the pinned layer or head function and checks its receipt against the frozen roster before returning operands and oracle outputs. Python 3 and NumPy are required.

```sh
mkdir -p target/m0
QPROOF=../qwen-bf16-rmsnorm/Qproof
m0/build-exporter.sh "$QPROOF" target/m0/exporter
uv venv --python 3.12 target/m0/venv
uv pip install --python target/m0/venv/bin/python numpy
target/m0/venv/bin/python -m unittest discover -s m0 -v
target/m0/venv/bin/python m0/run.py "$QPROOF/target/latency.aBb7uA/run" target/m0/exporter target/m0/report.json
```

The final command is the owner-run measurement. It checks 21 layer matrices (0, 19 and 35) and all 75 head shards. The report prints matrix results as it runs and writes the combined gate after all checks pass. Select a fresh output filename for a rerun. The frozen transcript proposes its 12 winning token IDs. The native oracle supplies the proposed winning score; a deployed proof must establish each winner through the reported certificate or checker path. The runner checks every actual shard for a greater value or an earlier ID tied with the winner.

The implementation uses the pasted Python reference's five integer products, including the extra truncation-indicator cross term, and its conservative (K+32) underflow allowance. It uses exact rational comparisons with integer-safe exponents, handles all-zero rows, checks signed Goldilocks bounds against the actual prime, and uses the true signed BF16 midpoint for head elimination. The oracle is read only after the eligibility decision for each layer output. E1 rejects the entire call. E2, E3 and E4 count the first condition that failed.

For layers, avoided FMAs count certified rows. For the head, they count certified winners and strictly eliminated losers. Every overlapping loser is charged to the checker, including ties. A false certification or false elimination aborts the measurement. The combined denominator is exactly the three sampled layers plus the complete head. Extrapolating it to all 36 layers requires an explicit workload model. Proof times, sumcheck costs, and actual memory-planner slot counts are separate measurements.
