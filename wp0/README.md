# WP0 frozen Qwen golden export

Run from the design worktree. The example uses two fresh sibling directories so the manifest records the same relative checkpoint root in each. The existing M0 environment supplies Python 3.12 and NumPy. Cargo resolves the native crates from the sibling Qproof tree and builds offline under the WP0 output directory.

```sh
qproof=../qwen-bf16-rmsnorm/Qproof
run=$qproof/target/latency.aBb7uA/run
model=../qwen-bf16-rmsnorm/target/full-gpu-attempt/instruct/model
py=target/m0/venv/bin/python
dest=target/wp0-rerun
mkdir -p "$dest"
wp0/build-exporter.sh "$qproof" "$dest/exporter"
for name in golden golden-repeat; do
    "$py" -B wp0/capture.py "$run" "$dest/exporter" "$dest/raw-$name" &&
    "$py" -B wp0/export.py "$run" "$model" "$dest/raw-$name" "$dest/$name" || exit 1
done
"$py" -B wp0/compare.py "$dest/golden" "$dest/golden-repeat" "$dest/determinism.json"
"$py" -B wp0/replay.py "$dest/golden" "$dest/replay" "$dest/parity.json"
```

## Reading the manifest

- Each projection entry links an input and output NPY, a stage, a tensor, and its exact checkpoint byte range. Add the tensor's shard path to the checkpoint root resolved relative to the manifest directory. Seek to the projection's absolute byte offset in that shard and read its byte length. The resulting BF16 bytes are row-major, little-endian, and covered by the projection SHA-256. The tensor entry covers the whole checkpoint tensor, including the full tied embedding underlying head shards.
- Projection inputs preserve the FP32 bits passed to the native kernel before BF16 rounding. Outputs preserve the BF16-rounded results widened to FP32. Array SHA-256 values cover complete NPY files, including headers. All activation and logit arrays are position-major.
- The 37 residual boundaries have shape `[39, 2560]`. Packaging verifies each adjacent pair against the layer's frozen child digest. For layers 0, 19 and 35 it additionally compares the record input and freshly executed output with those arrays.
- Each of the 75 heads consumes the same `[12, 2560]` final-normalized activation. Head array row 0 corresponds to consumed position 27. The stage entry gives the shard's first token ID, 12 shard winners, and their FP32 logit bits. Global winners use the first maximum in vocabulary order. The head logit digest covers the assembled little-endian FP32 payload in prediction-major, token-major order.
- Capture checks each native receipt and samples three full-width rows per matrix through the independent software dot_f8 kernel. The separate replay executable reads the exported NPY payloads and original checkpoint ranges and compares every output bit. Its JSON report gives the checked count per projection and the manifest digest.
