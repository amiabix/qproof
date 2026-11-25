#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: $0 QPROOF DEST_BINARY" >&2
    exit 2
fi
qproof=$(cd "$1" && pwd -P)
src=$(cd "$(dirname "$0")" && pwd -P)
scratch=$(mktemp -d "$(dirname "$2")/build.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
cp "$src/export.rs" "$src/replay.rs" "$scratch/"
# JSON strings also quote ordinary filesystem paths in TOML.
qpath=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1], ensure_ascii=False)[1:-1])' "$qproof")
cat > "$scratch/Cargo.toml" <<TOML
[package]
name = "exporter"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "exporter"
path = "export.rs"

[[bin]]
name = "replay"
path = "replay.rs"

[workspace]

[dependencies]
pci-layer = { path = "$qpath/harness/layer" }
pci-stages = { path = "$qpath/harness/client/stages" }
pci-dotf8 = { path = "$qpath/harness/dotf8" }
sha2 = "0.10.9"
TOML
cargo build --offline --release --manifest-path "$scratch/Cargo.toml" --target-dir "$scratch/target"
install -m 755 "$scratch/target/release/exporter" "$2"
install -m 755 "$scratch/target/release/replay" "$(dirname "$2")/replay"
