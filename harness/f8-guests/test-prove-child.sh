#!/usr/bin/env bash
# Exercise setup sharing, resume, and rejection without a GPU.
set -euo pipefail
runner=$(realpath "$(dirname "$0")/prove-child.sh")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
export PROVER=$PWD/prover PINNED_VERIFY=$PWD/verify PROVING_KEY=$PWD/key
leaf=key/zisk/vadcop_final/vadcop_final.verkey.json
mkdir -p pins receipts inputs elf "$(dirname "$leaf")"
echo '[1,2,3,4]' >setup.json
cp setup.json "$leaf"
echo layer >elf/layer
echo '{"setup_vk":[1,2,3,4],"programs":{"layer":[[0,0,0,0]]}}' >release.json
cat >prover <<'EOF'
#!/usr/bin/env bash
set -eu
if [[ $1 == program-setup ]]; then
    echo setup >>calls
    head -c32 /dev/zero >"$ZISK_CACHE_DIR/program.verkey.bin"
else
    cat "receipts/$(basename "${7%.stdin}")" >"${!#}"
fi
EOF
cat >verify <<'EOF'
#!/usr/bin/env bash
set -eu
[[ $(cat "$1") == "$4" ]]
echo "publics $4"
EOF
chmod +x prover verify
for n in layer00 layer01; do
    echo "$n" >"inputs/$n.stdin"
    printf '%0512d\n' 0 >"receipts/$n"
    sha256sum elf/layer "inputs/$n.stdin" "receipts/$n" setup.json >"pins/$n.sha256"
done
bash "$runner" "$PWD" layer00 & a=$!
bash "$runner" "$PWD" layer01 & b=$!
wait "$a"; wait "$b"
[[ $(wc -l <calls) == 1 ]]
bash "$runner" "$PWD" layer00
echo bad >proofs/layer00.proof.bin
if bash "$runner" "$PWD" layer00; then exit 1; fi
echo '{"setup_vk":[1,2,3,4],"programs":{"layer":[]}}' >release.json
if bash "$runner" "$PWD" layer01; then exit 1; fi
mv release.json release.saved
if bash "$runner" "$PWD" layer01; then exit 1; fi
echo 'PASS shared setup, verified resume, tampered proof, unapproved and missing release rejection'
