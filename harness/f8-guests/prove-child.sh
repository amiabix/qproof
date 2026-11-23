#!/usr/bin/env bash
# Run inside the machine's capped benchmark slot. WORK contains ELF, input,
# expected receipt, trusted setup.json, and a per-child sha256sum manifest.
set -euo pipefail
cd "${1:?usage: prove-child.sh WORK CHILD}"
child=${2:?child name}
case "$child" in
    embed) role=embed ;;
    layer[0-9][0-9]) role=layer ;;
    head-[0-9][0-9][0-9][0-9][0-9]) role=head ;;
    *) echo "invalid child name" >&2; exit 2 ;;
esac
: "${PROVER:?}" "${PROVING_KEY:?}" "${PCI_CLIENT:-${PINNED_VERIFY:?set PCI_CLIENT or PINNED_VERIFY}}"
release=${CLIENT_RELEASE:-release.json}
[[ -r $release ]] || { echo "missing approved release: $release" >&2; exit 1; }
sha256sum --check "pins/$child.sha256"
leaf=$PROVING_KEY/zisk/vadcop_final/vadcop_final.verkey.json
python3 - "$leaf" <<'PY'
import json, sys
if json.load(open(sys.argv[1])) != json.load(open("setup.json")):
    sys.exit("setup key differs from the client pin")
PY
# One setup per exact program and backend. Only publish a ready cache after setup succeeds.
identity=$(sha256sum "elf/$role" setup.json "$PROVER" | sha256sum | cut -d' ' -f1)
export ZISK_CACHE_DIR="$PWD/cache/$role-$identity"
mkdir -p cache proofs logs
exec 8>"cache/$role-$identity.lock"
flock 8
shopt -s nullglob
if [[ ! -f $ZISK_CACHE_DIR/.ready ]]; then
    if [[ -d $ZISK_CACHE_DIR ]]; then
        mv "$ZISK_CACHE_DIR" "$(mktemp -d "cache/failed-$role-XXXXXX")/incomplete"
    fi
    mkdir -p "$ZISK_CACHE_DIR"
    "$PROVER" program-setup --gpu --elf "elf/$role" --proving-key "$PROVING_KEY" >"logs/$child.setup" 2>&1
    keys=("$ZISK_CACHE_DIR"/*.verkey.bin)
    [[ ${#keys[@]} == 1 && -s ${keys[0]} ]] || { echo "expected one program key for $role" >&2; exit 1; }
    (cd "$ZISK_CACHE_DIR" && sha256sum "${keys[0]##*/}" >.ready.partial)
    mv "$ZISK_CACHE_DIR/.ready.partial" "$ZISK_CACHE_DIR/.ready"
fi
(cd "$ZISK_CACHE_DIR" && sha256sum --check .ready)
keys=("$ZISK_CACHE_DIR"/*.verkey.bin)
[[ ${#keys[@]} == 1 ]] || { echo "ambiguous program key for $role" >&2; exit 1; }
# Rental packages carry the release. A client can supply its independent copy.
python3 - "$release" "$role" "${keys[0]}" <<'PY'
import json, struct, sys
release = json.load(open(sys.argv[1]))
with open(sys.argv[3], "rb") as f:
    key = list(struct.unpack("<4Q", f.read()))
if release["setup_vk"] != json.load(open("setup.json")) or key not in release["programs"][sys.argv[2]]:
    sys.exit("program key is not approved for this role and setup")
PY
flock -u 8
exec 9>"proofs/$child.lock"
flock 9
expected=$(cat "receipts/$child")
[[ $expected =~ ^[0-9a-f]{64}0{448}$ ]] || { echo "bad receipt $child" >&2; exit 1; }
verify() {
    if [[ -n ${PCI_CLIENT:-} ]]; then
        "$PCI_CLIENT" proof --release "$release" --kind "$role" --proof "$1" --digest "${expected:0:64}" >"logs/$child.verify" 2>&1
        return
    fi
    "$PINNED_VERIFY" "$1" "$leaf" "${keys[0]}" "$expected" >"logs/$child.verify"
    grep -Fxq "publics $expected" "logs/$child.verify"
}
proof="proofs/$child.proof.bin"
if [[ -e $proof ]]; then
    verify "$proof"
else
    attempt=$(mktemp -d "proofs/.$child-XXXXXX")
    trap 'rmdir "$attempt" 2>/dev/null || true' EXIT
    { time -p "$PROVER" prove --gpu --minimal-memory \
        --elf "elf/$role" --inputs "inputs/$child.stdin" --proving-key "$PROVING_KEY" \
        --verify-proofs --output "$attempt/proof.bin" >"logs/$child.prove" 2>&1; } 2>"logs/$child.time"
    verify "$attempt/proof.bin"
    mv "$attempt/proof.bin" "$proof"
fi
echo "verified $child"
