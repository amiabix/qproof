"""Run the real preflight and package code on a tiny synthetic 112-child request."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

script = Path(__file__).with_name('preflight.py')
with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    for folder in ('elf', 'inputs', 'shards'):
        (run / folder).mkdir()
    for role in ('embed', 'layer', 'head'):
        (run / 'elf' / role).write_text(role)
    digest = 'ab' * 32
    x = dict(model='m', suite='s', request='q', positions=1, reply=[4], predictions=1,
             predicted=[4], a0_root='0', a36_root='36', children=[])
    for i in range(36):
        name = f'layer{i:02}'
        layer = dict(model_id='m', suite_id='s', request_id='q', positions=1, layer=i,
                     digest=digest, receipt=dict(input=str(i), output=str(i+1), index=i))
        (run / f'inputs/{name}.json').write_text(json.dumps(layer))
        (run / f'inputs/{name}.v1.stdin').write_bytes(bytes.fromhex(digest))
    for i in range(-1, 75):
        name = 'embed' if i == -1 else f'head-{i:05}'
        receipt = dict(output='0') if i == -1 else dict(
            input='36', shard=i, shard_log2=11, positions=1, predictions=1)
        x['children'].append(dict(name=name, digest=digest, receipt=receipt))
        (run / f'shards/{name}.stdin').write_bytes(bytes.fromhex(digest))
    expected = run / 'shards/expected.json'
    expected.write_text(json.dumps(x))
    release = run / 'release.json'
    release.write_text(json.dumps(dict(derived=dict(model='m', suite='s'),
                                       head_shard_log2=11, setup_vk=[1], programs={k:[[0]*4] for k in ('embed','layer','head')})))
    (run / 'setup.json').write_text('[1]')
    (run / 'inputs/bundle.json').write_text('{}')
    (run / 'request.json').write_text('{}')
    emulator = run / 'emu'
    emulator.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do case "$1" in -o) out=$2;; -i) input=$2;; esac; shift; done\ncat "$input" >"$out"\ndd if=/dev/zero bs=224 count=1 >>"$out" 2>/dev/null\necho "STEPS 2"; echo "TOTAL 5"\n')
    emulator.chmod(0o700)
    # The real client joins cryptographic receipts in the Qwen integration run.
    client = run / 'client'
    client.write_text("""#!/bin/sh
set -eu
case "$1" in
    assemble) echo assemble >"$(dirname "$0")/client-call";;
    check-prepared) test "$(cat "$(dirname "$0")/client-call")" = assemble
        shift; while [ "$1" != --digests ]; do shift; done
        test "$(wc -l <"$2")" -eq 112;;
    *) exit 1;;
esac
""")
    client.chmod(0o700)
    args = [sys.executable, script, run, release, emulator, '--client', client]
    out = subprocess.check_output([*args, '--package', run / 'package', '--setup', run / 'setup.json'])
    assert json.loads(out) == dict(children=112, steps=224, cost=560)
    assert len((run / 'package/roster').read_text().splitlines()) == 112
    assert (run / 'package/layers.json').is_file()
    if shutil.which('shasum'):
        subprocess.run(['shasum', '-a', '256', '-c', 'pins/head-00074.sha256'], cwd=run/'package', check=True)
    (run / 'shards/head-00074.stdin').write_bytes(bytes(32))
    res = subprocess.run(args, capture_output=True, text=True)
    assert res.returncode and 'head-00074 receipt' in res.stderr
    x['children'].pop()
    expected.write_text(json.dumps(x))
    res = subprocess.run(args, capture_output=True, text=True)
    assert res.returncode and 'incomplete or unordered head' in res.stderr
    print('PASS export, changed tail input, missing tail shard')
