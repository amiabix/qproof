"""Replay a complete prepared Qwen request before renting a prover.

Usage: python3 preflight.py RUN RELEASE ZISKEMU --client PCI_CLIENT [--workers N] [--package DIR --setup VK.json]
RUN contains elf/{embed,layer,head}, inputs/layerNN.{json,v1.stdin}, and
shards/{expected.json,*.stdin}. Keep RUN immutable during execution.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stages(run, release):
    transcript = json.loads((run / 'shards/expected.json').read_text())
    require(transcript['model'] == release['derived']['model'] and
            transcript['suite'] == release['derived']['suite'], 'release mismatch')
    require(transcript['predicted'] == transcript['reply'] and len(transcript['reply']) == transcript['predictions'],
            'reply mismatch')
    shard_log2 = release['head_shard_log2']
    rows_per_shard = 1 << shard_log2
    count = (151_936 + rows_per_shard - 1) // rows_per_shard
    names = ['embed'] + [f'head-{i:05}' for i in range(count)]
    require([c['name'] for c in transcript['children']] == names, 'incomplete or unordered head')
    embed, *heads = transcript['children']
    root = embed['receipt']['output']
    require(root == transcript['a0_root'], 'embedding boundary')
    yield 'embed', 'embed', run / 'shards/embed.stdin', embed['digest']
    for i in range(36):
        name = f'layer{i:02}'
        layer = json.loads((run / f'inputs/{name}.json').read_text())
        receipt = layer['receipt']
        require((layer['model_id'], layer['suite_id'], layer['request_id'], layer['positions'],
                 layer['layer'], receipt['input'], receipt['index']) ==
                (transcript['model'], transcript['suite'], transcript['request'], transcript['positions'], i, root, i), name)
        root = receipt['output']
        yield name, 'layer', run / f'inputs/{name}.v1.stdin', layer['digest']
    require(root == transcript['a36_root'], 'final layer boundary')
    for i, child in enumerate(heads):
        receipt = child['receipt']
        name = child['name']
        require((receipt['input'], receipt['shard'], receipt['shard_log2'], receipt['positions'], receipt['predictions']) ==
                (root, i, shard_log2, transcript['positions'], transcript['predictions']), name)
        yield name, 'head', run / f'shards/{name}.stdin', child['digest']


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def package(destination, run, jobs, results, setup, release):
    require(not destination.exists(), 'package destination already exists')
    require(json.loads(setup.read_text()) == release['setup_vk'], 'setup pin differs from release')
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        work = Path(tmp) / 'work'
        for folder in ('elf', 'inputs', 'receipts', 'pins'):
            (work / folder).mkdir(parents=True)
        shutil.copy2(setup, work / 'setup.json')
        shutil.copy2(Path(__file__).with_name('prove-child.sh'), work / 'prove-child.sh')
        for (name, role, stdin, digest), result in zip(jobs, results, strict=True):
            files = [f'elf/{role}', f'inputs/{name}.stdin', f'receipts/{name}', 'setup.json']
            elf, data, receipt, key = (work / f for f in files)
            if not elf.exists():
                shutil.copy2(run / 'elf' / role, elf)
            shutil.copyfile(stdin, data)
            receipt.write_text(digest + '00' * 224 + '\n')
            hashes = list(map(file_hash, (elf, data, receipt, key)))
            require(hashes[:2] == [result['identity']['elf'], result['identity']['input']],
                    name + ' changed after replay')
            (work / f'pins/{name}.sha256').write_text(''.join(
                f'{h}  {path}\n' for h, path in zip(hashes, files, strict=True)))
        shutil.copy2(run / 'request.json', work / 'request.json')
        shutil.copy2(run / 'shards/expected.json', work / 'expected.json')
        shutil.copy2(run / 'inputs/bundle.json', work / 'layers.json')
        (work / 'release.json').write_text(json.dumps(release))
        (work / 'roster').write_text(''.join(name + '\n' for name, *_ in jobs))
        work.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'release', 'emulator'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--client', required=True, type=Path, help='pci-client with check-prepared')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--package', type=Path, help='export successfully replayed inputs for prove-child.sh')
    parser.add_argument('--setup', type=Path, help='trusted setup JSON for export')
    args = parser.parse_args()
    if args.package and not args.setup:
        parser.error('--package requires --setup')
    run = args.run.resolve()
    release = json.loads(args.release.read_text())
    if args.package:
        require(not args.package.exists(), 'package destination already exists')
        require(json.loads(args.setup.read_text()) == release['setup_vk'], 'setup pin differs from release')
        require(all(release['programs'].get(role) for role in ('embed', 'layer', 'head')),
                'export requires approved program keys for every role')
    jobs = list(stages(run, release))
    output = run / 'preflight'
    output.mkdir(exist_ok=True)
    # Run the same request, weights, windows and argmax checks as the final client.
    digests = output / 'expected-digests'
    digests.write_text(''.join(digest + '\n' for *_, digest in jobs))
    bundle = output / 'receipt-bundle'
    client = args.client.resolve()
    subprocess.run([client, 'assemble', '--expected', run / 'shards/expected.json',
                    '--layers', run / 'inputs/bundle.json', '--proofs', output / 'no-proofs',
                    '--out', bundle], check=True, stdout=sys.stderr)
    subprocess.run([client, 'check-prepared', '--release', args.release, '--request', run / 'request.json',
                    '--bundle', bundle, '--digests', digests], check=True, stdout=sys.stderr)
    elfs = {role: file_hash(run / 'elf' / role) for role in ('embed', 'layer', 'head')}
    emulator = args.emulator.resolve()
    em_hash = file_hash(emulator)

    def replay(job):
        name, role, stdin, digest = job
        identity = dict(input=file_hash(stdin), elf=elfs[role], emulator=em_hash, receipt=digest)
        record = output / f'{name}.json'
        log = output / f'{name}.log'
        receipt = output / f'{name}.out'
        expected = bytes.fromhex(digest) + bytes(224)
        require(len(expected) == 256, name + ' invalid digest')
        if record.exists() and receipt.is_file() and log.is_file():
            result = json.loads(record.read_text())
            if (result.get('identity') == identity and receipt.read_bytes() == expected
                    and result.get('log_sha256') == file_hash(log)):
                return result
        receipt.unlink(missing_ok=True)
        with log.open('w') as stream:
            subprocess.run([emulator, '-e', run / 'elf' / role, '-i', stdin, '-o', receipt,
                            '--max-input-mem', str(stdin.stat().st_size + 8),
                            '-n', str((1 << 63) - 1), '-X'], stdout=stream, stderr=subprocess.STDOUT,
                           check=True)
        require(receipt.read_bytes() == expected, name + ' receipt')
        report = log.read_text()
        result = {'identity': identity, 'name': name, 'log_sha256': file_hash(log)}
        for field, label in [('steps', 'STEPS'), ('cost', 'TOTAL')]:
            result[field] = int(re.search(r'^' + label + r'\s+([\d,]+)', report, re.M)[1].replace(',', ''))
        tmp = record.with_suffix('.tmp')
        tmp.write_text(json.dumps(result))
        tmp.replace(record)
        return result

    # Exercise every ELF, including the partial vocabulary shard, at startup.
    by_name = {}
    order = [jobs[0], jobs[-1], *jobs[1:-1]]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(replay, job) for job in order]
        try:
            for future in as_completed(futures):
                result = future.result()
                by_name[result['name']] = result
                print(f"{len(by_name)}/{len(jobs)} {result['name']} passed", file=sys.stderr, flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    results = [by_name[name] for name, *_ in jobs]
    if args.package:
        package(args.package, run, jobs, results, args.setup, release)
    print(json.dumps({'children': len(results), **{
        key: sum(r[key] for r in results) for key in ('steps', 'cost')}}, indent=2))


if __name__ == '__main__':
    main()
