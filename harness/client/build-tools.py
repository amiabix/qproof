"""Build the release client or the GPU aggregate host against a supplied backend.

The verifier must be the generated crate for the release's actual recursive setup.
Uses the project's locked Cargo workspace without editing its manifests.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tool', choices=['client', 'aggregate-host'])
    p.add_argument('backend', type=Path, help='directory containing source/{verifier,sdk} and proofman-common')
    p.add_argument('out', type=Path, help='Cargo target and release binaries directory')
    a = p.parse_args()
    client = Path(__file__).resolve().parent
    backend = a.backend.resolve()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = client if a.tool == 'client' else client / 'aggregate-host'
    package = 'pci-' + a.tool
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        root = Path(tmp)
        workspace = root / 'harness/client'
        workspace.mkdir(parents=True)
        for dep in ('dotf8', 'receipts', 'layer'):
            (workspace.parent / dep).symlink_to(client.parent / dep)
        (root / 'common').symlink_to(client.parent.parent / 'common')
        if a.tool == 'client':
            shutil.copyfile(client / 'Cargo.toml', workspace / 'Cargo.toml')
            for dep in ('stages', 'embed-guest', 'head-guest'):
                (workspace / dep).symlink_to(client / dep)
            member = workspace / 'cli'
            manifest = client / 'cli/Cargo.toml'
        else:
            (workspace / 'recursion').symlink_to(client / 'recursion')
            workspace = workspace / 'aggregate-host'
            member = workspace
            manifest = source / 'Cargo.toml'
        member.mkdir()
        for folder in ('src', 'tests'):
            path = manifest.parent / folder
            if path.exists():
                (member / folder).symlink_to(path)
        text = manifest.read_text()
        config = tomllib.loads(text)
        if a.tool == 'client':
            paths = [(config['dependencies']['zisk-verifier'], backend / 'source/verifier')]
        else:
            paths = [(config['dependencies']['zisk-sdk'], backend / 'source/sdk'),
                     (config['patch']['crates-io']['proofman-common'], backend / 'proofman-common')]
        for dependency, path in paths:
            if not (path / 'Cargo.toml').is_file():
                p.error(f'missing backend crate at {path}')
            text = text.replace(json.dumps(dependency['path']), json.dumps(str(path), ensure_ascii=False))
        (member / 'Cargo.toml').write_text(text)
        shutil.copyfile(source / 'Cargo.lock', workspace / 'Cargo.lock')
        subprocess.run(['cargo', 'build', '--locked', '--release', '-p', package],
                       cwd=workspace, env={**os.environ, 'CARGO_TARGET_DIR': str(out)}, check=True)


if __name__ == '__main__':
    main()
