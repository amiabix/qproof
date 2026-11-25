"""Execute the frozen native reference and sampled independent parity checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m0"))
import run as m0


def records(run):
    roster = {r["name"]: r for r in m0.pinned_json(run / "roster.json", m0.ROSTER)}
    for i in m0.LAYERS:
        name = f"layer{i:02}"
        yield name, "layer", run / "inputs" / f"{name}.v1.stdin", roster[name]["digest"]
    for i in range((m0.VOCAB + m0.SHARD - 1) // m0.SHARD):
        name = m0.head_name(i)
        yield name, "head", run / roster[name]["source"], roster[name]["digest"]


def capture(run, binary, out):
    out.mkdir()
    def one(row):
        name, kind, path, digest = row
        subprocess.run([binary, kind, path, out / name, digest], check=True)
        print(name, flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, records(run)))
    (out / "complete").touch()


if __name__ == "__main__":
    capture(*map(Path, sys.argv[1:]))
