"""Replay all exported projection outputs directly from the NPY files and checkpoint."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def replay(golden, binary, report):
    manifest = json.loads((golden / "manifest.json").read_text())
    checkpoint = golden / manifest["checkpoint_root"]
    def one(item):
        name, p = item
        x, y = (golden / p[key] for key in ("x","out"))
        a, b = (np.load(path, mmap_mode="r", allow_pickle=False) for path in (x,y))
        t, k = a.shape
        r = p["rows"]
        if b.shape != (t,r) or k != p["cols"] or any(
                v.dtype.str != "<f4" or not v.flags.c_contiguous for v in (a,b)):
            raise ValueError(name)
        w = checkpoint / manifest["tensors"][p["tensor"]]["path"]
        subprocess.run([binary,x,str(a.offset),w,str(p["byte_offset"]),y,str(b.offset),
                        str(t),str(r),str(k)],check=True)
        print(name, t*r, flush=True)
        return name, t*r
    with ThreadPoolExecutor(max_workers=6) as pool:
        counts = dict(pool.map(one, manifest["projections"].items()))
    result = dict(manifest_sha256=hashlib.sha256((golden/"manifest.json").read_bytes()).hexdigest(),
                  kernel="pci_dotf8::Kernel<SoftDotF8>::project_group_bf16",
                  outputs_compared=sum(counts.values()), compared_by_projection=counts, bit_mismatches=0)
    report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    replay(*map(Path,sys.argv[1:]))
