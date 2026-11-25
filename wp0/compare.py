"""Verify two complete golden exports against their manifest and each other."""
import json
from pathlib import Path
import sys

import numpy as np
from export import check, file_hash


def compare(first, second, report):
    manifest = json.loads((first / "manifest.json").read_bytes())
    arrays = manifest["arrays"]
    expected = {"manifest.json", *arrays}
    for root in (first, second):
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        check(actual == expected, f"{root} file set")
        check((first / "manifest.json").read_bytes() == (root / "manifest.json").read_bytes(),
              "manifest differs")
        for name, spec in arrays.items():
            array = np.load(root / name, mmap_mode="r", allow_pickle=False)
            check(array.dtype.str == spec["dtype"] and list(array.shape) == spec["shape"],
                  f"{root}/{name} layout")
            check(file_hash(root / name) == spec["sha256"], f"{root}/{name} checksum")

    result = dict(manifest_sha256=file_hash(first / "manifest.json"),
                  identical_files=len(expected), verified_arrays=len(arrays))
    report.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    compare(*map(Path, sys.argv[1:]))
