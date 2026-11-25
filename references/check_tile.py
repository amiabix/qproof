"""Check the plan's 64-row tile against the frozen native output."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import m0_certificate_reference as cert


def measure(golden):
    manifest = json.loads((golden / "manifest.json").read_text())
    proj = manifest["projections"]["L0/k"]
    checkpoint = golden / manifest["checkpoint_root"]
    file = checkpoint / manifest["tensors"][proj["tensor"]]["path"]
    weights = np.memmap(file, dtype="<u2", mode="r", offset=proj["byte_offset"],
                        shape=(64, proj["cols"]))
    inputs = cert.bf16(np.load(golden / proj["x"]))
    expected = np.load(golden / proj["out"]).T.view("<u4")
    enc = cert.encode(weights, inputs)
    sums = cert.host_sums(enc)
    certified = 0
    for row, position in np.ndindex(sums[0].shape):
        scales = (enc["sig"][row], enc["bet"][row], enc["tau"][position], enc["chi"][position])
        accepted, value, scale = cert.certify(
            *(int(matrix[row, position]) for matrix in sums),
            *map(int, scales), enc["K"], enc["M"])
        if accepted:
            bits = np.float32(math.ldexp(value, -scale)).view("<u4")
            if bits != expected[row, position]:
                raise ValueError(f"incorrect certification at row {row}, position {position}")
            certified += 1
    total = sums[0].size
    return dict(outputs=total, certified=certified, exact_path=total-certified, wrong=0)


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1]))))
