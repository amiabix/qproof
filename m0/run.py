"""Measure certificate coverage on the frozen Qwen request. No proving or GPU rental."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

from certificate import assess, lower_boundary

LAYERS = (0, 19, 35)
MATRICES = ("q", "k", "v", "o", "gate", "up", "down")
SHAPES = ((4096, 2560), (1024, 2560), (1024, 2560), (2560, 4096),
          (9728, 2560), (9728, 2560), (2560, 9728))
VOCAB, SHARD = 151936, 2048
ROSTER = "56093b52f0771b2fbe628da0f6971c439522dfcd1f3f7a8a5c39739b8487eef2"
TRANSCRIPT = "a15bff17da14153e65c30fb885cbc15c247bdef663fdef82a72b6553a2f741cc"


def pinned_json(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"frozen request file differs: {path}")
    return json.loads(raw)


def captures(path, record, shapes, positions):
    size = record.stat().st_size
    with path.open("rb") as f:
        for rows, cols in shapes:
            offset, t, r, k = struct.unpack("<4Q", f.read(32))
            if (t, r, k) != (positions, rows, cols) or 8 + offset + 2*r*k > size:
                raise ValueError("capture shape or weight offset")
            bits = np.fromfile(f, dtype="<u4", count=t*k).reshape(t, k)
            x = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & 0xffff0000).view("<f4")
            oracle = np.fromfile(f, dtype="<f4", count=t*r).reshape(t, r)
            weights = np.memmap(record, dtype="<u2", mode="r", offset=8+offset, shape=(r,k))
            yield weights, x, oracle
        if f.read(1):
            raise ValueError("unexpected extra matrix")


def head_name(i):
    return f"head-{i:05}"


def main(run, exporter, output):
    if output.exists():
        raise FileExistsError(output)
    roster = {row["name"]: row for row in pinned_json(run/"roster.json", ROSTER)}
    winners = pinned_json(run/"transcript.json", TRANSCRIPT)["reply_ids"]
    report = {"roster_sha256": ROSTER, "transcript_sha256": TRANSCRIPT, "matrices": {}}
    with tempfile.TemporaryDirectory() as directory:
        capture_dir = Path(directory)

        def export(name):
            head = name.startswith("head-")
            record = run/roster[name]["source"] if head else run/"inputs"/f"{name}.v1.stdin"
            dest = capture_dir/name
            if not dest.exists():
                subprocess.run([exporter, "head" if head else "layer", record, dest,
                                roster[name]["digest"]], check=True)
            return dest, record

        def head_matrix(i):
            files = export(head_name(i))
            return next(captures(*files, [(min(SHARD, VOCAB-i*SHARD),2560)], len(winners)))

        # The transcript proposes a winner. Its pinned score is witnessed here;
        # the selected proof path must establish that value in the real guest.
        scores = [None]*len(winners)
        paths = [None]*len(winners)
        candidates = [0]*len(winners)
        eliminated = [0]*len(winners)
        for shard in sorted({v//SHARD for v in winners}):
            _, _, oracle = head_matrix(shard)
            for t, v in enumerate(winners):
                if v//SHARD == shard:
                    scores[t] = float(oracle[t, v%SHARD])
        boundaries = [lower_boundary(score) for score in scores]

        def measure(name, weights, x, oracle, first_id=None):
            counts, maxima = Counter(), {}
            rows, k = weights.shape
            saved = 0
            for start in range(0, rows, 64):
                w = (weights[start:start+64].astype("<u4") << 16).view("<f4")
                for r,t,status,y,upper in assess(w, x, maxima):
                    row = start+r
                    ref = oracle[t,row]
                    counts[status] += 1
                    cert = status == "certified"
                    if cert and np.float32(y).tobytes() != ref.tobytes():
                        raise ValueError(f"false certification: {name} row {row} position {t}, {y} vs {ref}")
                    if first_id is None:
                        saved += int(cert)
                        continue
                    token = first_id + row
                    if ref > scores[t] or (ref == scores[t] and token < winners[t]):
                        raise ValueError(f"transcript winner disproved at {name}, token {token}, position {t}")
                    if token == winners[t]:
                        paths[t] = "certificate" if cert else "checker"
                        saved += int(cert)
                    elif upper is not None and upper < boundaries[t]:
                        if ref >= scores[t]:
                            raise ValueError(f"false winner elimination: {name} row {row} position {t}")
                        eliminated[t] += 1
                        saved += 1
                    else:
                        candidates[t] += 1
            total = rows*len(x)
            if sum(counts.values()) != total:
                raise ValueError("incomplete certificate results")
            result = dict(outputs=total, fmas=total*k, certified_but_wrong=0,
                          certified_share=counts["certified"]/total,
                          certified_fma_share=counts["certified"]/total,
                          fmas_avoiding_checker=saved*k, conditions=dict(counts),
                          max_integer_abs={key:int(value) for key,value in maxima.items()})
            report["matrices"][name] = result
            print(name, json.dumps(result), flush=True)

        for layer in LAYERS:
            name = f"layer{layer:02}"
            for op, args in zip(MATRICES, captures(*export(name), SHAPES, 39), strict=True):
                measure(f"{name}.{op}", *args)
        for i in range((VOCAB+SHARD-1)//SHARD):
            measure(head_name(i), *head_matrix(i), first_id=i*SHARD)
        if None in paths or any(a+b+1 != VOCAB for a,b in zip(candidates,eliminated)):
            raise ValueError("head coverage")
    report["head"] = [dict(token=v, winner_path=p, eliminated=e, other_candidates_to_check=c,
                           eliminated_share_among_losers=e/(VOCAB-1))
                      for v,p,e,c in zip(winners,paths,eliminated,candidates)]
    layer_matrices = [m for name,m in report["matrices"].items() if name.startswith("layer")]
    report["layer_fma_share"] = sum(m["fmas_avoiding_checker"] for m in layer_matrices) / sum(
        m["fmas"] for m in layer_matrices)
    report["head_losers_eliminated"] = sum(eliminated)
    report["head_other_candidates_to_check"] = sum(candidates)
    report["head_loser_elimination_share"] = sum(eliminated) / ((VOCAB-1)*len(winners))
    total = sum(m["fmas"] for m in report["matrices"].values())
    saved = sum(m["fmas_avoiding_checker"] for m in report["matrices"].values())
    share = saved/total
    report.update(total_fmas=total, fmas_avoiding_checker=saved, combined_fma_share=share,
                  gate="checker only" if share < .4 else "checker critical path" if share <= .7
                  else "matched tile experiment")
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(f"{saved}/{total} FMAs avoiding the checker ({share:.2%}): {report['gate']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "exporter", "output"):
        parser.add_argument(arg,type=Path)
    args = parser.parse_args()
    main(args.run, args.exporter.resolve(), args.output)
