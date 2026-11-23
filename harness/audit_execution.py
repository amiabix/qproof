"""Audit archived Qwen3-4B 32+8 profiler reports, output digests and greedy IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

STAGES = ["embed", *(f"layer{i:02}" for i in range(36)), "head"]
VOCAB = 151936


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(run, log, feedback):
    text = log.read_text()
    matches = re.findall(r"^(\w+) input .*?output matches native: yes$", text, re.M)
    require(matches == STAGES, "incomplete or mismatched segment roster")
    stages = []
    for name in STAGES:
        data = (run / (name + ".output")).read_bytes()
        require(len(data) == 256 and not any(data[32:]), f"{name}: public output layout")
        report = (run / (name + ".txt")).read_text()
        require("cost version 2" in report, f"{name}: profiler version")
        metrics = []
        for field in ("STEPS", "COST"):
            m = re.search(r"║\s+" + field + r"\s+([\d,]+)\s+║", report)
            require(m is not None, f"{name}: missing {field}")
            metrics.append(int(m[1].replace(",", "")))
        stages.append(dict(stage=name, steps=metrics[0], cost=metrics[1], output=data[:32].hex()))
    raw = (run / "logits.f32").read_bytes()
    count = len(feedback) + 1
    require(len(raw) == 4 * count * VOCAB, "head shape")
    require(hashlib.sha256(raw).hexdigest() == stages[-1]["output"], "head output digest")
    predicted = []
    for i in range(count):
        row = struct.unpack_from(f"<{VOCAB}f", raw, i * 4 * VOCAB)
        predicted.append(max(range(VOCAB), key=row.__getitem__))
    require(predicted[:-1] == feedback and str(predicted) in text, "feedback/prediction mismatch")
    return dict(steps=sum(s["steps"] for s in stages), cost=sum(s["cost"] for s in stages),
                predicted=predicted, stages=stages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qwen_dir", type=Path)
    parser.add_argument("scale_runs", type=Path)
    args = parser.parse_args()
    q, s = args.qwen_dir, args.scale_runs
    prompt = [int(i) for i in (q / "prompt32.txt").read_text().strip().split(",")]
    feedback = [13, 3776, 1899, 11, 1340, 1730, 264]
    require(len(prompt) == 32, "fixture prompt length")
    arms = {
        "original_floatdot": audit(q / "runs/big", q / "runs/big.log", feedback),
        "quantized_dot_f8": audit(s / "dotf8-full39", s / "dotf8-full39.log", feedback),
    }
    require(len({a["stages"][0]["output"] for a in arms.values()}) == 1, "embedding mismatch")
    print(json.dumps(dict(consumed_ids=prompt + feedback, prompt_count=len(prompt),
                          predictions=8, arms=arms), indent=2))
