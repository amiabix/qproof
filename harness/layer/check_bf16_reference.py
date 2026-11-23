"""Compare a native Qwen bundle with the pinned CPU reference, including cached decode.

Example: python check_bf16_reference.py MODEL_DIR BUNDLE_DIR --prompt 2,36518,25
For a supplied multi-token reply pass the original prompt and the bundle containing
all predictions. Reference feedback always comes from the reference's own argmax.
"""
import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    if (platform.system(), platform.machine(), torch.version.git_version,
        transformers.__version__) != ("Darwin", "arm64",
        "cf30153c4c131c8164ee7798e5022d810682e2cb", "5.15.1"):
        raise SystemExit("This comparison requires the pinned Mac arm64 Torch and Transformers builds")
    torch.set_num_threads(6)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval()
    bundle = json.loads((args.bundle / "bundle.json").read_text())
    prompt = [int(v) for v in args.prompt.split(",")]
    ids = torch.tensor([prompt])
    expected_reply = bundle["reply"]
    positions = len(prompt) + len(expected_reply) - 1
    info = bundle["info"]
    prefix = (
        b"PCI/v1/activation\0"
        + bytes.fromhex(info["model_id"])
        + bytes.fromhex(info["request_id"])
    )
    dim = model.config.hidden_size
    layers = model.model.layers
    roots = [
        hashlib.sha256(prefix + struct.pack("<III", i, positions, dim))
        for i in range(len(layers) + 1)
    ]
    row_counts = [0] * len(roots)

    def capture(i):
        def hook(_module, inputs):
            values = inputs[0][0].detach().float().numpy()
            if values.ndim != 2 or values.shape[1] != dim:
                raise ValueError(f"unexpected activation shape at boundary {i}: {values.shape}")
            roots[i].update(values.astype("<f4", copy=False).tobytes())
            row_counts[i] += values.shape[0]
        return hook

    handles = [
        module.register_forward_pre_hook(capture(i))
        for i, module in enumerate([*layers, model.model.norm])
    ]
    actual = np.memmap(args.bundle / "logits.f32", dtype="<u4", mode="r")
    actual = actual.reshape(len(expected_reply), model.config.vocab_size)
    cache = None
    reply = []
    mismatches = 0
    with torch.inference_mode():
        for actual_bits in actual:
            out = model(ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            scores = out.logits[0, -1]
            reference_bits = scores.float().numpy().view("<u4")
            mismatches += int(np.count_nonzero(actual_bits != reference_bits))
            ids = scores.argmax().reshape(1, 1)
            reply.append(ids.item())
    for handle in handles:
        handle.remove()

    wrong = [
        i for i, root in enumerate(roots)
        if row_counts[i] != positions or root.hexdigest() != bundle["children"][i]["output"]
    ]
    print(json.dumps({
        "different_boundaries": wrong,
        "different_logits": mismatches,
        "total_logits": actual.size,
        "reference_ids": reply,
        "harness_ids": expected_reply,
    }))
    return int(bool(wrong or mismatches or reply != expected_reply))


if __name__ == "__main__":
    raise SystemExit(main())
