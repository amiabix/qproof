"""Inventory the BF16 checkpoint used by the Qwen3-4B harness. Tokens are the API."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate key: {name}")
        result[name] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def members():
    tied = "model.embed_tokens.weight"
    shapes = {tied: [151936, 2560], "model.norm.weight": [2560]}
    stages = [{"kind": "embedding", "tensors": [tied]}]
    for i in range(36):
        group = {
            "input_layernorm": [2560], "post_attention_layernorm": [2560],
            "self_attn.q_norm": [128], "self_attn.k_norm": [128],
            "self_attn.q_proj": [4096, 2560], "self_attn.k_proj": [1024, 2560],
            "self_attn.v_proj": [1024, 2560], "self_attn.o_proj": [2560, 4096],
            "mlp.gate_proj": [9728, 2560], "mlp.up_proj": [9728, 2560],
            "mlp.down_proj": [2560, 9728],
        }
        tensors = {f"model.layers.{i}.{k}.weight": v for k, v in group.items()}
        shapes.update(tensors)
        stages.append({"kind": "layer", "index": i, "tensors": sorted(tensors)})
    stages.append({"kind": "head", "tensors": ["model.norm.weight", tied]})
    return shapes, stages


def read_shard(path, expected):
    whole = hashlib.sha256()
    with path.open("rb") as f:
        prefix = f.read(8)
        size, = struct.unpack("<Q", prefix)
        if size > 16 * 1024 * 1024:
            raise ValueError("oversize shard header")
        header = f.read(size)
        records = parse(header)
        records.pop("__metadata__", None)
        if records.keys() != expected.keys():
            raise ValueError(f"shard membership mismatch: {path.name}")
        whole.update(prefix + header)
        cursor = 0
        tensors = {}
        for key, record in sorted(records.items(), key=lambda kv: kv[1]["data_offsets"]):
            shape = expected[key]
            start, end = record["data_offsets"]
            if (record["dtype"] != "BF16" or record["shape"] != shape
                    or start != cursor or end - start != math.prod(shape) * 2):
                raise ValueError(f"invalid tensor: {key}")
            tensor = hashlib.sha256()
            remaining = end - start
            while remaining:
                chunk = f.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f"truncated tensor: {key}")
                tensor.update(chunk)
                whole.update(chunk)
                remaining -= len(chunk)
            tensors[key] = {"dtype": "BF16", "shape": shape, "sha256": tensor.hexdigest()}
            cursor = end
        if f.read(1):
            raise ValueError("trailing tensor data")
    return whole.hexdigest(), tensors


def check_release(files, release):
    if files != release["files"]:
        raise ValueError("checkpoint differs from the pinned publisher files")


def enroll(directory):
    shapes, stages = members()
    config = (directory / "config.json").read_bytes()
    cfg = parse(config)
    checks = {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 2560,
              "intermediate_size": 9728, "num_attention_heads": 32,
              "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
              "tie_word_embeddings": True, "hidden_act": "silu",
              "use_sliding_window": False, "attention_bias": False,
              "rope_theta": 1000000, "rms_norm_eps": 1e-06, "rope_scaling": None}
    if any(cfg.get(k) != v for k, v in checks.items()):
        raise ValueError("unsupported graph")
    index = (directory / "model.safetensors.index.json").read_bytes()
    mapping = parse(index)["weight_map"]
    if mapping.keys() != shapes.keys():
        raise ValueError("incorrect tensor roster")
    files = {"config.json": digest(config), "model.safetensors.index.json": digest(index)}
    tensors = {}
    for shard in sorted(set(mapping.values())):
        if not re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", shard):
            raise ValueError("invalid shard name")
        files[shard], entries = read_shard(directory / shard,
                                           {k: shapes[k] for k in mapping if mapping[k] == shard})
        tensors.update(entries)
    release = parse(Path(__file__).with_name("qwen3-4b-release.json").read_bytes())
    check_release(files, release)
    inventory = {"version": 1, "files": files, "tensors": tensors, "stages": stages}
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {"checkpoint_id": digest(b"PCI/checkpoint-inventory/v1\0" + payload), **inventory}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    print(json.dumps(enroll(parser.parse_args().checkpoint), indent=2, sort_keys=True))
