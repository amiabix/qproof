"""Independently derive stage weight roots and ModelID from an enrollment
inventory and compare with a bundle's reported ModelID. Usage: INVENTORY BUNDLE.json"""
import hashlib
import json
import struct
import sys

LAYER = ["input_layernorm", "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
         "post_attention_layernorm", "self_attn.k_norm", "self_attn.k_proj",
         "self_attn.o_proj", "self_attn.q_norm", "self_attn.q_proj", "self_attn.v_proj"]


def h(domain, body):
    return hashlib.sha256(b"PCI/v1/" + domain + b"\0" + body).digest()


def profile(t, r):
    # n_out carries the prediction count R; ModelID uses the template (0, 0).
    return (struct.pack("<8I", 1, 2560, 9728, 32, 8, 128, 151936, t)
            + bytes.fromhex("00247449bd378635") + struct.pack("<II", 32, r) + bytes(16))


def stages():
    tied = "model.embed_tokens.weight"
    yield 0, 0, [tied]
    for i in range(36):
        yield 1, i, [f"model.layers.{i}.{t}.weight" for t in LAYER]
    yield 2, 0, ["model.norm.weight", tied]


def model_id(inv):
    roots = []
    for kind, index, names in stages():
        digests = [bytes.fromhex(inv["tensors"][n]["sha256"]) for n in names]
        roots.append(h(b"stage-weights", bytes([kind]) + struct.pack("<II", index, len(digests))
                       + b"".join(digests)))
    return h(b"model/qwen3-4b/mac-arm64-bf16-sdpa-v1", bytes.fromhex(inv["checkpoint_id"])
             + profile(0, 0) + struct.pack("<I", len(roots)) + b"".join(roots)), roots


if __name__ == "__main__":
    inv = json.load(open(sys.argv[1]))
    assert struct.unpack("<f", bytes.fromhex("00247449"))[0] == 1e6
    mid, _ = model_id(inv)
    bundle = json.load(open(sys.argv[2]))
    print("python model_id", mid.hex())
    print("bundle model_id", bundle["info"]["model_id"])
    weights = [c["weights"] for c in bundle["children"]]
    assert weights == [r.hex() for r in model_id(inv)[1]], "stage roots differ"
    assert mid.hex() == bundle["info"]["model_id"], "ModelID differs"
    r = len(bundle["reply"])
    for c in bundle["children"][1:]:
        t = c["positions"]
        assert (c["first_prediction"], c["predictions"]) == (t - r, r), "prediction window differs"
    print("stage roots, ModelID and prediction windows agree")
