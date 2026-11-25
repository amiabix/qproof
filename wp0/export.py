"""Package a completed capture. Paths in the manifest resolve from its declared roots."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np
from capture import m0, records

INVENTORY = "8c6f94b9f4d13329b6fdb06cd9ead465dc67451292662b70e127547c25c05579"
T, D, R = 39, 2560, 12


def check(ok, label):
    if not ok:
        raise ValueError(label)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_at(path, offset, size):
    with path.open("rb") as f:
        f.seek(offset)
        result = f.read(size)
    check(len(result) == size, f"truncated {path}")
    return result


def export(run, checkpoint, raw, out):
    check((raw / "complete").exists(), "incomplete native capture")
    inventory = m0.pinned_json(run / "inventory.json", INVENTORY)
    transcript = m0.pinned_json(run / "transcript.json", m0.TRANSCRIPT)
    roster = {r["name"]: r for r in m0.pinned_json(run / "roster.json", m0.ROSTER)}
    index = json.loads((checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    headers = {}
    for path, sha in inventory["files"].items():
        check(file_hash(checkpoint / path) == sha, path)
        if path.endswith(".safetensors"):
            with (checkpoint / path).open("rb") as f:
                size, = struct.unpack("<Q", f.read(8))
                headers[path] = size + 8, json.loads(f.read(size))

    out.mkdir()
    manifest = dict(format="wp0-golden/v1", positions=T, first_prediction=T-R, predictions=R,
                    checkpoint_root=os.path.relpath(checkpoint, out), checkpoint_id=inventory["checkpoint_id"],
                    checkpoint_files=inventory["files"], roster_sha256=m0.ROSTER,
                    inventory_sha256=INVENTORY, transcript_sha256=m0.TRANSCRIPT,
                    arrays={}, tensors={}, projections={}, stages={})

    def save(key, data):
        path = out / key
        path.parent.mkdir(exist_ok=True, parents=True)
        np.save(path, data, allow_pickle=False)
        manifest["arrays"][key] = dict(shape=list(data.shape), dtype=data.dtype.str, sha256=file_hash(path))
        return key

    def tensor(name):
        if name not in manifest["tensors"]:
            info = inventory["tensors"][name]
            path = index[name]
            base, header = headers[path]
            entry = header[name]
            start, end = entry["data_offsets"]
            check(entry["shape"] == info["shape"] and entry["dtype"] == info["dtype"] == "BF16"
                  and end-start == 2*math.prod(info["shape"]), name)
            manifest["tensors"][name] = dict(**info, path=path, byte_offset=base+start, nbytes=end-start)
        return manifest["tensors"][name]

    # Reconstruct the pinned child digest from each pair of saved boundaries.
    boundary = (run / "wide/a0.f32").read_bytes()
    def save_boundary(i, data):
        return save(f"boundaries/A{i}.npy", np.frombuffer(data, "<f4").reshape(T, D))
    save_boundary(0, boundary)
    for i in range(36):
        name = f"layer{i:02}"
        meta = json.loads((run / "inputs" / f"{name}.json").read_bytes())
        model, suite, request = (bytes.fromhex(meta[k+"_id"]) for k in ("model","suite","request"))
        prefix = b"PCI/v1/activation\0" + model + request
        output = (run / "inputs" / f"{name}.output.f32").read_bytes()
        roots = [hashlib.sha256(prefix + struct.pack("<III", j,T,D) + a).digest()
                 for j,a in [(i,boundary),(i+1,output)]]
        weights = bytes.fromhex(meta["receipt"]["weights"])
        child = digest(b"PCI/v1/child\0" + model + suite + request + struct.pack("<B4I",1,i,T,T-R,R)
                       + b"".join(roots) + weights)
        check(child == roster[name]["digest"], f"{name} boundary receipt")
        if i in m0.LAYERS:
            record = run / "inputs" / f"{name}.v1.stdin"
            check(boundary == read_at(record, 8+176, T*D*4), name+" input")
            check(output == (raw / (name+".boundary")).read_bytes(), name+" output")
        save_boundary(i+1, output)
        boundary = output
    manifest.update(model_id=model.hex(), suite_id=suite.hex(), request_id=request.hex())

    logits = np.empty((R,m0.VOCAB), dtype="<f4")
    norm = None
    for name, kind, record, receipt in records(run):
        stage = dict(receipt_sha256=receipt, record=str(record.relative_to(run)))
        manifest["stages"][name] = stage
        if kind == "layer":
            i = int(name[5:])
            stage.update(input=f"boundaries/A{i}.npy", output=f"boundaries/A{i+1}.npy")
            names = inventory["stages"][i+1]["tensors"]
            offsets = {}
            at = 8+176+T*D*4
            for key in names:
                spec = tensor(key)
                check(digest(read_at(record, at, spec["nbytes"])) == spec["sha256"], key)
                offsets[at] = key
                at += spec["nbytes"]
            stage["tensors"] = names
            matrices = zip(m0.MATRICES, m0.SHAPES, strict=True)
            positions = T
        else:
            i = int(name.split("-")[1])
            first = i*m0.SHARD
            width = min(m0.SHARD, m0.VOCAB-first)
            matrices = [("logits",(width,D))]
            positions = R
            check(boundary == read_at(record, 8+120, T*D*4), name+" input")
            stage.update(input="boundaries/A36.npy", norm_weight="model.norm.weight", first_token=first)
            norm_offset = 8+120+T*D*4
            check(digest(read_at(record,norm_offset,D*2)) == tensor("model.norm.weight")["sha256"], name+" norm")

        with (raw / name).open("rb") as f:
            for op,(rows,cols) in matrices:
                offset,t,r,k = struct.unpack("<4Q", f.read(32))
                check((t,r,k) == (positions,rows,cols), name+" capture shape")
                x = np.fromfile(f,"<f4",t*k).reshape(t,k)
                y = np.fromfile(f,"<f4",t*r).reshape(t,r)
                if kind == "layer":
                    key = offsets[offset+8]
                    check(key.endswith(f".{op}_proj.weight"), name+" matrix identity")
                    base = f"L{i}/{op}"
                    xfile = save(base+".x.npy",x)
                    start = 0
                else:
                    key = "model.embed_tokens.weight"
                    check(offset+8 == norm_offset + D*2, name+" head matrix offset")
                    start = first*D*2
                    base = "head/"+name
                    xfile = "head/final_norm.npy"
                    if norm is None:
                        norm = x.copy()
                        save(xfile,norm)
                    check(norm.tobytes() == x.tobytes(), name+" normalized activation")
                    logits[:,first:first+r] = y
                    ids = np.argmax(y,axis=1)
                    stage["winner_ids"] = (first+ids).tolist()
                    stage["winner_logit_bits"] = y[np.arange(R),ids].view("<u4").tolist()
                yfile = save(base+".out.npy",y)
                spec = tensor(key)
                size = r*k*2
                check(start+size <= spec["nbytes"], key+" range")
                byte_offset = spec["byte_offset"] + start
                weight_sha = digest(read_at(checkpoint/spec["path"],byte_offset,size))
                check(weight_sha == digest(read_at(record,offset+8,size)), name+" checkpoint bytes")
                manifest["projections"][base] = dict(x=xfile,out=yfile,tensor=key,stage=name,
                    byte_offset=byte_offset,nbytes=size,sha256=weight_sha,rows=r,cols=k)
            check(not f.read(1), name+" trailing capture")

    check(logits.tobytes() == (run / "inputs/logits.f32").read_bytes(), "frozen logits")
    winners = np.argmax(logits,axis=1).astype("<u4")
    check(winners.tolist() == transcript["reply_ids"], "transcript winners")
    manifest["head"] = dict(final_norm="head/final_norm.npy",
        winners=save("head/winners.npy",winners),
        winner_logits=save("head/winner_logits.npy",logits[np.arange(R),winners]),
        logits_sha256=digest(logits.tobytes()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    export(*map(Path,sys.argv[1:]))
