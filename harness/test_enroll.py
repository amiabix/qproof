import json
import struct
import tempfile
import unittest
from pathlib import Path
from enroll import check_release, digest, members, read_shard


class CheckpointTest(unittest.TestCase):
    def test_release_substitution(self):
        release = {"files": {"weights": digest(b"original")}}
        check_release(release["files"], release)
        with self.assertRaises(ValueError):
            check_release({"weights": digest(b"altered")}, release)

    def test_graph(self):
        shapes, stages = members()
        self.assertEqual(len(shapes), 398)
        self.assertEqual([s["index"] for s in stages[1:-1]], list(range(36)))
        self.assertEqual(stages[0]["tensors"][0], stages[-1]["tensors"][-1])

    def test_shard(self):
        key = "weight"
        header = json.dumps({key: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        raw = struct.pack("<Q", len(header)) + header + b"abcd"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "shard"
            p.write_bytes(raw)
            whole, tensors = read_shard(p, {key: [2]})
            self.assertEqual(whole, digest(raw))
            self.assertEqual(tensors[key]["sha256"], digest(b"abcd"))
            for invalid in (raw[:-1], raw + b"x", raw.replace(b"BF16", b"FP16")):
                p.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    read_shard(p, {key: [2]})
