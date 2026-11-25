"""tools/sandbox_scene.py: entity world rows, nearest-first selection and the packed vertex layout."""
import hashlib
import io
import json
import os
import sys
import struct
import subprocess
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import sandbox_scene
import world_names

class TestSandboxScene(unittest.TestCase):
    def test_draw_rows(self):
        result = sandbox_scene.draw_rows((0, 0, 0, 1), 1.0, 1.0, (10, 20, 30))
        self.assertAlmostEqual(result[0][0], 1.0, places=6)
        self.assertAlmostEqual(result[0][1], 0.0, places=6)
        self.assertAlmostEqual(result[0][2], 0.0, places=6)
        self.assertAlmostEqual(result[1][0], 0.0, places=6)
        self.assertAlmostEqual(result[1][1], 1.0, places=6)
        self.assertAlmostEqual(result[1][2], 0.0, places=6)
        self.assertAlmostEqual(result[2][0], 0.0, places=6)
        self.assertAlmostEqual(result[2][1], 0.0, places=6)
        self.assertAlmostEqual(result[2][2], 1.0, places=6)
        self.assertAlmostEqual(result[3][0], 10.0, places=6)
        self.assertAlmostEqual(result[3][1], 20.0, places=6)
        self.assertAlmostEqual(result[3][2], 30.0, places=6)

    def test_draw_rows_rotation(self):
        result = sandbox_scene.draw_rows((0, 0, -0.70710678, 0.70710678), 1, 1, (0, 0, 0))
        self.assertAlmostEqual(result[0][0], 0.0, places=6)
        self.assertAlmostEqual(result[0][1], 1.0, places=6)
        self.assertAlmostEqual(result[0][2], 0.0, places=6)
        self.assertAlmostEqual(result[1][0], -1.0, places=6)
        self.assertAlmostEqual(result[1][1], 0.0, places=6)
        self.assertAlmostEqual(result[1][2], 0.0, places=6)
        self.assertAlmostEqual(result[2][0], 0.0, places=6)
        self.assertAlmostEqual(result[2][1], 0.0, places=6)
        self.assertAlmostEqual(result[2][2], 1.0, places=6)

    def test_draw_rows_scale(self):
        result = sandbox_scene.draw_rows((0, 0, 0, 1), 2.0, 3.0, (0, 0, 0))
        self.assertAlmostEqual(result[0][0], 2.0, places=6)
        self.assertAlmostEqual(result[0][1], 0.0, places=6)
        self.assertAlmostEqual(result[0][2], 0.0, places=6)
        self.assertAlmostEqual(result[1][0], 0.0, places=6)
        self.assertAlmostEqual(result[1][1], 2.0, places=6)
        self.assertAlmostEqual(result[1][2], 0.0, places=6)
        self.assertAlmostEqual(result[2][0], 0.0, places=6)
        self.assertAlmostEqual(result[2][1], 0.0, places=6)
        self.assertAlmostEqual(result[2][2], 3.0, places=6)

    def test_select_rows(self):
        rows = [
            {"_pos": [1, 1, 0]},
            {"_pos": [5, 5, 0]},
            {"_pos": [20, 20, 0]},
            {"_pos": [3, 3, 0]}
        ]
        result = sandbox_scene.select_rows(rows, (0, 0), 10, 10)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]["_pos"][0], 1.0, places=6)
        self.assertAlmostEqual(result[0]["_pos"][1], 1.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][0], 3.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][1], 3.0, places=6)
        self.assertAlmostEqual(result[2]["_pos"][0], 5.0, places=6)
        self.assertAlmostEqual(result[2]["_pos"][1], 5.0, places=6)

        result = sandbox_scene.select_rows(rows, (0, 0), 10, 2)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["_pos"][0], 1.0, places=6)
        self.assertAlmostEqual(result[0]["_pos"][1], 1.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][0], 3.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][1], 3.0, places=6)

    def test_full_selection_memory(self):
        # Searching past missing models needs every nearby candidate, without
        # retaining distance/row tuples and duplicate full-size reference lists.
        rows = [{"_pos": (0.0, 0.0)}] * 100_000
        tracemalloc.start()
        try:
            picked = sandbox_scene.select_rows(rows, (0, 0), 1, len(rows))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(picked), len(rows))
        self.assertTrue(all(row is rows[0] for row in picked))
        # Allow eight pointers per row, including temporary sorting storage.
        # The original tuple-based selection needs more than twelve on x64.
        self.assertLess(peak, 8 * struct.calcsize("P") * len(rows))

    def test_mesh_bytes(self):
        prim = {
            "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "normals": [(0, 0, 1)] * 3,
            "uvs": [(0, 0), (1, 0), (0, 1)],
            "indices": [0, 1, 2]
        }
        b = sandbox_scene.mesh_bytes(prim)
        self.assertEqual(len(b), 116)
        self.assertEqual(tuple(struct.unpack_from('<II', b, 0)), (3, 3))
        self.assertEqual(tuple(struct.unpack_from('<8f', b, 40)), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(tuple(struct.unpack_from('<3I', b, 104)), (0, 1, 2))

    def test_mesh_bytes_no_normals(self):
        prim = {
            "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "normals": [],
            "indices": [0, 1, 2]
        }
        b = sandbox_scene.mesh_bytes(prim)
        self.assertEqual(tuple(struct.unpack_from('<8f', b, 8)), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(tuple(struct.unpack_from('<8f', b, 40)), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

class TestCliErrors(unittest.TestCase):
    """An unreadable input or an unwritable --out is one ERROR: line and exit 2, not a traceback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = os.path.join(self.tmp.name, "models")
        os.makedirs(self.models)
        self.entities = os.path.join(self.tmp.name, "e.jsonl")
        with open(self.entities, "w", encoding="utf-8") as f:
            f.write(json.dumps({"archetype": 123, "position": [0, 0, 0],
                                "rotation": [0, 0, 0, 1], "lod_level": 0}) + "\n")

    def _run(self, entities, out):
        err = io.StringIO()
        with redirect_stderr(err):
            rc = sandbox_scene.main(["--entities", entities, "--models", self.models,
                                     "--center", "0", "0", "--out", out])
        return rc, err.getvalue()

    def test_out_directory_is_one_error_line(self):
        out = os.path.join(self.tmp.name, "iamdir")
        os.makedirs(out)
        rc, err = self._run(self.entities, out)
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("ERROR:"))
        # the refusal is up front and says why, not a repr'd OSError with doubled backslashes
        self.assertIn("is a folder, not a scene file", err)
        self.assertNotIn("[Errno", err)

    def test_out_under_a_regular_file_is_one_error_line(self):
        reg = os.path.join(self.tmp.name, "reg")
        with open(reg, "w", encoding="utf-8") as f:
            f.write("x")
        rc, err = self._run(self.entities, os.path.join(reg, "o.bin"))
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("ERROR:"))

    def test_missing_entities_is_one_error_line(self):
        rc, err = self._run(os.path.join(self.tmp.name, "nope.jsonl"),
                            os.path.join(self.tmp.name, "o.bin"))
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("ERROR:"))

    def test_nonfinite_or_overflowing_center_is_usage_error(self):
        for x, y in (("nan", "0"), ("inf", "0"), ("0", "-inf"), ("1e40", "0"), ("0", "-1e40")):
            with self.subTest(center=(x, y)):
                out = os.path.join(self.tmp.name, f"bad_{x}_{y}.bin")
                err = io.StringIO()
                with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    sandbox_scene.main(["--entities", self.entities, "--models", self.models,
                                        "--center", x, y, "--out", out])
                self.assertEqual(cm.exception.code, 2)
                self.assertIn("--center", err.getvalue())
                self.assertFalse(os.path.exists(out))

    def test_writable_out_still_builds(self):
        out = os.path.join(self.tmp.name, "o.bin")
        err = io.StringIO()
        with redirect_stderr(err):
            rc = sandbox_scene.main(["--entities", self.entities, "--models", self.models,
                                     "--center", "0", "0", "--out", out])
        self.assertEqual(rc, 0)
        self.assertEqual(err.getvalue(), "")
        with open(out, "rb") as f:
            self.assertEqual(f.read(4), b"LSB1")


class TestAtomicSceneOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.script = os.path.join(self.root, "tools", "sandbox_scene.py")
        self.models = os.path.join(self.tmp.name, "models")
        os.mkdir(self.models)
        model_name = "synthetic_atomic_scene_prop"
        vertex_count = 60
        obj = "\n".join([
            *(f"v {i} 0 0" for i in range(vertex_count)),
            *(f"f {i} {i + 1} {i + 2}" for i in range(1, vertex_count - 1, 3)),
        ]) + "\n"
        with open(os.path.join(self.models, model_name + ".obj"), "w", encoding="utf-8") as f:
            f.write(obj)
        with open(os.path.join(self.models, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"Items": [{"Name": model_name, "File": model_name + ".obj", "MeshShaders": [0]}]}, f)
        self.entities = os.path.join(self.tmp.name, "entities.jsonl")
        with open(self.entities, "w", encoding="utf-8") as f:
            json.dump({"archetype": world_names.joaat(model_name), "position": [0, 0, 0],
                       "rotation": [0, 0, 0, 1], "lod_level": 0}, f)
            f.write("\n")

    def _run(self, out, limited=False):
        command = [sys.executable, "-I", self.script, "--entities", self.entities,
                   "--models", self.models, "--center", "0", "0", "--out", out]
        if limited:
            return subprocess.run(
                ["bash", "-c", 'trap "" XFSZ; ulimit -f 1; exec "$@"', "sandbox-scene", *command],
                cwd=self.root, capture_output=True, text=True)
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True)

    def test_failed_write_preserves_existing_output(self):
        out = os.path.join(self.tmp.name, "scene.bin")
        original = b"pre-existing scene contents"
        with open(out, "wb") as f:
            f.write(original)

        result = self._run(out, limited=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("File too large", result.stderr)
        with open(out, "rb") as f:
            self.assertEqual(f.read(), original)

    def test_failed_write_leaves_no_temporary_file(self):
        out = os.path.join(self.tmp.name, "scene.bin")
        with open(out, "wb") as f:
            f.write(b"existing")

        result = self._run(out, limited=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(set(os.listdir(self.tmp.name)), {"models", "entities.jsonl", "scene.bin"})

    def test_successful_write_preserves_scene_bytes(self):
        out = os.path.join(self.tmp.name, "scene.bin")
        with open(out, "wb") as f:
            f.write(b"old scene")
        os.chmod(out, 0o640)

        result = self._run(out)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        expected_sha256 = "c456bc9bd29f8f0d4dd36a983fd70c8fb25ebadbb9a4bd6c97672c2f11a09493"
        with open(out, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), expected_sha256)
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o640)
        self.assertEqual(set(os.listdir(self.tmp.name)), {"models", "entities.jsonl", "scene.bin"})


if __name__ == "__main__":
    unittest.main()
