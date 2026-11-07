"""Metadata-only tests of the audited discrete view; no synthetic GPU inputs."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import file_sha, read_json, write_json
from record_projection import comparable_numeric_fingerprint, complete_reference_view, project_records, projected_manifest
from bench import strict_manifest_checks


def metadata_record():
    # Small structural descriptors only, never consumed by a GPU workload.
    steps = []
    for index in (0, 1):
        steps.append({"step": index, "math_mode": {"tf32": False},
                      "invariants": {"gs_per_pixel": [{"shape": [20, 900, 1600], "dtype": "torch.int64", "sha256": f"indices{index}"}],
                                     "render_calls": index + 1, "flow_visualization_writes": 0},
                      "tensors": {f"out/module.taps.radii#{index}": {"shape": [100000], "dtype": "int32", "finite": True,
                                                                   "norm": None, "sha256": f"radii{index}", "kind": "out"},
                                  "grad/adam#0/module._xyz": {"shape": [100000, 3], "dtype": "float32", "finite": True,
                                                             "norm": 1.0, "sha256": "gradient", "kind": "grad"}}})
    return {"case": {"name": "metadata_only_case"}, "rank": 0, "adapter": "original-producer", "steps": steps}


class ProjectionTests(unittest.TestCase):
    def test_projection_changes_only_postupdate_discrete_view(self):
        raw = metadata_record()
        before = copy.deepcopy(raw)
        projected = projected_manifest(raw)
        self.assertEqual(raw, before)
        self.assertEqual(projected["steps"][0], raw["steps"][0])
        self.assertEqual(projected["adapter"], "original-producer")
        entry = projected["steps"][1]
        self.assertNotIn("sha256", entry["invariants"]["gs_per_pixel"][0])
        self.assertNotIn("out/module.taps.radii#1", entry["tensors"])
        self.assertEqual(entry["tensors"]["grad/adam#0/module._xyz"], raw["steps"][1]["tensors"]["grad/adam#0/module._xyz"])
        self.assertEqual(entry["diagnostics"]["postupdate_discrete"]["radii"]["out/module.taps.radii#1"],
                         raw["steps"][1]["tensors"]["out/module.taps.radii#1"])
        self.assertEqual(entry["invariants"]["postupdate_discrete_shapes"]["out/module.taps.radii#1"]["finite"], True)

    def compare(self, reference, candidate):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "ref/case.r0.json", reference)
            write_json(root / "candidate/case.r0.json", candidate)
            return strict_manifest_checks(root / "ref", root / "candidate")

    def test_initial_gs_and_radii_hash_mutations_are_rejected(self):
        reference = projected_manifest(metadata_record())
        candidate = copy.deepcopy(reference)
        candidate["steps"][0]["invariants"]["gs_per_pixel"][0]["sha256"] = "wrong"
        self.assertFalse(self.compare(reference, candidate)["ok"])
        candidate = copy.deepcopy(reference)
        candidate["steps"][0]["tensors"]["out/module.taps.radii#0"]["sha256"] = "wrong"
        self.assertFalse(self.compare(reference, candidate)["ok"])

    def test_postupdate_discrete_hash_drift_is_diagnostic(self):
        raw, changed = metadata_record(), metadata_record()
        changed["steps"][1]["invariants"]["gs_per_pixel"][0]["sha256"] = "legitimate-baseline-drift"
        changed["steps"][1]["tensors"]["out/module.taps.radii#1"]["sha256"] = "legitimate-baseline-drift"
        self.assertTrue(self.compare(projected_manifest(raw), projected_manifest(changed))["ok"])

    def test_every_postupdate_discrete_shape_dtype_and_finite_check_stays_exact(self):
        reference = projected_manifest(metadata_record())
        for target, field, value in (("gs", "shape", [20, 900, 1599]), ("gs", "dtype", "float64"),
                                     ("radii", "shape", [99999]), ("radii", "dtype", "float64"), ("radii", "finite", False)):
            candidate = copy.deepcopy(reference)
            desc = candidate["steps"][1]["invariants"]["gs_per_pixel"][0] if target == "gs" else candidate["steps"][1]["invariants"]["postupdate_discrete_shapes"]["out/module.taps.radii#1"]
            desc[field] = value
            self.assertFalse(self.compare(reference, candidate)["ok"], (target, field))

    def test_raw_pt_bytes_and_manifests_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            write_json(raw / "case.r0.json", metadata_record())
            (raw / "case.r0.s0.pt").write_bytes(b"metadata-test marker; not a tensor workload")
            before = {p.name: file_sha(p) for p in raw.iterdir()}
            project_records(raw, root / "view")
            self.assertEqual(before, {p.name: file_sha(p) for p in raw.iterdir()})
            self.assertTrue((root / "view/case.r0.s0.pt").is_symlink())
            self.assertEqual(file_sha(root / "view/case.r0.s0.pt"), before["case.r0.s0.pt"])
            with self.assertRaises(ValueError):
                projected_manifest(read_json(root / "view/case.r0.json"))

    def test_fingerprint_allows_only_declared_evaluator_change(self):
        recorded = {"judge_sha256": "old-producer-evaluator", "runtime": {"points": 100000}}
        actual = {"judge_sha256": "new-evaluator", "runtime": {"points": 100000}}
        self.assertTrue(comparable_numeric_fingerprint(actual, recorded, "new-evaluator"))
        self.assertEqual(recorded["judge_sha256"], "old-producer-evaluator")
        actual["runtime"]["points"] = 99999
        self.assertFalse(comparable_numeric_fingerprint(actual, recorded, "new-evaluator"))

    def test_complete_view_reference_padding_is_explicit_and_not_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, cases in (("reference", ("caseA", "caseB")), ("single", ("caseA",))):
                for case in cases:
                    record = projected_manifest(metadata_record())
                    record["case"]["name"] = case
                    write_json(root / folder / f"{case}.r0.json", record)
                    for step in (0, 1):
                        (root / folder / f"{case}.r0.s{step}.pt").write_bytes(b"metadata-only marker")
            provenance = complete_reference_view(root / "reference", root / "single", root / "complete")
            self.assertTrue(provenance["cases"]["caseA.r0.json"]["independent"])
            self.assertFalse(provenance["cases"]["caseB.r0.json"]["independent"])
            self.assertEqual(file_sha(root / "reference/caseB.r0.json"), file_sha(root / "complete/caseB.r0.json"))
            self.assertTrue((root / "complete/caseB.r0.s0.pt").is_symlink())


if __name__ == "__main__":
    unittest.main()
