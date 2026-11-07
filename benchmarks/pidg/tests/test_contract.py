"""CPU-only contract tests. These metadata fixtures are never GPU workloads."""
import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import signal
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import (GPU_UUID, POLICY, RASTER, ROOT, VARIANTS, compare_timing, file_sha,
                    manifest, new_directory, run_checks, selections, window_checks)
from bench import schedule, worker_job


class ContractTests(unittest.TestCase):
    def test_data_directory_symlinks_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public, fixture = root / "public", root / "fixture"
            public.mkdir()
            fixture.mkdir()
            resource = public / "camera-input.bin"
            resource.write_bytes(b"first downloaded resource")
            (fixture / "train").symlink_to(public, target_is_directory=True)
            before = manifest(fixture)
            self.assertIn("train/camera-input.bin", before)
            resource.write_bytes(b"changed resource")
            self.assertNotEqual(before, manifest(fixture))

    def test_data_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "loop").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                manifest(root)

    def test_existing_artifacts_never_reused_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            new_directory(root)
            with self.assertRaises(FileExistsError):
                new_directory(root)
            self.assertTrue((root / ".pidg-bench-owned.json").is_file())

    def test_each_variant_chooses_only_requested_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            trees = [(Path(directory) / name).resolve() for name in ("baseline", "candidate")]
            for tree in trees:
                package = tree / RASTER / "diff_gaussian_rasterization"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("# metadata-only selection fixture")
                (package / "_C.metadata-test.so").write_bytes(b"not a loadable GPU library")
            for variant in VARIANTS:
                selected = selections(*trees, variant)
                flow_tree = trees[1] if variant in ("flow-only", "both") else trees[0]
                raster_tree = trees[1] if variant in ("raster-only", "both") else trees[0]
                self.assertEqual(selected["flow_file"], str(flow_tree / "utils/flow_utils.py"))
                self.assertEqual(selected["raster_root"], str(raster_tree / RASTER))
                self.assertEqual(selected["base_tree"], str(trees[0]))

    def test_original_generic_component_copied_unchanged(self):
        source = ROOT.parent / "ResearchBoost-Agent" / "numerics"
        for path in source.glob("*.py"):
            self.assertEqual(file_sha(path), file_sha(ROOT / "numerics" / path.name), path.name)

    def test_standard_and_explicit_short_plans(self):
        args = argparse.Namespace(smoke=False, short=False, warmup=None, windows=None, iterations=None)
        self.assertEqual(schedule(args, "raster-regular")["warmup"], 20)
        self.assertEqual(schedule(args, "raster-regular")["iterations"], 35)
        args.short = True
        self.assertEqual(schedule(args, "triple-flow"), {"warmup": 1, "windows": 3, "iterations": 3, "protocol": "SHORT_EXPLORATORY"})
        self.assertEqual(schedule(args, "raster-regular")["iterations"], 35)
        args.short, args.warmup = False, 1
        with self.assertRaises(ValueError):
            schedule(args, "raster-regular")

    def test_timeout_targets_only_new_worker_process_group(self):
        class FakeProcess:
            pid = 999999
            returncode = None

            def wait(self, timeout):
                if timeout == POLICY["worker_timeout_seconds"]:
                    raise subprocess.TimeoutExpired("metadata-only mock", timeout)
                self.returncode = -15
                return self.returncode

            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            tree = root / "baseline"
            package = tree / RASTER / "diff_gaussian_rasterization"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("# metadata-only fixture")
            (package / "_C.metadata-test.so").write_bytes(b"not loadable")
            args = argparse.Namespace(tree=None, hashencoder_build=None, smoke=False, short=False,
                                      warmup=None, windows=None, iterations=None, _fixture_sha="fixture", _judge_sha="judge")
            with patch("bench.subprocess.Popen", return_value=FakeProcess()) as launch, patch("bench.os.killpg") as kill:
                with self.assertRaises(RuntimeError):
                    worker_job(args, {"baseline": str(tree), "fixture": "unused"}, root / "worker",
                               "time", "dry_ice", "raster-regular")
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                kill.assert_called_once_with(999999, signal.SIGTERM)
            self.assertTrue((root / "worker/timeout.json").is_file())
            self.assertTrue((root / "worker/result.json").is_file())

    @staticmethod
    def arm(wall=10., events=None, clocks=1200):
        events = events or [wall] * 35
        windows = [{"wall_ms_per_iteration": wall, "noise": window_checks(events),
                    "power": {"sm_clock_mhz_median": clocks}} for _ in range(3)]
        return {"windows": windows, "timing": run_checks(windows),
                "fingerprint": {"comparable": {"gpu_uuid": GPU_UUID, "iterations": 35}}}

    def test_within_window_contamination_invalidates(self):
        self.assertFalse(window_checks([10.] * 28 + [20.] * 7)["valid"])
        self.assertFalse(window_checks([10.] * 17 + [12.] * 18)["valid"])
        before, candidate, after = self.arm(), self.arm(events=[10.] * 28 + [20.] * 7), self.arm()
        self.assertEqual(compare_timing(before, candidate, after)["status"], "INVALID")

    def test_symmetric_window_contamination_invalidates(self):
        before, candidate, after = self.arm(), self.arm(5.), self.arm()
        for arm in (before, after):
            arm["windows"][1]["wall_ms_per_iteration"] = 12.
            arm["timing"] = run_checks(arm["windows"])
        self.assertEqual(compare_timing(before, candidate, after)["status"], "INVALID")

    def test_fingerprint_and_clock_mismatch_rejected(self):
        before, candidate, after = self.arm(), self.arm(5.), self.arm()
        candidate["fingerprint"]["comparable"]["iterations"] = 3
        self.assertEqual(compare_timing(before, candidate, after)["status"], "INVALID")
        candidate = self.arm(5., clocks=1000)
        self.assertEqual(compare_timing(before, candidate, after)["status"], "INVALID")

    def test_resolved_observation_remains_exploratory(self):
        report = compare_timing(self.arm(), self.arm(5.), self.arm())
        self.assertEqual(report["status"], "EXPLORATORY")
        self.assertEqual(report["observed_speedup"], 2.)
        self.assertEqual(report["formal_benefit_verdict"], "UNAVAILABLE_SHARED_WORKLOAD")
        self.assertTrue(report["resolved_above_observed_noise"])

    def test_missing_active_clock_samples_are_unavailable_not_quality_failure(self):
        before, candidate, after = self.arm(), self.arm(5.), self.arm()
        candidate["windows"][0]["power"]["sm_clock_mhz_median"] = None
        report = compare_timing(before, candidate, after)
        self.assertEqual(report["status"], "EXPLORATORY")
        self.assertFalse(report["clock_comparability_available"])
        self.assertFalse(report["resolved_above_observed_noise"])


if __name__ == "__main__":
    unittest.main()
