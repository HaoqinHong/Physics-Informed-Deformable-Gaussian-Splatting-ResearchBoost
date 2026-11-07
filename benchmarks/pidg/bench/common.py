"""CPU-only contracts, manifests and timing decisions for the exploratory PIDG bench."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("dry_ice", "rubber_duck", "balls_reaction", "mechanics_cloth", "motion_kuro")
VARIANTS = ("baseline", "flow-only", "raster-only", "both")
WORKLOADS = ("raster-regular", "raster-flow", "triple-flow")
RASTER = "submodules/flow-based-diff-gaussian-rasterization"
GPU_UUID = json.loads((ROOT / "task/bench.json").read_text())["gpu_uuid"]
SCOPE = "exploratory_module_diagnostics_only; no full-training acceptance or convergence evidence"
POLICY = {
    "warmup": 20, "windows": 3, "iterations": 35,
    "window_range_limit": 0.10,
    "within_window_p90_p10_limit": 0.20,
    "within_window_half_drift_limit": 0.10,
    "reference_window_drift_limit": 0.10,
    "clock_relative_gap_limit": 0.05,
    "clock_active_utilization_percent": 5.0,
    "clock_min_active_samples_per_window": 2,
    "worker_timeout_seconds": 1800,
    "smoke_timeout_seconds": 600,
    "worker_terminate_grace_seconds": 10,
    "worker_kill_grace_seconds": 5,
    "small_gain_resolution_floor": 0.01,
    "baseline_records": 4, "numeric_steps": 2, "seed": 20261105,
    "sketch_dim": 8192,
    "k_fwd": 2.0, "k_bwd": 3.0,
    # Unchanged generic component's published diagnostic floors. The measured
    # baseline-self envelope can make a gate larger, never candidate evidence.
    "floor_fwd": 1e-6, "floor_bwd": 1e-5,
    "shared_workload": True,
    "full_training_acceptance": False,
    "fp64_arbitration_available": False,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / ".pidg-bench-owned.json", {"created_unix": time.time(), "scope": SCOPE})
    return path


def manifest(root, source_only=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    ignored = {".git", "__pycache__", "build", ".pytest_cache", ".pycache"}
    visited = set()
    # Derivative fixture directories intentionally link into the downloaded
    # public tree. Hash their CONTENT, including optional loader inputs. Source
    # manifests do not traverse directory links out of an upstream checkout.
    for parent, dirs, names in os.walk(root, followlinks=not source_only):
        real_parent = Path(parent).resolve()
        if real_parent in visited:
            raise ValueError(f"cyclic or aliased directory in manifest: {parent}")
        visited.add(real_parent)
        dirs[:] = sorted(d for d in dirs if d not in ignored and not d.endswith(".egg-info"))
        for name in sorted(names):
            path = Path(parent) / name
            if name.startswith(".pidg-bench-") or name.endswith((".pyc", ".pyo")):
                continue
            if source_only and path.suffix not in {".py", ".cu", ".cuh", ".h", ".cpp", ".c", ".so"}:
                continue
            result[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": file_sha(path)}
    if not result:
        raise ValueError(f"empty manifest: {root}")
    return result


def judge_manifest():
    result = {}
    for directory in ("bench", "numerics", "task"):
        for rel, info in manifest(ROOT / directory).items():
            if (directory != "task" and rel.endswith(".py")) or rel == "bench.json":
                result[f"{directory}/{rel}"] = info
    return result


def extension_path(tree):
    package = Path(tree).resolve() / RASTER / "diff_gaussian_rasterization"
    candidates = sorted(package.glob("_C*.so"))
    if not (package / "__init__.py").is_file() or len(candidates) != 1:
        raise ValueError(f"require exactly one tree-local _C*.so and package wrapper in {package}; found {candidates}")
    return candidates[0]


def selections(baseline, candidate, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    base = Path(baseline).resolve()
    cand = Path(candidate).resolve() if candidate else None
    if variant != "baseline" and cand is None:
        raise ValueError("a candidate tree is required")
    flow = cand if variant in ("flow-only", "both") else base
    raster = cand if variant in ("raster-only", "both") else base
    return {"base_tree": str(base), "flow_file": str(flow / "utils/flow_utils.py"),
            "raster_root": str(raster / RASTER), "extension": str(extension_path(raster)), "variant": variant}


def validate_fixture(fixture_root, public_root, scenes=SCENES):
    """The derivative must retain exactly the true first two train frames and their times."""
    result = {}
    for scene in scenes:
        source = Path(public_root) / scene
        fixture = Path(fixture_root) / scene
        original = read_json(source / "transforms_train.json")
        train = read_json(fixture / "transforms_train.json")
        test = read_json(fixture / "transforms_test.json")
        if train.get("frames") != original["frames"][:2] or len(train["frames"]) != 2:
            raise ValueError(f"{scene}: derivative frames/time/poses do not equal the original first two train frames")
        original_header = {k: v for k, v in original.items() if k != "frames"}
        if {k: v for k, v in train.items() if k != "frames"} != original_header:
            raise ValueError(f"{scene}: derivative transform header changed")
        if test.get("frames") != []:
            raise ValueError(f"{scene}: two-frame derivative must have no test frames")
        names = [Path(f["file_path"]).name for f in train["frames"]]
        if names != ["0002", "0003"]:
            raise ValueError(f"{scene}: expected actual public first frames 0002,0003; got {names}")
        required = [f'{f["file_path"]}.png' for f in train["frames"]]
        required += ["points3d.ply", f"flows_flo/flow_fwd_{names[0]}.flo",
                     f"flows_flo/flow_bwd_{names[1]}.flo", f"depth_distill/{names[1]}.npy"]
        for rel in required:
            if not (fixture / rel).is_file():
                raise FileNotFoundError(fixture / rel)
            if not (source / rel).is_file() or file_sha(fixture / rel) != file_sha(source / rel):
                raise ValueError(f"{scene}: fixture resource differs from the downloaded public resource: {rel}")
        with (fixture / "points3d.ply").open("rb") as stream:
            header = stream.read(4096).split(b"end_header", 1)[0]
        if b"element vertex 100000\n" not in header.replace(b"\r\n", b"\n"):
            raise ValueError(f"{scene}: require the original 100000-point public PLY")
        if file_sha(fixture / "points3d.ply") != file_sha(source / "points3d.ply"):
            raise ValueError(f"{scene}: fixture PLY differs from the public PLY")
        result[scene] = {"frames": train["frames"], "required_files": required,
                         "original_train_sha256": file_sha(source / "transforms_train.json"),
                         "original_test_sha256": file_sha(source / "transforms_test.json"),
                         "derivative_scope": "first two original training frames only; empty test split; no frame/time/pose rewrites"}
    return result


def snapshot(uuid=GPU_UUID):
    queries = {
        "gpu": ["nvidia-smi", "-i", uuid, "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"],
        "processes": ["nvidia-smi", "-i", uuid, "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
    }
    result = {"unix": time.time(), "requested_uuid": uuid, "shared_workload": True,
              "host_load": list(os.getloadavg()), "queries": {}}
    for key, command in queries.items():
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result["queries"][key] = {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["queries"][key] = {"error": str(exc)}
    try:
        result["host_meminfo"] = Path("/proc/meminfo").read_text()
    except OSError:
        result["host_meminfo"] = None
    return result


def require_pinned_snapshot(record, uuid=GPU_UUID):
    row = record["queries"].get("gpu", {})
    if row.get("returncode") != 0 or not row.get("stdout", "").startswith(uuid + ","):
        raise RuntimeError(f"cannot verify assigned GPU UUID {uuid}: {row}")


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if lo != hi else values[lo]


def window_checks(iteration_ms, policy=POLICY):
    if not iteration_ms or any(not math.isfinite(x) or x <= 0 for x in iteration_ms):
        return {"valid": False, "reason": "missing, nonfinite or nonpositive iteration events"}
    median = statistics.median(iteration_ms)
    spread = (quantile(iteration_ms, .90) - quantile(iteration_ms, .10)) / median
    half = max(1, len(iteration_ms) // 2)
    drift = abs(statistics.median(iteration_ms[:half]) - statistics.median(iteration_ms[-half:])) / median
    return {"valid": spread <= policy["within_window_p90_p10_limit"] and drift <= policy["within_window_half_drift_limit"],
            "p90_p10_over_median": spread, "half_window_drift": drift, "median_event_ms": median}


def run_checks(windows, smoke=False, policy=POLICY):
    if not windows:
        return {"valid": False, "reason": "no windows"}
    times = [w["wall_ms_per_iteration"] for w in windows]
    if any(not math.isfinite(x) or x <= 0 for x in times):
        return {"valid": False, "reason": "nonfinite or nonpositive wall time"}
    spread = (max(times) - min(times)) / statistics.median(times)
    good = all(w["noise"]["valid"] for w in windows) and spread <= policy["window_range_limit"]
    return {"valid": good and not smoke, "smoke_only": smoke, "window_range_over_median": spread,
            "median_ms": statistics.median(times), "all_windows_valid": good}


def compare_timing(before, candidate, after, policy=POLICY):
    """No target speedup and no formal benefit verdict on the explicitly shared GPU."""
    result = {"status": "EXPLORATORY", "scope": SCOPE, "shared_workload": True,
              "formal_benefit_verdict": "UNAVAILABLE_SHARED_WORKLOAD", "errors": []}
    records = [before, candidate, after]
    fingerprints = [r["fingerprint"]["comparable"] for r in records]
    if fingerprints[0] != fingerprints[1] or fingerprints[0] != fingerprints[2]:
        result["errors"].append("run fingerprints differ outside the declared implementation selection")
    if not all(r["timing"]["valid"] for r in records):
        result["errors"].append("one or more timing arms failed per-window or within-window noise checks")
    b1, b2 = before["windows"], after["windows"]
    if len(b1) != len(b2) or len(candidate["windows"]) != len(b1):
        result["errors"].append("window counts differ")
    ratios = []
    for index, (left, right) in enumerate(zip(b1, b2)):
        a, b = left["wall_ms_per_iteration"], right["wall_ms_per_iteration"]
        gap = abs(a - b) / statistics.median([a, b])
        ratios.append(gap)
        if gap > policy["reference_window_drift_limit"]:
            result["errors"].append(f"baseline window {index} drift {gap:.6f} exceeds frozen policy")
    clocks = [statistics.median([w["power"]["sm_clock_mhz_median"] for w in r["windows"]])
              if all(w.get("power", {}).get("sm_clock_mhz_median") is not None for w in r["windows"]) else None
              for r in records]
    result["arm_median_sm_clock_mhz"] = clocks
    result["clock_comparability_available"] = all(x is not None for x in clocks)
    if all(x is not None for x in clocks):
        clock_gap = (max(clocks) - min(clocks)) / statistics.median(clocks)
        result["clock_relative_gap"] = clock_gap
        if clock_gap > policy["clock_relative_gap_limit"]:
            result["errors"].append("GPU clock differences exceed frozen comparability policy")
    else:
        result["clock_comparison_status"] = "UNAVAILABLE_INSUFFICIENT_ACTIVE_IN_WINDOW_SAMPLES"
    if result["errors"]:
        result["status"] = "INVALID"
        return result
    bm = statistics.median([w["wall_ms_per_iteration"] for w in b1 + b2])
    cm = candidate["timing"]["median_ms"]
    noise = max(r["timing"]["window_range_over_median"] for r in records)
    reference_gap = max(ratios)
    resolution = max(policy["small_gain_resolution_floor"], 2 * noise, 2 * reference_gap)
    result.update({"observed_speedup": bm / cm, "baseline_ms": bm, "candidate_ms": cm,
                   "baseline_window_relative_gaps": ratios, "noise_resolution_relative": resolution,
                   "resolved_above_observed_noise": bm / cm > 1 + resolution and result["clock_comparability_available"],
                   "benefit_claim": "exploratory observation only; no target speedup and no full-training verdict"})
    return result
