"""One independent process per arm/case. Never invoke this worker directly."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (GPU_UUID, POLICY, SCOPE, file_sha, read_json, require_pinned_snapshot,
                    run_checks, snapshot, window_checks, write_json)


class Telemetry:
    """Read only the assigned GPU. Sampling is identical for all timing arms."""
    def __init__(self, uuid):
        self.uuid, self.samples, self.stop = uuid, [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        command = ["nvidia-smi", "-i", self.uuid,
                   "--query-gpu=power.draw,clocks.sm,clocks.mem,power.limit,utilization.gpu,memory.used",
                   "--format=csv,noheader,nounits"]
        while not self.stop.is_set():
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5)
                self.samples.append({"unix": time.time(), "raw": result.stdout.strip(), "returncode": result.returncode})
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.samples.append({"unix": time.time(), "error": str(exc)})
            self.stop.wait(.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)


def finite_description(value):
    import torch
    value = value.detach()
    bad = ~torch.isfinite(value)
    bad_count = int(bad.sum().item())
    return {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel(),
            "nonfinite_count": bad_count, "nonzero_count": int(torch.count_nonzero(value).item()),
            "first_nonfinite_flat_indices": torch.nonzero(bad.reshape(-1), as_tuple=False)[:16].flatten().cpu().tolist()}


def finite_report(adapter, losses):
    """Do not hide upstream SVD/culled-Gaussian gradients behind masks."""
    tensors = {"loss/" + n: v for n, v in losses.items()}
    for name, param in adapter.group.named_parameters():
        if param.grad is not None:
            tensors["gradient/" + name] = param.grad
    for index, package in enumerate(adapter.group.last_packages):
        for name in ("render", "proj_means_2D", "conic_2D", "conic_2D_inv", "weight_per_gs_pixel", "x_mu"):
            tensors[f"render{index}/{name}"] = package[name]
        if package["viewspace_points"].grad is not None:
            tensors[f"gradient/render{index}/viewspace_points"] = package["viewspace_points"].grad
    for index, value in enumerate(adapter.group.deformation_inputs):
        if value.grad is not None:
            tensors[f"gradient/deformation{index}"] = value.grad
    result = {name: finite_description(tensor) for name, tensor in tensors.items()}
    failed = {name: desc for name, desc in result.items() if desc["nonfinite_count"]}
    return {"finite": not failed, "tensors": result, "nonfinite_tensors": failed,
            "note": "full tensors include culled Gaussians; no NaN masking, changed detaches or gate relaxation"}


def versions():
    import torch
    from numerics.record import math_mode
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "math_mode": math_mode(), "torch_threads": torch.get_num_threads(),
            "gpu_name": torch.cuda.get_device_name(0), "world_size": 1,
            "per_rank_batch": 1, "device_count": torch.cuda.device_count(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}


def fingerprint(job, adapter, mode):
    selection = job["selection"]
    comparable = {"gpu_uuid": job["gpu_uuid"], "scope": SCOPE, "shared_workload": True,
                  "fixture_sha256": job["fixture_sha256"], "judge_sha256": job["judge_sha256"],
                  "runtime": adapter.runtime, "versions": versions(),
                  "workload": job["workload"], "mode": mode, "seed": POLICY["seed"],
                  "warmup": job.get("warmup"), "windows": job.get("windows"), "iterations": job.get("iterations"),
                  "cache_state": "cold private dynamic compile caches; hashencoder startup cache seed recorded separately", "profiler": job["operation"] == "profile",
                  "nvtx": False, "synchronization": "window_boundaries_only; upstream flow retains its own host synchronizations",
                  "iteration_cuda_events": True, "gpu_telemetry_period_seconds": .2,
                  "clock_active_utilization_percent": POLICY["clock_active_utilization_percent"],
                  "clock_min_active_samples_per_window": POLICY["clock_min_active_samples_per_window"],
                  "protocol": job["protocol"],
                  "hashencoder_startup": job["hashencoder_startup"],
                  "hashencoder_build_manifest_sha256": job["hashencoder_build_manifest_sha256"],
                  "copy_memory": "upstream camera/flow loading; pageable CPU input tensors; no harness pinned-memory changes"}
    return {"comparable": comparable,
            "implementation": {**selection, "flow_sha256": file_sha(selection["flow_file"]),
                               "extension_sha256": file_sha(selection["extension"])},
            "command": sys.argv, "job": job}


def time_workload(job, adapter):
    import torch
    backward = job["mode"] == "forward-backward"
    for index in range(job["warmup"]):
        adapter.execute(index, backward=backward)
    torch.cuda.synchronize()
    windows = []
    for wi in range(job["windows"]):
        before = snapshot(job["gpu_uuid"])
        require_pinned_snapshot(before, job["gpu_uuid"])
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(job["iterations"])]
        host_ms = []
        torch.cuda.synchronize()
        begin = time.time()
        tic = time.perf_counter()
        for index, (start, end) in enumerate(events):
            host_tic = time.perf_counter()
            start.record()
            losses = adapter.execute(job["warmup"] + wi * job["iterations"] + index, backward=backward)
            end.record()
            host_ms.append((time.perf_counter() - host_tic) * 1000)
            del losses
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - tic) * 1000
        finish = time.time()
        per_iteration = [a.elapsed_time(b) for a, b in events]
        after = snapshot(job["gpu_uuid"])
        require_pinned_snapshot(after, job["gpu_uuid"])
        windows.append({"index": wi, "start_unix": begin, "end_unix": finish,
                        "wall_ms": wall_ms, "wall_ms_per_iteration": wall_ms / job["iterations"],
                        "iteration_event_ms": per_iteration, "host_enqueue_ms": host_ms,
                        "noise": window_checks(per_iteration), "gpu_before": before, "gpu_after": after})
    # One untimed diagnostic replay checks the path actually used by timing.
    losses = adapter.execute(job["warmup"] + job["windows"] * job["iterations"], backward=backward)
    torch.cuda.synchronize()
    report = finite_report(adapter, losses)
    return {"windows": windows, "timing": run_checks(windows, smoke=job.get("smoke", False)),
            "finite": report, "invariants": adapter.invariants()}


def attach_power(windows, samples, iterations):
    for window in windows:
        observations = []
        active_clocks = []
        for sample in samples:
            if window["start_unix"] <= sample["unix"] <= window["end_unix"] and sample.get("returncode") == 0:
                try:
                    values = [float(v.strip()) for v in sample["raw"].split(",")]
                    observations.append((values[0], values[1], values[3]))
                    if values[4] > POLICY["clock_active_utilization_percent"]:
                        active_clocks.append(values[1])
                except (ValueError, IndexError):
                    pass
        interior_count = len(observations)
        # Short raster windows can be shorter than the200ms sampling period.
        # Keep their two boundary observations explicitly as an estimate.
        for key in ("gpu_before", "gpu_after"):
            try:
                values = window[key]["queries"]["gpu"]["stdout"].split(",")
                observations.append((float(values[7]), float(values[5]), float(values[8])))
            except (ValueError, KeyError, IndexError):
                pass
        if observations:
            watts = statistics.mean(x[0] for x in observations)
            duration = window["end_unix"] - window["start_unix"]
            clock = statistics.median(active_clocks) if len(active_clocks) >= POLICY["clock_min_active_samples_per_window"] else None
            window["power"] = {"power_watts_mean": watts, "sm_clock_mhz_median": clock,
                               "clock_active_samples": len(active_clocks),
                               "clock_method": "interior samples only; GPU utilization>5%; at least2 samples/window; shared GPU activity cannot be attributed to one process",
                               "near_power_cap_fraction": statistics.mean(x[0] >= .98 * x[2] for x in observations),
                               "estimated_joules": watts * duration, "estimated_joules_per_iteration": watts * duration / iterations,
                               "interior_samples": interior_count, "boundary_samples": len(observations) - interior_count,
                               "method": "200ms samples plus before/after snapshots; approximate energy, especially for sub200ms windows"}
        else:
            window["power"] = {"sm_clock_mhz_median": None, "error": "no parseable power/clock data"}


def main(job_path):
    started = time.time()
    job = read_json(job_path)
    output = Path(job["worker_out"])
    result_path = output / "result.json"
    result = {"schema": "pidg-module-worker@1", "scope": SCOPE, "status": "ERROR",
              "shared_workload": True, "operation": job["operation"], "start_unix": started}
    try:
        if job["gpu_uuid"] != GPU_UUID or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
            raise RuntimeError("worker is not pinned to the single authorized UUID")
        before = snapshot(job["gpu_uuid"])
        require_pinned_snapshot(before, job["gpu_uuid"])
        result["gpu_before"] = before
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("worker must see exactly one assigned GPU")
        from pidg_adapter import PIDGAdapter
        from numerics.record import Recorder
        # Do not impose deterministic switches on the CUDA atomics path. Both
        # numerical recording and timing retain the upstream default math mode.
        adapter = PIDGAdapter(job["selection"], job["fixture"], job["scene"], job["workload"],
                              output / "visualizations", job["config"], initial_state=job.get("initial_state"))
        if job["operation"] == "record":
            case = {"name": f'{job["scene"]}__{job["workload"]}', "seed": POLICY["seed"],
                    "steps": job["steps"], "phase": job["workload"]}
            Recorder(adapter, job["record_out"], dtype_trace=False, sketch_dim=POLICY["sketch_dim"]).run(case)
            result.update({"record_case": case, "record_out": job["record_out"]})
            # Recorder captures all finite flags, gradients, Adam deltas/state,
            # output hashes/shapes/dtypes and workload/RNG invariants.
        else:
            adapter.build(POLICY["seed"])
        torch.cuda.synchronize()
        result["startup_seconds"] = time.time() - started
        if job["operation"] == "prepare":
            adapter.save_initial(job["save_initial"])
            result["initial_sha256"] = file_sha(job["save_initial"])
        elif job["operation"] == "smoke":
            losses = adapter.execute(0, backward=True)
            torch.cuda.synchronize()
            result["finite"] = finite_report(adapter, losses)
            result["invariants"] = adapter.invariants()
            result["losses"] = {n: float(v.detach()) for n, v in losses.items()}
            if not result["finite"]["finite"]:
                result["status"] = "UPSTREAM_NONFINITE" if job["selection"]["variant"] == "baseline" else "CANDIDATE_NONFINITE"
        elif job["operation"] == "time":
            with Telemetry(job["gpu_uuid"]) as monitor:
                result.update(time_workload(job, adapter))
            result["telemetry"] = monitor.samples
            attach_power(result["windows"], monitor.samples, job["iterations"])
            if not result["finite"]["finite"]:
                result["status"] = "NONFINITE"
        elif job["operation"] == "profile":
            for index in range(job["warmup"]):
                adapter.execute(index, backward=job["mode"] == "forward-backward")
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True, with_stack=False) as profiler:
                for index in range(job["iterations"]):
                    adapter.execute(index, backward=job["mode"] == "forward-backward")
                    profiler.step()
            torch.cuda.synchronize()
            profiler.export_chrome_trace(str(output / "trace.json"))
            (output / "operators.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
            result["trace"] = str(output / "trace.json")
            result["instrumented_profile_not_timing"] = True
        result["invariants"] = adapter.invariants()
        result["runtime"] = adapter.runtime
        result["fingerprint"] = fingerprint(job, adapter, job.get("mode", "forward-backward"))
        if result["status"] == "ERROR":
            result["status"] = "OK"
    except Exception as exc:
        result.update({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        result["gpu_after"] = snapshot(job["gpu_uuid"])
        result["end_unix"] = time.time()
        result["cost"] = {"gpu_hours": (result["end_unix"] - started) / 3600,
                          "count": 1, "includes_startup_warmup_failures": True}
        write_json(result_path, result)
        write_json(output / "cleanup.json", {"removed_bytes": 0, "retained": "owned caches, logs and diagnostics retained for provenance; no shared path deleted"})
    print(json.dumps({"status": result["status"], "result": str(result_path)}), flush=True)
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
