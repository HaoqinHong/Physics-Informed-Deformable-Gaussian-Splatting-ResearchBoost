#!/usr/bin/env python3
"""Frozen exploratory PIDG module bench, not the ResearchBoost training judge.

CPU-safe entry points: --help, inspect, verify. GPU entry points are launched in
fresh processes pinned to the one authorized UUID. This file never uses SSH.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (GPU_UUID, POLICY, ROOT, SCENES, SCOPE, VARIANTS, WORKLOADS, compare_timing,
                    file_sha, judge_manifest, manifest, new_directory, read_json, selections,
                    validate_fixture, value_sha, write_json)
from record_projection import (RADII, REASON as PROJECTION_REASON, SCHEMA as PROJECTION_SCHEMA,
                               comparable_numeric_fingerprint, discrete_exact_failures,
                               complete_reference_view, project_records, read_manifests)


def configuration(args):
    config = read_json(ROOT / "task/bench.json")
    for name, key in (("baseline", "baseline"), ("fixture", "fixture"), ("public_fixture", "public_fixture")):
        if getattr(args, name, None):
            config[key] = str(Path(getattr(args, name)).resolve())
    if config["gpu_uuid"] != GPU_UUID:
        raise ValueError("configuration must use the single authorized GPU UUID")
    return config


def initial_manifests(frozen):
    return {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in sorted((Path(frozen) / "initial").glob("*.pt"))}


def record_manifest(path):
    root = Path(path)
    result = {}
    for pattern in ("*.r*.json", "*.r*.s*.pt"):
        for file in sorted(root.glob(pattern)):
            result[file.name] = {"bytes": file.stat().st_size, "sha256": file_sha(file)}
    if not result:
        raise ValueError(f"missing numerical records: {root}")
    return result


def verify_frozen(path):
    path = Path(path).resolve()
    contract = read_json(path / "formulation.json")
    if contract.get("status") != "FROZEN":
        raise ValueError("baseline formulation is not frozen/certified; candidate runs are unavailable")
    expected = contract["judge_manifest"]
    if judge_manifest() != expected:
        raise ValueError("judge code/config changed after baseline freeze")
    if contract["policy"] != POLICY:
        raise ValueError("timing/numerical policy changed after freeze")
    config = contract["config"]
    checks = {
        "baseline_source": manifest(config["baseline"], source_only=True),
        "fixture": manifest(config["fixture"]),
        "initial_states": initial_manifests(path),
    }
    for name, actual in checks.items():
        if actual != contract[name + "_manifest"]:
            raise ValueError(f"frozen {name} changed")
    if validate_fixture(config["fixture"], config["public_fixture"]) != contract["fixture_provenance"]:
        raise ValueError("public transform/PLY provenance changed")
    if file_sha(path / "thresholds.json") != contract["thresholds_file_sha256"]:
        raise ValueError("frozen baseline-self thresholds changed")
    for relative, expected in contract["record_manifests"].items():
        if record_manifest(path / relative) != expected:
            raise ValueError(f"frozen numerical records changed: {relative}")
    for absolute, expected in contract.get("raw_record_manifests", {}).items():
        if record_manifest(absolute) != expected:
            raise ValueError(f"archived raw numerical records changed: {absolute}")
    source = contract.get("raw_source_formulation")
    if source and file_sha(source["path"]) != source["sha256"]:
        raise ValueError("archived raw producer formulation changed")
    producer = contract.get("raw_record_producer", {}).get("producer_manifest", {})
    current = judge_manifest()
    if any(current.get(name) != desc for name, desc in producer.items()):
        raise ValueError("raw record producer code changed; CPU projection cannot certify a different producer")
    return contract


def schedule(args, workload):
    if args.smoke:
        chosen = {"warmup": 1, "windows": 1, "iterations": 1, "protocol": "SMOKE_ONLY"}
    elif args.short and workload == "triple-flow":
        chosen = {"warmup": 1, "windows": 3, "iterations": 3, "protocol": "SHORT_EXPLORATORY"}
    else:
        chosen = {"warmup": POLICY["warmup"], "windows": POLICY["windows"], "iterations": POLICY["iterations"], "protocol": "STANDARD_EXPLORATORY"}
    for key in ("warmup", "windows", "iterations"):
        value = getattr(args, key, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"--{key} must be positive")
            chosen[key] = value
    if any(chosen[key] < POLICY[key] for key in ("warmup", "windows", "iterations")) and not (args.short or args.smoke):
        raise ValueError("a shorter-than-standard schedule requires explicit --short or --smoke")
    if not args.smoke and any(chosen[key] < POLICY[key] for key in ("warmup", "windows", "iterations")):
        chosen["protocol"] = "SHORT_EXPLORATORY"
    return chosen


def worker_job(args, config, out, operation, scene, workload, variant="baseline", frozen=None, **extra):
    out = new_directory(out)
    selection = selections(config["baseline"], args.tree, variant)
    job = {"operation": operation, "worker_out": str(out), "scene": scene, "workload": workload,
           "selection": selection, "config": config, "fixture": config["fixture"], "gpu_uuid": GPU_UUID,
           "fixture_sha256": args._fixture_sha, "judge_sha256": args._judge_sha,
           "smoke": args.smoke, **schedule(args, workload), **extra}
    if frozen:
        job["initial_state"] = str(Path(frozen).resolve() / "initial" / f"{scene}.pt")
    # Copy a known compatible build into the worker's PRIVATE directory. Normal
    # upstream backend initialization may run ninja here; it cannot modify the
    # viewer's/shared build. No module stubs or replacement hashes are used.
    jit_source = Path(args.hashencoder_build).resolve() if args.hashencoder_build else Path(config["baseline"]) / "tmp_build"
    if jit_source.is_dir():
        job["hashencoder_build_manifest_sha256"] = value_sha(manifest(jit_source))
        shutil.copytree(jit_source, out / "tmp_build", ignore=shutil.ignore_patterns("lock"))
        job["hashencoder_startup"] = "private copy of compatible build; unchanged upstream import"
    else:
        job["hashencoder_build_manifest_sha256"] = None
        job["hashencoder_startup"] = "no compatible cache supplied; unchanged upstream import may compile in private cwd"
    job_path = out / "job.json"
    write_json(job_path, job)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": GPU_UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(out / "cache/inductor"), "TRITON_CACHE_DIR": str(out / "cache/triton"),
                "TORCH_EXTENSIONS_DIR": str(out / "cache/extensions"), "MPLCONFIGDIR": str(out / "cache/matplotlib"),
                "TMPDIR": str(out / "tmp")})
    (out / "tmp").mkdir()
    command = [sys.executable, str(ROOT / "bench/worker.py"), str(job_path)]
    started = time.time()
    timed_out = None
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        proc = subprocess.Popen(command, cwd=out, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        limit = POLICY["smoke_timeout_seconds"] if operation == "smoke" else POLICY["worker_timeout_seconds"]
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = {"status": "TIMEOUT", "pid": proc.pid, "process_group": proc.pid,
                         "timeout_seconds": limit, "signals": [],
                         "scope": "only this worker's new session/process group; viewer and unrelated processes untouched"}
            for sig, grace in ((signal.SIGTERM, POLICY["worker_terminate_grace_seconds"]),
                               (signal.SIGKILL, POLICY["worker_kill_grace_seconds"])):
                if proc.poll() is not None:
                    break
                try:
                    os.killpg(proc.pid, sig)
                    timed_out["signals"].append(sig.name)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    continue
            timed_out["exit_code"] = proc.poll()
            timed_out["worker_still_alive_after_bounded_kill"] = proc.poll() is None
            write_json(out / "timeout.json", timed_out)
    result_file = out / "result.json"
    if timed_out:
        previous = read_json(result_file) if result_file.is_file() else None
        write_json(result_file, {**timed_out, "worker_result_before_timeout": previous,
                                 "cost": {"gpu_hours": (time.time() - started) / 3600, "count": 1,
                                          "includes_startup_warmup_failures": True}})
        raise RuntimeError(f"worker exceeded fixed timeout; see {out / 'timeout.json'}")
    if not result_file.is_file():
        raise RuntimeError(f"worker exit {proc.returncode} without result artifact; inspect {out}/stderr.log")
    result = read_json(result_file)
    signatures = (r"falling back to eager", r"fallback to eager", r"graph capture failed",
                  r"CUDA out of memory", r"illegal memory access", r"device-side assert")
    hits = []
    for logfile in (out / "stdout.log", out / "stderr.log"):
        for line in logfile.read_text(errors="replace").splitlines():
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in signatures):
                hits.append({"file": logfile.name, "line": line[:1000]})
    result["invalid_log_signatures"] = hits
    if hits and result["status"] == "OK":
        result["status"] = "INVALID_LOG_SIGNATURE"
    result["launch"] = {"command": command, "returncode": proc.returncode, "duration_seconds": time.time() - started}
    write_json(result_file, result)
    if proc.returncode != 0 or result["status"] != "OK":
        raise RuntimeError(f"worker {result['status']}; exact diagnostic in {result_file}")
    if operation == "profile" and not Path(result.get("trace", "")).is_file():
        raise RuntimeError("profile exit 0 without a trace")
    if operation == "record":
        case = result["record_case"]["name"]
        if not (Path(job["record_out"]) / f"{case}.r0.json").is_file():
            raise RuntimeError("record exit 0 without numerical manifest")
    print(json.dumps({"scene": scene, "workload": workload, "variant": variant,
                      "operation": operation, "result": str(result_file)}), flush=True)
    return result


def certify_records(ref, variants, thresholds, out):
    from numerics.judge import calibrate, compare, leave_one_out
    from numerics.selftest import run as generic_selftest
    reports = []
    for variant in variants:
        report = compare(ref, variant, thresholds)
        strict = strict_manifest_checks(ref, variant)
        reports.append({"variant": str(variant), "generic_status": report["status"], "strict": strict,
                        "checked": report["checked"], "bitwise_identical": report["bitwise_identical"],
                        "unresolved": report["unresolved"], "exact_failures": report["exact_failures"], "errors": report["errors"]})
    loo = leave_one_out(ref, variants, k_fwd=POLICY["k_fwd"], k_bwd=POLICY["k_bwd"],
                        floor_fwd=POLICY["floor_fwd"], floor_bwd=POLICY["floor_bwd"])
    # The generic summary truncates examples. Preserve the complete failed CPU
    # replay so an additional continuous failure cannot hide behind five radii.
    for item in loo["results"]:
        if item["status"] != "PASS":
            held = item["variant"]
            th = calibrate(ref, [v for v in variants if str(v) != held], k_fwd=POLICY["k_fwd"], k_bwd=POLICY["k_bwd"],
                           floor_fwd=POLICY["floor_fwd"], floor_bwd=POLICY["floor_bwd"])
            detail = compare(ref, held, th)
            name = f"leave-one-out-detail-{Path(held).name}.json"
            write_json(Path(out) / name, detail)
            item["full_unresolved"] = detail["unresolved"]
            item["detail_artifact"] = name
    negative = generic_selftest(ref, variants[0], thresholds, out=str(Path(out) / "generic-selftest"))
    strict_negative = strict_selftests(ref, variants[0], out)
    result = {"baseline_self": reports, "leave_one_out": loo, "generic_negative_selftests": negative,
              "strict_negative_selftests": strict_negative,
              "ok": all(r["generic_status"] == "PASS" and r["strict"]["ok"] for r in reports) and loo["ok"] and negative["ok"] and strict_negative["ok"],
              "scope": SCOPE,
              "control_arm": "generic numerical perturbation mutations only; no training-trajectory/lr control arm, outside module scope"}
    write_json(Path(out) / "certification.json", result)
    return result


def strict_selftests(ref, variant, out):
    """Record mutations prove the additional exact guards are connected."""
    results = []
    original_files = sorted(Path(variant).glob("*.r*.json"))
    first = read_json(original_files[0])
    gradient = next((n for n, d in first["steps"][0]["tensors"].items()
                     if n.startswith("grad/") and d.get("norm", 0) > 0 and d["shape"]), None)
    if gradient is None:
        return {"ok": False, "error": "no nonzero baseline gradient available to test dead-gradient guard"}
    mutations = ("unmodified", "shape", "dtype", "dead_gradient", "nonfinite", "math_mode", "extra_case",
                 "initial_gs_hash", "initial_radii_hash", "postupdate_gs_shape", "postupdate_gs_dtype",
                 "postupdate_radii_shape", "postupdate_radii_dtype")
    with tempfile.TemporaryDirectory(prefix="strict-selftest-", dir=out) as temp:
        for mutation in mutations:
            directory = Path(temp) / mutation
            directory.mkdir()
            for path in original_files:
                shutil.copyfile(path, directory / path.name)
            changed = read_json(directory / original_files[0].name)
            descriptor = changed["steps"][0]["tensors"][gradient]
            if mutation == "shape":
                descriptor["shape"][-1] += 1
            elif mutation == "dtype":
                descriptor["dtype"] = "float64" if descriptor["dtype"] != "float64" else "float32"
            elif mutation == "dead_gradient":
                descriptor["norm"] = 0.0
            elif mutation == "nonfinite":
                descriptor["finite"] = False
            elif mutation == "math_mode":
                changed["steps"][0]["math_mode"]["__mutation__"] = True
            elif mutation == "extra_case":
                extra = copy.deepcopy(changed)
                extra["case"]["name"] += "__unexpected"
                write_json(directory / "unexpected.r0.json", extra)
            elif mutation == "initial_gs_hash":
                changed["steps"][0]["invariants"]["gs_per_pixel"][0]["sha256"] = "__mutation__"
            elif mutation == "initial_radii_hash":
                name = next(n for n in changed["steps"][0]["tensors"] if RADII.fullmatch(n))
                changed["steps"][0]["tensors"][name]["sha256"] = "__mutation__"
            elif mutation.startswith("postupdate_gs_"):
                desc = changed["steps"][1]["invariants"]["gs_per_pixel"][0]
                if mutation.endswith("shape"):
                    desc["shape"][-1] += 1
                else:
                    desc["dtype"] = "float64"
            elif mutation.startswith("postupdate_radii_"):
                desc = next(iter(changed["steps"][1]["invariants"]["postupdate_discrete_shapes"].values()))
                if mutation.endswith("shape"):
                    desc["shape"][-1] += 1
                else:
                    desc["dtype"] = "float64"
            write_json(directory / original_files[0].name, changed)
            result = strict_manifest_checks(ref, directory)
            caught = result["ok"] if mutation == "unmodified" else not result["ok"]
            results.append({"mutation": mutation, "ok": caught, "report": result})
    return {"ok": all(r["ok"] for r in results), "results": results}


def strict_manifest_checks(ref, cand):
    """Stronger exact guards around the unchanged generic diagnostic component."""
    refs, cands = read_manifests(ref), read_manifests(cand)
    failures, nonfinite = [], []
    if refs.keys() != cands.keys():
        failures.append({"check": "case_set", "missing": [str(x) for x in refs.keys() - cands.keys()],
                         "unexpected": [str(x) for x in cands.keys() - refs.keys()]})
    for key in refs.keys() & cands.keys():
        rm, cm = refs[key], cands[key]
        if len(rm["steps"]) != len(cm["steps"]):
            failures.append({"case": key[0], "rank": key[1], "check": "step_count"})
        failures.extend(discrete_exact_failures(rm, cm))
        for r, c in zip(rm["steps"], cm["steps"]):
            where = {"case": key[0], "rank": key[1], "step": r["step"]}
            if r.get("math_mode") != c.get("math_mode"):
                failures.append({**where, "check": "math_mode"})
            if r["tensors"].keys() != c["tensors"].keys():
                failures.append({**where, "check": "tensor_names"})
            for name in r["tensors"].keys() & c["tensors"].keys():
                rd, cd = r["tensors"][name], c["tensors"][name]
                for field in ("shape", "dtype"):
                    if rd[field] != cd[field]:
                        failures.append({**where, "check": field, "tensor": name, "baseline": rd[field], "candidate": cd[field]})
                if not rd["finite"] or not cd["finite"]:
                    nonfinite.append({**where, "tensor": name, "baseline_finite": rd["finite"], "candidate_finite": cd["finite"]})
                if name.startswith("grad/") and (rd.get("norm") or 0) > 0 and cd.get("norm") == 0:
                    failures.append({**where, "check": "dead_gradient", "tensor": name})
    return {"ok": not failures and not nonfinite, "failures": failures, "nonfinite": nonfinite}


def run_baseline(args, config, out):
    from numerics.judge import calibrate, thresholds_sha
    if args.scene or args.workload or args.smoke or args.short:
        raise ValueError("baseline freeze requires all five scenes, all three workloads and the frozen numerical plan")
    provenance = validate_fixture(config["fixture"], config["public_fixture"])
    baseline_sources = manifest(config["baseline"], source_only=True)
    fixture_files = manifest(config["fixture"])
    judges = judge_manifest()
    (out / "initial").mkdir()
    for scene in SCENES:
        result = worker_job(args, config, out / "workers/prepare" / scene, "prepare", scene, "raster-regular",
                            save_initial=str(out / "initial" / f"{scene}.pt"))
        write_json(out / "initial" / f"{scene}.json", result["runtime"])
    raw_dirs = []
    reference_fingerprints = {}
    for repetition in range(POLICY["baseline_records"]):
        record_out = out / "raw-records" / f"r{repetition}"
        record_out.mkdir(parents=True)
        raw_dirs.append(record_out)
        for scene in SCENES:
            for workload in WORKLOADS:
                result = worker_job(args, config, out / "workers" / f"record{repetition}" / scene / workload,
                                    "record", scene, workload, frozen=out, record_out=str(record_out), steps=POLICY["numeric_steps"])
                fingerprint = result["fingerprint"]["comparable"]
                key = f"{scene}/{workload}"
                if repetition == 0:
                    reference_fingerprints[key] = fingerprint
                elif fingerprint != reference_fingerprints[key]:
                    raise ValueError(f"baseline-self fingerprint differs for {key}")
    record_dirs = []
    for index, raw in enumerate(raw_dirs):
        projected = out / "records" / f"r{index}"
        project_records(raw, projected)
        record_dirs.append(projected)
    thresholds = calibrate(record_dirs[0], record_dirs[1:], k_fwd=POLICY["k_fwd"], k_bwd=POLICY["k_bwd"],
                           floor_fwd=POLICY["floor_fwd"], floor_bwd=POLICY["floor_bwd"])
    write_json(out / "thresholds.json", thresholds)
    certification = certify_records(record_dirs[0], record_dirs[1:], thresholds, out)
    # Fingerprints are taken before GPU work and rechecked before freeze so
    # concurrent source/fixture changes cannot become an unnoticed new baseline.
    unchanged = (baseline_sources == manifest(config["baseline"], source_only=True)
                 and fixture_files == manifest(config["fixture"]) and judges == judge_manifest())
    contract = {"schema": "pidg-module-formulation@1", "status": "FROZEN" if certification["ok"] and unchanged else "BASELINE_UNCERTIFIED",
                "scope": SCOPE, "full_training_acceptance": False, "shared_workload": True, "config": config, "policy": POLICY,
                "frozen_unix": time.time(), "gpu_uuid": GPU_UUID, "baseline_source_manifest": baseline_sources,
                "fixture_manifest": fixture_files, "fixture_provenance": provenance, "judge_manifest": judges,
                "evaluator_sha256": value_sha(judges), "initial_states_manifest": initial_manifests(out),
                "record_manifests": {str(p.relative_to(out)): record_manifest(p) for p in record_dirs},
                "raw_record_manifests": {str(p): record_manifest(p) for p in raw_dirs},
                "record_projection": {"schema": PROJECTION_SCHEMA, "reason": PROJECTION_REASON},
                "reference_records": "records/r0", "thresholds_file_sha256": file_sha(out / "thresholds.json"),
                "reference_numeric_fingerprints": reference_fingerprints,
                "thresholds_sha256": thresholds_sha(thresholds), "inputs_unchanged_during_formulation": unchanged,
                "baseline_bitwise": all(r["checked"] == r["bitwise_identical"] for r in certification["baseline_self"]),
                "baseline_self_drift": "measured from four independent original-baseline processes per case; gates frozen before candidate",
                "no_fp64_path": "CUDA rasterizer lacks fp64; tolerance excess is unresolved/needs review, never silently accepted",
                "trajectory_and_convergence": "not measured; no full ResearchBoost four-gate PASS is possible"}
    write_json(out / "formulation.json", contract)
    return {"status": contract["status"], "formulation": str(out / "formulation.json"), "scope": SCOPE}


def run_recertification(args, out):
    """CPU-only pre-candidate correction; archive identity is never rewritten."""
    if not args.source_frozen or not args.pre_candidate:
        raise ValueError("recertify requires --source-frozen and explicit --pre-candidate attestation")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source = Path(args.source_frozen).resolve()
    original = read_json(source / "formulation.json")
    original_cert = read_json(source / "certification.json")
    if original["status"] != "BASELINE_UNCERTIFIED" or not original["inputs_unchanged_during_formulation"]:
        raise ValueError("CPU correction requires the unchanged, uncertified original baseline formulation")
    if original["policy"] != POLICY:
        raise ValueError("continuous/timing policy changed; this is not a discrete-view-only correction")
    config = original["config"]
    if manifest(config["baseline"], source_only=True) != original["baseline_source_manifest"]:
        raise ValueError("baseline source changed since raw recording")
    if manifest(config["fixture"]) != original["fixture_manifest"]:
        raise ValueError("fixture changed since raw recording")
    if validate_fixture(config["fixture"], config["public_fixture"]) != original["fixture_provenance"]:
        raise ValueError("fixture provenance changed")
    if initial_manifests(source) != original["initial_states_manifest"]:
        raise ValueError("raw initial states changed")
    if file_sha(source / "thresholds.json") != original["thresholds_file_sha256"]:
        raise ValueError("original thresholds changed")
    judges = judge_manifest()
    producer = {name: desc for name, desc in original["judge_manifest"].items()
                if name in ("bench/common.py", "bench/pidg_adapter.py", "bench/worker.py", "task/bench.json")
                or name.startswith("numerics/")}
    if any(judges.get(name) != desc for name, desc in producer.items()):
        raise ValueError("raw record producer changed; this correction may only change the CPU evaluator view")
    # A pre-candidate discrete exception must not disguise any observed full-
    # envelope continuous failure. LOO is replayed below and may still reject.
    for report in original_cert["baseline_self"]:
        if report["errors"] or not report["strict"]["ok"]:
            raise ValueError("original baseline has non-discrete structure/finite errors")
        for failure in report["exact_failures"]:
            changed = [k for k in failure.get("reference", {})
                       if failure["reference"][k] != failure.get("candidate", {}).get(k)]
            if failure["check"] != "invariants" or changed != ["gs_per_pixel"] or not failure["where"].endswith(" s1"):
                raise ValueError("original exact failure is outside the scoped post-update GS exception")
        if any(not RADII.fullmatch(u["tensor"]) or not u["where"].endswith(" s1") for u in report["unresolved"]):
            raise ValueError("original full-envelope numerical failure is not post-update integer radii")
    raw_dirs, projected_dirs, raw_hashes = [], [], {}
    for relative, expected in original["record_manifests"].items():
        raw = source / relative
        actual = record_manifest(raw)
        if actual != expected:
            raise ValueError(f"original raw records changed: {raw}")
        raw_dirs.append(raw)
        raw_hashes[str(raw)] = actual
        projected = out / relative
        project_records(raw, projected)
        projected_dirs.append(projected)
    if len(raw_dirs) != POLICY["baseline_records"]:
        raise ValueError("missing original baseline repetitions")
    original_exact = [strict_manifest_checks(raw_dirs[0], directory) for directory in raw_dirs[1:]]
    if not all(r["ok"] for r in original_exact):
        raise ValueError("initial GS/radii values or an exact shape/dtype/finite check cannot be retained")
    (out / "initial").mkdir()
    for file in (source / "initial").glob("*.pt"):
        (out / "initial" / file.name).symlink_to(file)
    shutil.copyfile(source / "thresholds.json", out / "thresholds.json")
    original_thresholds = read_json(out / "thresholds.json")
    independent_counts = {f"{scene}/{workload}": 4 for scene in SCENES for workload in WORKLOADS}
    supplemental_evidence = None
    if args.supplement:
        supplemental_root = Path(args.supplement).resolve()
        supplement = read_json(supplemental_root / "supplement.json")
        if supplement["status"] != "COMPLETE" or supplement["case"] != "rubber_duck/raster-flow" or len(supplement["samples"]) != 8:
            raise ValueError("require the eight declared original rubber_duck/raster-flow supplemental samples")
        if supplement["source_formulation_sha256"] != file_sha(source / "formulation.json"):
            raise ValueError("supplement uses a different raw baseline source")
        if supplement["producer_manifest"] != producer:
            raise ValueError("supplement changed the original raw producer")
        for index, sample in enumerate(supplement["samples"]):
            raw = Path(sample["records"])
            if record_manifest(raw) != sample["record_manifest"]:
                raise ValueError("supplemental raw records changed")
            if set(read_manifests(raw)) != {("rubber_duck__raster-flow", 0)}:
                raise ValueError("supplement contains duplicated or unexpected independent cases")
            raw_hashes[str(raw)] = sample["record_manifest"]
            projected = out / "supplemental-single-cases" / f"s{index}"
            project_records(raw, projected)
            complete = out / "pooled-records" / f"s{index}"
            complete_reference_view(projected_dirs[0], projected, complete)
            projected_dirs.append(complete)
        independent_counts["rubber_duck/raster-flow"] += 8
        supplemental_evidence = {"path": str(supplemental_root / "supplement.json"), "sha256": file_sha(supplemental_root / "supplement.json"),
                                 "sampling_plan": "eight fresh original-baseline processes, fixed existing initial state, two actual Adam updates",
                                 "reference_padding": "other14cases are r0 evaluation padding with zero error, explicitly NOT independent samples"}
        from numerics.judge import calibrate
        thresholds = calibrate(projected_dirs[0], projected_dirs[1:], k_fwd=POLICY["k_fwd"], k_bwd=POLICY["k_bwd"],
                               floor_fwd=POLICY["floor_fwd"], floor_bwd=POLICY["floor_bwd"])
        write_json(out / "thresholds.json", thresholds)
    else:
        thresholds = original_thresholds
    certification = certify_records(projected_dirs[0], projected_dirs[1:], thresholds, out)
    unchanged = all(record_manifest(path) == expected for path, expected in raw_hashes.items())
    unchanged = unchanged and judges == judge_manifest()
    threshold_changes = [{"key": key, "old": original_thresholds["gates"].get(key), "new": gate}
                         for key, gate in thresholds["gates"].items() if gate != original_thresholds["gates"].get(key)]
    audit = {"schema": "pidg-pre-candidate-judge-change@1", "pre_candidate_attestation": True,
             "reason": PROJECTION_REASON, "source_formulation": str(source / "formulation.json"),
             "source_formulation_sha256": file_sha(source / "formulation.json"),
             "raw_producer_evaluator_sha256": original["evaluator_sha256"],
             "new_evaluator_sha256": value_sha(judges),
             "projection_schema": PROJECTION_SCHEMA,
             "old_rule": "all steps' GS hashes exact; all integer radii sent through generic float-style gate with zero calibrated envelope",
             "new_rule": "initial GS/radii hashes exact; every shape/dtype/finite/workload check exact; only post-update discrete values diagnostic",
             "thresholds_byte_identical": file_sha(out / "thresholds.json") == original["thresholds_file_sha256"],
             "continuous_thresholds_unchanged": not threshold_changes, "threshold_changes": threshold_changes,
             "supplemental_evidence": supplemental_evidence, "independent_samples_per_case": independent_counts,
             "calibration_formula": "unchanged generic family-pooled maximum baseline envelope across cases and steps, k_fwd2/k_bwd3, floors1e-6/1e-5; no hand-set increase",
             "generic_component_unchanged": True,
             "raw_records_unchanged": unchanged, "gpu_execution": False,
             "old_certification_sha256": file_sha(source / "certification.json"),
             "old_baseline_self": [{"status": r["generic_status"], "exact_failures": len(r["exact_failures"]),
                                    "unresolved": len(r["unresolved"])} for r in original_cert["baseline_self"]],
             "new_baseline_self": [{"status": r["generic_status"], "exact_failures": len(r["exact_failures"]),
                                    "unresolved": len(r["unresolved"])} for r in certification["baseline_self"]],
             "new_leave_one_out": certification["leave_one_out"],
             "module_scope_only": True}
    write_json(out / "judge-change-audit.json", audit)
    contract = copy.deepcopy(original)
    contract.update({"status": "FROZEN" if certification["ok"] and unchanged else "BASELINE_UNCERTIFIED",
                     "frozen_unix": time.time(), "judge_manifest": judges, "evaluator_sha256": value_sha(judges),
                     "record_manifests": {str(p.relative_to(out)): record_manifest(p) for p in projected_dirs},
                     "raw_record_manifests": raw_hashes,
                     "raw_source_formulation": {"path": str(source / "formulation.json"), "sha256": audit["source_formulation_sha256"]},
                     "raw_record_producer": {"evaluator_sha256": original["evaluator_sha256"], "producer_manifest": producer,
                                             "identity": "original v1 producer; not relabeled as the new evaluator"},
                     "record_projection": {"schema": PROJECTION_SCHEMA, "reason": PROJECTION_REASON},
                     "supplemental_evidence": supplemental_evidence, "independent_samples_per_case": independent_counts,
                     "thresholds_file_sha256": file_sha(out / "thresholds.json"),
                     "judge_change_audit_sha256": file_sha(out / "judge-change-audit.json"),
                     "cpu_only_recertification": True, "raw_records_unchanged": unchanged})
    from numerics.judge import thresholds_sha
    contract["thresholds_sha256"] = thresholds_sha(thresholds)
    write_json(out / "formulation.json", contract)
    text = ("# Pre-candidate discrete-output rule correction\n\n" + PROJECTION_REASON + "\n\n"
            "No candidate had been evaluated. The original v1 formulation, certification, records and hashes remain archived unchanged. "
            "The new records are explicitly labeled CPU projections; PT storage is linked to the original bytes. "
            "The original raw producer identity is retained separately from the new evaluator/projection identity.\n\n"
            + ("All existing continuous gates and the thresholds file are byte-identical. " if not supplemental_evidence else
             "Thresholds are recomputed from eight additional original-baseline samples using the unchanged generic formula; all changes and per-case independent counts are recorded in the JSON audit. ") +
            "Initial GS indices and radii retain full exact hashes. "
            "Post-update discrete values remain raw diagnostics, while shapes, dtypes, finiteness and workload counts remain exact. "
            "No training trajectory/convergence acceptance is claimed.\n\n"
            "See judge-change-audit.json for the old/new rule, baseline evidence and regression table; certification.json contains "
            "the baseline-self, leave-one-out, generic negative and strict initial-discrete negative replays.\n")
    (out / "judge_changes.md").write_text(text)
    return {"status": contract["status"], "formulation": str(out / "formulation.json"), "scope": SCOPE,
            "gpu_execution": False, "gpu_hours": 0, "raw_records_reused": 60 + (8 if supplemental_evidence else 0)}


def run_supplement(args, out):
    if not args.source_frozen or args.samples != 8 or args.tree or args.variant:
        raise ValueError("supplement requires --source-frozen, exactly8 original-baseline samples, and no candidate selection")
    source = Path(args.source_frozen).resolve()
    original = read_json(source / "formulation.json")
    config = original["config"]
    current = judge_manifest()
    producer = {name: desc for name, desc in original["judge_manifest"].items()
                if name in ("bench/common.py", "bench/pidg_adapter.py", "bench/worker.py", "task/bench.json") or name.startswith("numerics/")}
    if any(current.get(name) != desc for name, desc in producer.items()) or original["policy"] != POLICY:
        raise ValueError("supplement must use the identical original raw record producer and policy")
    if manifest(config["baseline"], source_only=True) != original["baseline_source_manifest"] or manifest(config["fixture"]) != original["fixture_manifest"]:
        raise ValueError("baseline/data changed before supplemental sampling")
    if initial_manifests(source) != original["initial_states_manifest"]:
        raise ValueError("frozen initial state changed")
    args._fixture_sha = value_sha(original["fixture_manifest"])
    args._judge_sha = value_sha(current)
    reference_fp = original["reference_numeric_fingerprints"]["rubber_duck/raster-flow"]
    report = {"schema": "pidg-original-baseline-supplement@1", "status": "RUNNING", "case": "rubber_duck/raster-flow",
              "source_formulation_sha256": file_sha(source / "formulation.json"), "producer_manifest": producer,
              "raw_producer_evaluator_sha256": original["evaluator_sha256"], "sampling_evaluator_sha256": args._judge_sha,
              "independent_processes": 8, "steps_per_process": 2, "samples": [], "pre_candidate": True,
              "scope": "additional original-baseline evidence only; no candidate and no threshold changes during sampling"}
    write_json(out / "supplement.json", report)
    for index in range(8):
        records = out / "samples" / f"s{index}" / "records"
        records.mkdir(parents=True)
        result = worker_job(args, config, out / "samples" / f"s{index}" / "worker", "record", "rubber_duck", "raster-flow",
                            frozen=source, record_out=str(records), steps=POLICY["numeric_steps"])
        if not comparable_numeric_fingerprint(result["fingerprint"]["comparable"], reference_fp, args._judge_sha):
            raise ValueError("supplemental worker fingerprint changed outside the declared evaluator identity")
        report["samples"].append({"index": index, "records": str(records), "record_manifest": record_manifest(records),
                                  "worker_result": str(out / "samples" / f"s{index}" / "worker/result.json"),
                                  "independent": True, "fingerprint": result["fingerprint"]["comparable"]})
        write_json(out / "supplement.json", report)
    report["status"] = "COMPLETE"
    write_json(out / "supplement.json", report)
    return {"status": "COMPLETE", "supplement": str(out / "supplement.json"), "scope": SCOPE, "independent_original_samples": 8}


def suite(args, config, out, operation, frozen=None):
    scenes, workloads = args.scene or list(SCENES), args.workload or list(WORKLOADS)
    variants = args.variant or ["baseline"]
    results = []
    for variant in variants:
        if variant != "baseline" and frozen is None:
            raise ValueError("candidate diagnostics require a certified frozen original baseline")
        for scene in scenes:
            for workload in workloads:
                result = worker_job(args, config, out / variant / scene / workload, operation, scene, workload, variant,
                                    frozen=frozen, mode="forward-backward" if operation == "smoke" or args.mode == "both" else args.mode)
                results.append({"variant": variant, "scene": scene, "workload": workload, "status": result["status"]})
    return {"status": "OK", "scope": SCOPE, "results": results}


def run_time(args, config, out, frozen):
    results = []
    variants = args.variant or ["flow-only", "raster-only", "both"]
    scenes, workloads = args.scene or list(SCENES), args.workload or list(WORKLOADS)
    modes = ["forward", "forward-backward"] if args.mode == "both" else [args.mode]
    for variant in variants:
        for scene in scenes:
            for workload in workloads:
                for mode in modes:
                    case_out = out / variant / scene / workload / mode
                    arms = []
                    for arm, choice in (("baseline-before", "baseline"), ("candidate", variant), ("baseline-after", "baseline")):
                        arms.append(worker_job(args, config, case_out / arm, "time", scene, workload, choice,
                                               frozen=frozen, mode=mode))
                    report = compare_timing(*arms)
                    report.update({"variant": variant, "scene": scene, "workload": workload, "mode": mode,
                                   "protocol": schedule(args, workload), "order": "baseline-candidate-baseline; three independent processes"})
                    write_json(case_out / "comparison.json", report)
                    results.append(report)
    return {"status": "INVALID" if any(r["status"] == "INVALID" for r in results) else "EXPLORATORY",
            "scope": SCOPE, "results": results, "formal_benefit_verdict": "UNAVAILABLE_SHARED_WORKLOAD"}


def numeric_diagnostics(ref, cand):
    """Magnitude and direction diagnostics; CountSketch entries remain labeled."""
    import torch
    from numerics.judge import load_step, manifests
    refs, cands = manifests(ref), manifests(cand)
    result = []
    for key in refs.keys() & cands.keys():
        for r, c in zip(refs[key]["steps"], cands[key]["steps"]):
            rt, ct = load_step(ref, *key, r["step"]), load_step(cand, *key, c["step"])
            for name in rt.keys() & ct.keys():
                a, b = rt[name], ct[name]
                if a.shape != b.shape or not a.is_floating_point() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    continue
                a, b = a.double().flatten(), b.double().flatten()
                norm_a, norm_b = float(a.norm()), float(b.norm())
                difference = b - a
                cosine = float(torch.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else (1.0 if norm_a == norm_b else 0.0)
                result.append({"case": key[0], "rank": key[1], "step": r["step"], "tensor": name,
                               "stored": r["tensors"][name].get("stored", "full"), "cosine": cosine,
                               "stored_max_abs_difference": float(difference.abs().max()) if difference.numel() else 0.0,
                               "stored_relative_l2": float(difference.norm()) / max(norm_a, 1e-300)})
    return result


def run_check(args, config, out, frozen, contract):
    from numerics.judge import compare
    reports = []
    variants = args.variant or ["flow-only", "raster-only", "both"]
    if args.scene or args.workload or args.smoke or args.short:
        raise ValueError("check covers the entire frozen five-scene, three-workload numerical suite")
    reference = Path(frozen) / contract["reference_records"]
    thresholds = read_json(Path(frozen) / "thresholds.json")
    for variant in variants:
        record_out = out / variant / "records"
        raw_out = out / variant / "raw-records"
        raw_out.mkdir(parents=True)
        fingerprint_failures = []
        for scene in SCENES:
            for workload in WORKLOADS:
                result = worker_job(args, config, out / variant / "workers" / scene / workload, "record", scene, workload, variant,
                                    frozen=frozen, record_out=str(raw_out), steps=POLICY["numeric_steps"])
                key = f"{scene}/{workload}"
                if not comparable_numeric_fingerprint(result["fingerprint"]["comparable"],
                                                      contract["reference_numeric_fingerprints"][key],
                                                      contract["evaluator_sha256"]):
                    fingerprint_failures.append(key)
        project_records(raw_out, record_out)
        strict = strict_manifest_checks(reference, record_out)
        strict["fingerprint_failures"] = fingerprint_failures
        strict["ok"] = strict["ok"] and not fingerprint_failures
        generic = compare(reference, record_out, thresholds)
        diagnostic = numeric_diagnostics(reference, record_out)
        status = "MODULE_DIAGNOSTICS_WITHIN_FROZEN_BASELINE_ENVELOPE" if strict["ok"] and generic["status"] == "PASS" else "MODULE_DIAGNOSTICS_REJECTED_OR_UNRESOLVED"
        report = {"status": status, "variant": variant, "scope": SCOPE, "strict": strict, "generic": generic,
                  "full_training_acceptance": False,
                  "limitations": ["initial state and two diagnostic Adam updates only", "no deformation model, TV, densification or Cauchy physics",
                                  "no trajectory, convergence or final quality evidence", "no fp64 raster path; threshold excess requires review",
                                  "non-bitwise records use full hashes/norms plus 8192-dimensional CountSketch diagnostics"],
                  "thresholds_file_sha256": contract["thresholds_file_sha256"]}
        write_json(out / variant / "check.json", report)
        write_json(out / variant / "tensor-diagnostics.json", diagnostic)
        reports.append(report)
    return {"status": "EXPLORATORY_MODULE_CHECK_COMPLETE", "scope": SCOPE, "results": reports,
            "full_training_acceptance": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("inspect", "verify", "recertify", "supplement", "smoke", "profile", "time", "check", "baseline"))
    p.add_argument("--out", help="new artifact directory; must not already exist")
    p.add_argument("--frozen", help="certified baseline directory created by baseline")
    p.add_argument("--source-frozen", help="original uncertified baseline archive for CPU-only recertify")
    p.add_argument("--supplement", help="completed eight-sample original-baseline evidence for CPU recertification")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--pre-candidate", action="store_true", help="attest that no candidate has been evaluated before this audited correction")
    p.add_argument("--tree", help="candidate PIDG tree containing candidate flow file and tree-local raster build")
    p.add_argument("--baseline")
    p.add_argument("--fixture")
    p.add_argument("--public-fixture")
    p.add_argument("--hashencoder-build", help="known compatible tmp_build copied privately before unchanged upstream imports")
    p.add_argument("--variant", action="append", choices=VARIANTS)
    p.add_argument("--scene", action="append", choices=SCENES)
    p.add_argument("--workload", action="append", choices=WORKLOADS)
    p.add_argument("--mode", choices=("forward", "forward-backward", "both"), default="both")
    p.add_argument("--smoke", action="store_true", help="time only: tiny1x1 smoke, never a performance observation")
    p.add_argument("--short", action="store_true", help="triple-flow defaults1warmup+3x3; explicit short exploratory protocol")
    p.add_argument("--warmup", type=int)
    p.add_argument("--windows", type=int)
    p.add_argument("--iterations", type=int)
    return p


def main(argv=None):
    command_started = time.time()
    args = parser().parse_args(argv)
    config = configuration(args)
    if args.command == "inspect":
        print(json.dumps({"scope": SCOPE, "config": config, "policy": POLICY, "judge_sha256": value_sha(judge_manifest()),
                          "gpu_execution": False, "freeze_status": "not inspected"}, indent=2))
        return 0
    if args.command == "verify":
        if not args.frozen:
            raise ValueError("verify requires --frozen")
        contract = verify_frozen(args.frozen)
        print(json.dumps({"status": "FROZEN_HASHES_VERIFIED", "evaluator_sha256": contract["evaluator_sha256"], "gpu_execution": False}))
        return 0
    if args.command == "recertify":
        if not args.out:
            raise ValueError("recertify requires a new --out directory")
        out = new_directory(args.out)
        report = run_recertification(args, out)
        write_json(out / "summary.json", report)
        print(json.dumps(report), flush=True)
        return 0 if report["status"] == "FROZEN" else 2
    if args.command == "supplement":
        if not args.out:
            raise ValueError("supplement requires a new --out directory")
        out = new_directory(args.out)
        report = run_supplement(args, out)
        write_json(out / "summary.json", report)
        print(json.dumps(report), flush=True)
        return 0
    if not args.out:
        raise ValueError("GPU commands require --out pointing to a new directory")
    frozen, contract = None, None
    if args.frozen:
        frozen = str(Path(args.frozen).resolve())
        contract = verify_frozen(frozen)
        config = contract["config"]
    if args.command in ("time", "check") and frozen is None:
        raise ValueError(f"{args.command} requires the frozen original baseline")
    out = new_directory(args.out)
    args._judge_sha = value_sha(judge_manifest())
    args._fixture_sha = value_sha(manifest(config["fixture"]))
    validate_fixture(config["fixture"], config["public_fixture"], args.scene or SCENES)
    try:
        if args.command == "baseline":
            report = run_baseline(args, config, out)
        elif args.command in ("smoke", "profile"):
            report = suite(args, config, out, args.command, frozen)
        elif args.command == "time":
            report = run_time(args, config, out, frozen)
        else:
            report = run_check(args, config, out, frozen, contract)
        worker_costs = [read_json(path).get("cost", {}).get("gpu_hours", 0) for path in out.rglob("result.json")]
        report["cost"] = {"gpu_hours": (time.time() - command_started) / 3600, "gpu_count": 1,
                          "worker_process_gpu_hours": sum(worker_costs), "worker_processes": len(worker_costs),
                          "basis": "elapsed command hold time on one assigned shared GPU, including startup/warmup/evaluator work; no exclusive allocation implied"}
        write_json(out / "summary.json", report)
        print(json.dumps({"status": report["status"], "summary": str(out / "summary.json"), "scope": SCOPE}), flush=True)
        return 0 if report["status"] not in ("INVALID", "BASELINE_UNCERTIFIED") and all(r.get("status") != "MODULE_DIAGNOSTICS_REJECTED_OR_UNRESOLVED" for r in report.get("results", [])) else 2
    except Exception as exc:
        write_json(out / "summary.json", {"status": "ERROR", "scope": SCOPE, "error_type": type(exc).__name__, "error": str(exc),
                                           "cost": {"gpu_hours": (time.time() - command_started) / 3600, "gpu_count": 1,
                                                    "includes_startup_warmup_and_failed_workers": True}})
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
