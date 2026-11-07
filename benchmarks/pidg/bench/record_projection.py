"""CPU-only, audited view of raw records; never changes recorded tensors/math.

The original baseline's first atomic backward/Adam update changes downstream
discrete assignments across otherwise legitimate processes. Only post-update
discrete VALUE checks are diagnostic. Initial values and every shape/dtype stay
exact. Continuous tensors and their frozen numerical gates are untouched.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re

from common import file_sha, read_json, value_sha, write_json

SCHEMA = "pidg-postupdate-discrete-view@1"
RADII = re.compile(r"^out/module\.taps\.radii#\d+$")
REASON = ("Original baseline atomic-backward drift changes the first Adam update; "
          "post-update discrete GS assignments and integer radii cannot be bitwise invariants. "
          "Initial values and all shapes/dtypes remain exact; continuous gates unchanged.")


def read_manifests(root):
    result = {}
    for path in sorted(Path(root).glob("*.r*.json")):
        record = read_json(path)
        key = (record["case"]["name"], record["rank"])
        if key in result:
            raise ValueError(f"duplicate record case/rank: {key}")
        result[key] = record
    if not result:
        raise ValueError(f"no record manifests: {root}")
    return result


def projected_manifest(raw):
    result = copy.deepcopy(raw)
    if result.get("record_projection"):
        raise ValueError("refuse to project an already projected manifest")
    result["record_projection"] = {"schema": SCHEMA, "reason": REASON,
                                   "raw_manifest_sha256": value_sha(raw),
                                   "raw_producer_adapter": raw.get("adapter")}
    for entry in result["steps"]:
        if entry["step"] == 0:
            continue
        raw_gs = copy.deepcopy(entry["invariants"].get("gs_per_pixel", []))
        for desc in entry["invariants"].get("gs_per_pixel", []):
            desc.pop("sha256", None)
        radii = {name: copy.deepcopy(desc) for name, desc in entry["tensors"].items() if RADII.fullmatch(name)}
        entry["invariants"]["postupdate_discrete_shapes"] = {
            name: {"shape": desc["shape"], "dtype": desc["dtype"], "finite": desc["finite"]} for name, desc in sorted(radii.items())}
        for name in radii:
            del entry["tensors"][name]
        entry.setdefault("diagnostics", {})["postupdate_discrete"] = {
            "reason": REASON, "gs_per_pixel": raw_gs, "radii": radii,
            "raw_tensor_storage_retained": True}
    return result


def project_records(raw_root, out):
    """Create new JSON views and read-only-by-convention links to original PTs."""
    raw_root, out = Path(raw_root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = {"schema": SCHEMA, "raw_root": str(raw_root), "reason": REASON,
             "raw_manifests": {}, "tensor_storage": "symlinks to unchanged raw PT files"}
    for path in sorted(raw_root.glob("*.r*.json")):
        raw = read_json(path)
        projected = projected_manifest(raw)
        write_json(out / path.name, projected)
        audit["raw_manifests"][path.name] = file_sha(path)
    if not audit["raw_manifests"]:
        raise ValueError(f"no raw manifests: {raw_root}")
    for path in sorted(raw_root.glob("*.r*.s*.pt")):
        (out / path.name).symlink_to(path)
    write_json(out / "projection-audit.json", audit)
    return audit


def discrete_exact_failures(reference, candidate):
    failures = []
    for r, c in zip(reference["steps"], candidate["steps"]):
        where = {"case": reference["case"]["name"], "rank": reference["rank"], "step": r["step"]}
        rg, cg = r["invariants"].get("gs_per_pixel", []), c["invariants"].get("gs_per_pixel", [])
        if len(rg) != len(cg):
            failures.append({**where, "check": "discrete_render_count"})
        for index, (rd, cd) in enumerate(zip(rg, cg)):
            fields = ("shape", "dtype", "sha256") if r["step"] == 0 else ("shape", "dtype")
            for field in fields:
                if rd.get(field) != cd.get(field):
                    failures.append({**where, "check": "gs_per_pixel_" + field, "render": index})
        if r["step"] == 0:
            for name, rd in r["tensors"].items():
                if RADII.fullmatch(name) and name in c["tensors"] and rd["sha256"] != c["tensors"][name]["sha256"]:
                    failures.append({**where, "check": "initial_radii_sha256", "tensor": name})
        elif r["invariants"].get("postupdate_discrete_shapes") != c["invariants"].get("postupdate_discrete_shapes"):
            failures.append({**where, "check": "postupdate_radii_shape_or_dtype"})
    return failures


def comparable_numeric_fingerprint(actual, recorded, current_judge_sha):
    """Retain original producer identity; allow only the declared evaluator view change."""
    a, b = copy.deepcopy(actual), copy.deepcopy(recorded)
    if a.pop("judge_sha256", None) != current_judge_sha:
        return False
    b.pop("judge_sha256", None)
    return a == b


def complete_reference_view(reference, single_case, out):
    """Generic-API alignment only: reference padding is NOT independent data."""
    reference, single_case, out = Path(reference).resolve(), Path(single_case).resolve(), Path(out).resolve()
    singles = list(single_case.glob("*.r*.json"))
    if len(singles) != 1:
        raise ValueError("supplement must contain exactly one independent case")
    selected = singles[0].name
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"schema": "pidg-explicit-reference-padding@1", "independent_case": selected,
                  "padding_is_independent": False, "padding_adds_calibration_error": 0.0,
                  "purpose": "align generic complete-record API without inventing independent case samples", "cases": {}}
    for path in sorted(reference.glob("*.r*.json")):
        independent = path.name == selected
        source = single_case if independent else reference
        record = read_json(source / path.name)
        write_json(out / path.name, record)
        provenance["cases"][path.name] = {"independent": independent, "source": str(source / path.name)}
        name, rank = record["case"]["name"], record["rank"]
        for step in record["steps"]:
            tensor = source / f'{name}.r{rank}.s{step["step"]}.pt'
            (out / tensor.name).symlink_to(tensor.resolve())
    if selected not in provenance["cases"]:
        raise ValueError("supplemental case does not exist in the original reference")
    write_json(out / "complete-view-provenance.json", provenance)
    return provenance
