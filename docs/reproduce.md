# Reproduction commands

Run from the repository root in a compatible environment described in
[environment.md](environment.md). Use new output directories for every run.
Commands below execute actual upstream modules/native training, never substitute
generated geometry, a smaller image size or an approximate loss.

## PIDG module experiment

```bash
python scripts/prepare.py pidg
python scripts/fetch_pidg.py --output "$PWD/workspace/data/pidg"
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python scripts/build.py pidg
python scripts/configure.py pidg --gpu YOUR_ALLOCATED_GPU --data "$PWD/workspace/data/pidg"

H="$PWD/workspace/benchmarks/pidg"
B="$PWD/workspace/pidg/baseline"
C="$PWD/workspace/pidg/candidate"
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python "$H/bench/bench.py" baseline \
  --out "$PWD/workspace/pidg-reference" --hashencoder-build "$B/tmp_build"
```

Stop if reference self-controls/negative controls fail. Do not widen a gate after
seeing candidate results. Historical PIDG calibration used14cases×4reference
records plus12for rubber_duck/raster-flow (68total); a fresh baseline has its own
recorded acceptance and may require a separately audited reference-only
formulation. No historical reference tensors or numerical certification are
silently reused.

With a successfully frozen new baseline:

```bash
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python "$H/bench/bench.py" check \
  --tree "$C" --frozen "$PWD/workspace/pidg-reference" \
  --out "$PWD/workspace/pidg-check" --hashencoder-build "$B/tmp_build"

# Five-scene primary flow-only group:1warmup +3windows×1iteration.
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python "$H/bench/bench.py" time \
  --tree "$C" --frozen "$PWD/workspace/pidg-reference" --variant flow-only \
  --workload triple-flow --mode forward-backward --short --warmup 1 --windows 3 --iterations 1 \
  --out "$PWD/workspace/pidg-flow-time" --hashencoder-build "$B/tmp_build"

# Longer retained dry_ice check:1warmup +3windows×3iterations.
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python "$H/bench/bench.py" time \
  --tree "$C" --frozen "$PWD/workspace/pidg-reference" --variant flow-only \
  --scene dry_ice --workload triple-flow --mode forward-backward --short \
  --warmup 1 --windows 3 --iterations 3 \
  --out "$PWD/workspace/pidg-flow-confirm" --hashencoder-build "$B/tmp_build"
```

The group retains all five PNG writes. The2-frame download is a module fixture,
not a complete training dataset. Rubber's default training disables flow;
clock-insufficient and noise-unresolved rows remain so regardless of large
ratios. Default candidate contains no rejected raster change. To study the
rejected candidate, prepare a **different** workspace with `--include-rejected`,
build with the same flag, then point `--tree` at `pidg/rejected-both`. `raster-only`
selects baseline flow plus that raster; `both` selects both candidates. Preserve
the real numerical rejection and do not adopt that tree as the retained model.

## Original3DGS full native training and quality

```bash
python scripts/prepare.py gs3d
python scripts/fetch_mip360.py --output "$PWD/workspace/data/mip360"
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python scripts/build.py gs3d
python scripts/prepare_sfm.py --data "$PWD/workspace/data/mip360"
python scripts/configure.py gs3d --gpu YOUR_ALLOCATED_GPU --data "$PWD/workspace/data/mip360"
# On an owner-authorized shared card only, configure additionally with:
# --viewer HOST_PID:CONTAINER_PID:BASELINE_GPU_MEMORY_MIB
# This records the process start identity before freeze; it never stops it.

H="$PWD/workspace/benchmarks/gs3d"
python "$H/bench/freeze_data.py" --config "$H/task/config.json"
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python scripts/run_native.py \
  --tree baseline --out "$PWD/workspace/native-baseline"

# Freeze the baseline-only two-seed quality yardstick BEFORE candidate training.
python "$H/bench/quality_gate.py" freeze \
  --inputs "$PWD/workspace/native-baseline/quality-inputs.json" \
  --out "$PWD/workspace/quality-reference"
python scripts/gpu.py --gpu YOUR_ALLOCATED_GPU -- python scripts/run_native.py \
  --tree candidate --out "$PWD/workspace/native-candidate"
python "$H/bench/quality_gate.py" compare \
  --inputs "$PWD/workspace/native-candidate/quality-inputs.json" \
  --frozen "$PWD/workspace/quality-reference/quality_frozen.json" \
  --out "$PWD/workspace/quality-comparison"
```

Both trees run unmodified `train.py`, native30000steps, all scheduled
densification/pruning/reset/SH/LR/Adam events, eval/saves at7k/30k and checkpoints
100/7k/29900/30k. The wrapper applies matched Python/NumPy/Torch seeds and a
separate fresh-CPU artifact guard. All saved fields must be finite; finalPLY
matches the final capture bitwise. Quality uses all37/25/24heldout views and the
original metric functions; no LPIPS weights are downloaded. `run_native.py`
accepts `--scene` and `--seed` for bounded individual runs.

Read full native durations from existing artifacts using the predeclared formula:

```text
native_complete.postcheck_cpu_process.process.started_unix
    - full_train.process.started_unix
```

Both paired runs must have exit0/statuscomplete, the same config/data/seed/native
schedule and a successful fresh CPU guard. Do not substitute progress-bar time,
recovered exits or wrapper elapsed time that includes the added guard. The
historical table has only seed1 eligible; fresh successful runs can expose both
seeds, with their provenance retained.

## Static fixed-phase harness

The checked-in `benchmarks/gs3d/bench/` also provides `make_cases.py`,
`bench.py baseline/record/time/profile/check`, `freeze_numerics.py`,
`freeze_task.py` and the unchanged generic numerics. Use completed **reference**
checkpoints to formulate all9cases (three real training cameras per scene at
100/7k/29900), collect independent reference records and freeze the new evaluator
and gates before any candidate check. `bench.py --help` and the preparation CLIs
give their required paths. Every GPU command must use the same one-GPU launcher.

Standard timing is B-C-B,20warmup,3×175steps with exact state restore outside
the clock; the native training cases are separate. Profiles for both `record`
and `timing` paths identify the actual `renderCUDAFullWarp` symbol and exact
loaded source/binary, rather than relying on compiler counters. The historical
unchanged-reference controls and the candidate both failed continuous numerical
gates. This repository does not provide a command that converts those failures
into PASS. Full-training/quality exploration is explicitly labeled as such.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s benchmarks/pidg/tests -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s benchmarks/gs3d/tests -v
```

Publication validation checks portable configuration, gradient semantics,
artifact guards and source patch application against the pinned upstream code.
No new GPU timing is claimed from publication-only checks. Rebuilt binary hashes,
fresh numerical acceptance and speed observations belong to the new deployment.
