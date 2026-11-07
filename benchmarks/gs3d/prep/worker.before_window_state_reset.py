"""Only frozen GPU work paths: record, steady timing, or profiler."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import assert_data, assert_source_contract, digest, sha256, write_json
from gs3d_adapter import GS3DAdapter
from runtime import Telemetry, gpu_snapshot

def compile_diagnostics(torch):
    # Counters are diagnostic, never proof that compiled kernels actually ran.
    # Kernel traces for both the untraced record and timing paths remain required.
    try:
        from torch._dynamo.utils import counters
        return {str(name): {str(key): int(value) for key, value in values.items()} for name, values in counters.items() if values}
    except (ImportError, AttributeError):
        return {'availability': 'unavailable'}

def profile_kernel_inventory(path):
    from collections import Counter
    trace = json.loads(Path(path).read_text())
    events = trace.get('traceEvents', trace) if isinstance(trace, dict) else trace
    return dict(Counter(event['name'] for event in events if event.get('ph') == 'X' and event.get('cat') == 'kernel'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['record', 'record-dtype', 'time', 'profile'])
    p.add_argument('--config', required=True)
    p.add_argument('--tree', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['forward', 'forward-backward', 'step'], default='step')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--detailed', action='store_true')
    p.add_argument('--lr-control', action='store_true', help='formulation-only real LR x1.02 negative control')
    p.add_argument('--profile-path', choices=['timing', 'record'], default='timing')
    a = p.parse_args()
    out = Path(a.out)
    case = next(c for c in json.loads(Path(a.cases).read_text()) if c['name'] == a.case)
    import torch
    from numerics.record import Recorder, math_mode, seed_everything
    seed_everything(case['seed'])
    if a.lr_control and a.action != 'record':
        raise ValueError('LR control is for formulation recordings only')
    primary_record_path = a.action in ['record', 'record-dtype'] or (a.action == 'profile' and a.profile_path == 'record')
    adapter = GS3DAdapter(a.config, a.tree, out / 'scene_scratch', timing=not primary_record_path, lr_control=a.lr_control)
    adapter.configure(case)
    # Leave all native precision and determinism switches unchanged. The
    # original custom raster backward uses atomics even in deterministic mode.
    if a.action in ['record', 'record-dtype']:
        dtype_trace = a.action == 'record-dtype'
        Recorder(adapter, out / 'records', sketch_dim=8192, dtype_trace=dtype_trace).run(case)
        write_json(out / 'record_complete.json', {'case': case['name'], 'binding': adapter.binding, 'source': adapter.source, 'data_sha256': adapter.data_sha, 'optimizer_state_device_inventory': adapter.optimizer_state_device_inventory, 'dtype_trace': dtype_trace, 'record_role': 'dtype supplement only; NEVER primary numerics or timing evidence' if dtype_trace else 'primary actual path, explicit dtype_trace=False', 'fresh_process': True, 'compile_diagnostics': compile_diagnostics(torch), 'actual_fast_path_profile_proof': 'UNVERIFIED; compiler counters or dtype trace cannot substitute for record-path and timing-path kernel traces', 'native_atomic_backward': True, 'fp64_arbitration': 'unavailable'})
        return
    if a.action == 'profile' and a.profile_path == 'record':
        # A separate fresh process profiles the same untraced Recorder path.
        # Its values are supplemental; primary records never run under a
        # TorchDispatchMode or inherit a previously dtype-traced process.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=a.detailed, with_stack=a.detailed, profile_memory=a.detailed) as prof:
            Recorder(adapter, out / 'profile_records', sketch_dim=8192, dtype_trace=False).run(case)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out / 'trace.json'))
        write_json(out / 'profile_complete.json', {'case': case['name'], 'case_spec': case, 'camera_tensor_sha256': adapter.camera_hashes, 'math_mode': math_mode(), 'mode': 'step', 'profile_path': 'record', 'dtype_trace': False, 'steps': case['steps'], 'detailed': a.detailed, 'binding': adapter.binding, 'source': adapter.source, 'data_sha256': adapter.data_sha, 'optimizer_state_device_inventory': adapter.optimizer_state_device_inventory, 'trace_sha256': sha256(out / 'trace.json'), 'kernel_inventory': profile_kernel_inventory(out / 'trace.json'), 'compile_diagnostics': compile_diagnostics(torch), 'fast_path_verification': 'REVIEW_REQUIRED: identify actual candidate fast-path kernels here and in timing-path trace; counters alone are insufficient', 'not_for_speed_comparison': True})
        return
    before = gpu_snapshot(adapter.c['gpu_uuid'], idle=True)
    startup = time.time()
    adapter.build(case['seed'])
    torch.cuda.synchronize()
    setup_duration = time.time() - startup
    warmup, steps, windows = (1, 1, 1) if a.smoke else (20, 175, 3)
    def one(step):
        adapter.set_phase(case['phase'], step)
        return adapter.run(step, adapter.batch(step, 0), mode=a.mode)
    monitor = Telemetry(adapter.c['gpu_uuid'])
    monitor.start()
    records = []
    warmup_start = time.time()
    try:
        for step in range(warmup):
            losses = one(step)
        torch.cuda.synchronize()
        warmup_duration = time.time() - warmup_start
        if a.action == 'profile':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=a.detailed, with_stack=a.detailed, profile_memory=a.detailed) as prof:
                for i in range(3):
                    one(warmup + i)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(out / 'trace.json'))
            write_json(out / 'profile_complete.json', {'case': case['name'], 'case_spec': case, 'camera_tensor_sha256': adapter.camera_hashes, 'math_mode': math_mode(), 'mode': a.mode, 'profile_path': 'timing', 'dtype_trace': False, 'detailed': a.detailed, 'steps': 3, 'binding': adapter.binding, 'source': adapter.source, 'data_sha256': adapter.data_sha, 'optimizer_state_device_inventory': adapter.optimizer_state_device_inventory, 'trace_sha256': sha256(out / 'trace.json'), 'kernel_inventory': profile_kernel_inventory(out / 'trace.json'), 'compile_diagnostics': compile_diagnostics(torch), 'fast_path_verification': 'REVIEW_REQUIRED: identify actual candidate fast-path kernels here and in untraced record-path trace; counters alone are insufficient', 'not_for_speed_comparison': True})
            return
        for window in range(windows):
            torch.cuda.synchronize()
            monitor.samples.append(monitor.sample())
            counts_before = dict(adapter.calls)
            adam_before = adapter.observed_adam_steps
            window_losses = []
            start = time.time()
            timer = time.perf_counter()
            for i in range(steps):
                losses = one(warmup + window * steps + i)
                window_losses.extend(value.detach() for value in losses.values())
            torch.cuda.synchronize()
            elapsed, end = time.perf_counter() - timer, time.time()
            monitor.samples.append(monitor.sample())
            state_tensors = list(adapter.holder.parameters()) + list(adapter.holder.buffers()) + [value for state in adapter.gaussian.optimizer.state.values() for value in state.values() if torch.is_tensor(value)]
            finite_losses = bool(torch.isfinite(torch.stack(window_losses)).all().item())
            finite_state = all(torch.isfinite(value).all().item() for value in state_tensors)
            finite_gradients = (len(adapter.last_gradients) == 6 and all(value is not None and torch.isfinite(value).all().item() for value in adapter.last_gradients.values())) if a.mode == 'forward-backward' else None
            window_finite = finite_losses and finite_state and finite_gradients is not False
            records.append({'index': window, 'start_unix': start, 'end_unix': end, 'seconds': elapsed, 'ms_per_step': elapsed * 1000 / steps, 'steps': steps, 'calls': {k: adapter.calls[k] - counts_before[k] for k in adapter.calls}, 'observed_adam_steps': adapter.observed_adam_steps - adam_before, 'finite': window_finite, 'finite_all_window_losses': finite_losses, 'finite_parameter_buffer_adam_state': finite_state, 'finite_last_gradients': finite_gradients, 'last_gradient_count': len(adapter.last_gradients)})
            if not window_finite:
                break
    finally:
        monitor.stop()
    for row in records:
        row['telemetry'] = monitor.window(row['start_unix'], row['end_unix'])
        expected = {'render': steps, 'backward': 0 if a.mode == 'forward' else steps, 'adam': steps if a.mode == 'step' else 0, 'densification_stats': steps if a.mode == 'step' and case['iteration'] < 15000 else 0}
        if row['calls'] != expected:
            raise ValueError('timed workload call counts changed')
        if row['observed_adam_steps'] != expected['adam']:
            raise ValueError('actual optimizer hook count disagrees with timed workload')
    finite = all(row['finite'] for row in records)
    med = statistics.median(row['ms_per_step'] for row in records)
    noise = (max(row['ms_per_step'] for row in records) - min(row['ms_per_step'] for row in records)) / med
    valid = finite and len(records) == windows and noise <= .10 and (a.smoke or all(row['telemetry']['available'] for row in records))
    if adapter.source != assert_source_contract(a.tree, adapter.c['reference']) or adapter.data_sha != assert_data(adapter.c, case['scene']):
        raise ValueError('source/data changed during timed worker')
    fingerprint = {'case': case, 'camera_tensor_sha256': adapter.camera_hashes, 'parameter_inventory': {name: {'shape': list(t.shape), 'dtype': str(t.dtype)} for name, t in adapter.holder.named_parameters()}, 'optimizer_state_device_inventory': adapter.optimizer_state_device_inventory, 'config_sha256': digest(adapter.c), 'gpu_uuid': adapter.c['gpu_uuid'], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0), 'math': math_mode(), 'mode': a.mode, 'warmup': warmup, 'steps_per_window': steps, 'windows': windows, 'cold_cache': True, 'profiler': False, 'dtype_trace': False, 'detailed': False, 'boundary_synchronization_only': True, 'finite_loss_retention': 'detached scalars for every timed loss; checked at window boundary', 'finite_grad_retention': 'last gradient references, released before next render' if a.mode == 'forward-backward' else 'generic record hooks / Adam-state boundary checks', 'actual_adam_hook_counter': True, 'timing_contract': 'fixed-N three-camera three-position phase probe; native densification events excluded', 'smoke': a.smoke}
    write_json(out / 'timing.json', {'status': 'VALID' if valid else 'INVALID', 'windows': records, 'median_ms': med, 'relative_window_range': noise, 'finite': finite, 'fingerprint': fingerprint, 'binding': adapter.binding, 'source': adapter.source, 'data_sha256': adapter.data_sha, 'compile_diagnostics': compile_diagnostics(torch), 'actual_fast_path_profile_proof': 'UNVERIFIED; require actual untraced record-path and timing-path kernel evidence before compile acceptance', 'gpu_before': before, 'setup_seconds': setup_duration, 'warmup_seconds': warmup_duration, 'cost': {'gpu_hours': (time.time() - startup) / 3600}})
    if not valid:
        raise RuntimeError('INVALID timing; inspect each window and telemetry')

if __name__ == '__main__':
    main()
