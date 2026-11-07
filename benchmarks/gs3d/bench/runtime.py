"""Assigned-GPU snapshots and low-overhead NVML power/clock sampling."""
import ctypes as C
import json
import os
import subprocess
import threading
import time

# Portable deployments register an owner-approved viewer in the frozen config.
# No viewer is permitted by default; no process is ever stopped by this module.
REGISTERED_VIEWERS = {}

def configure_runtime(config):
    global REGISTERED_VIEWERS
    viewer = config.get('registered_viewer')
    REGISTERED_VIEWERS = {config['gpu_uuid']: viewer} if viewer else {}

def proc_start_ticks(pid):
    with open(f'/proc/{pid}/stat') as stream:
        # comm may contain spaces or parentheses; fields after its final ')'
        # begin at field 3, so field 22 (starttime) is element 19.
        return int(stream.read().rsplit(')', 1)[1].split()[19])

def classify_processes(uuid, lines):
    registered = REGISTERED_VIEWERS.get(uuid)
    parsed, unknown = [], []
    for line in lines:
        fields = [s.strip() for s in line.split(',')]
        if len(fields) != 4 or fields[0] != uuid:
            raise RuntimeError('malformed assigned-GPU compute-process observation')
        process = {'uuid': fields[0], 'host_pid': int(fields[1]), 'process_name': fields[2], 'memory_mib': fields[3], 'registered_viewer': bool(registered and int(fields[1]) == registered['host_pid'])}
        if process['registered_viewer']:
            try:
                ticks = proc_start_ticks(registered['container_pid'])
            except (OSError, ValueError, IndexError) as error:
                raise RuntimeError('registered viewer identity cannot be verified') from error
            if ticks != registered['container_start_ticks']:
                raise RuntimeError('registered viewer container identity changed; refuse PID reuse')
            process['container_pid'] = registered['container_pid']
            process['container_start_ticks'] = ticks
        else:
            unknown.append(process)
        parsed.append(process)
    return parsed, unknown

class GPUSnapshotError(RuntimeError):
    def __init__(self, reason, snapshot):
        self.reason, self.snapshot = reason, snapshot
        super().__init__(reason + '; snapshot=' + json.dumps(snapshot, sort_keys=True))

def gpu_snapshot(uuid, idle=False, timeout=20):
    deadline = time.monotonic() + timeout
    raw = {'unix': time.time(), 'requested_uuid': uuid, 'gpu_query_raw': None, 'compute_query_assigned_raw': None}
    def query(command, label):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GPUSnapshotError('GPU snapshot query deadline exceeded', raw)
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=remaining)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            raw[label + '_error'] = {'type': type(error).__name__, 'stdout': str(getattr(error, 'stdout', None)), 'stderr': str(getattr(error, 'stderr', None)), 'command': command}
            raise GPUSnapshotError('GPU snapshot query failed: ' + label, raw) from error
        return result
    command = ['nvidia-smi', '-i', uuid, '--query-gpu=uuid,name,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu', '--format=csv,noheader,nounits']
    result = query(command, 'gpu_query')
    raw['gpu_query_raw'] = result.stdout
    row = [s.strip() for s in result.stdout.strip().split(',')]
    if len(row) != 9 or row[0] != uuid:
        raise GPUSnapshotError('assigned GPU query identity/shape failed', raw)
    processes = query(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], 'compute_query')
    own = [line for line in processes.stdout.splitlines() if line.split(',')[0].strip() == uuid]
    raw['compute_query_assigned_raw'] = '\n'.join(own)
    try:
        parsed, unknown = classify_processes(uuid, own)
    except (RuntimeError, ValueError) as error:
        raw['classification_error'] = str(error)
        raise GPUSnapshotError('assigned GPU process identity/classification failed', raw) from error
    viewer_present = any(p['registered_viewer'] for p in parsed)
    memory_budget = (REGISTERED_VIEWERS[uuid]['baseline_memory_mib'] if viewer_present else 0) + 100
    memory = {}
    for label, name in [('available_bytes', 'SC_AVPHYS_PAGES'), ('total_bytes', 'SC_PHYS_PAGES')]:
        try: memory[label] = os.sysconf(name) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError): memory[label] = None
    pressure = None
    try:
        with open('/proc/pressure/memory') as f: pressure = f.read().strip()
    except OSError:
        pass
    snapshot = {'unix': time.time(), 'uuid': row[0], 'name': row[1], 'memory_mib': row[2], 'utilization_percent': row[3], 'sm_clock_mhz': row[4], 'memory_clock_mhz': row[5], 'power_w': row[6], 'power_limit_w': row[7], 'temperature_c': row[8], 'processes': own, 'parsed_compute_processes': parsed, 'unknown_compute_processes': unknown, 'registered_viewer': REGISTERED_VIEWERS.get(uuid), 'registered_viewer_present': viewer_present, 'idle_residency_budget_mib': memory_budget, 'formal_benefit': 'UNAVAILABLE_shared_machine_with_preserved_viewer', 'host_load': list(os.getloadavg()), 'host_memory': memory, 'host_memory_pressure': pressure, 'raw': raw}
    failures = []
    if unknown: failures.append('unknown compute process observed')
    try:
        if float(row[2]) > memory_budget: failures.append(f'excess residency: {row[2]} MiB > {memory_budget} MiB')
        if float(row[3]) > 5: failures.append(f'utilization: {row[3]}% > 5%')
    except ValueError as error:
        raise GPUSnapshotError('GPU memory/utilization observation is not numeric', snapshot) from error
    snapshot['idle_check_failures'] = failures
    if idle and failures:
        raise GPUSnapshotError('assigned GPU is not idle apart from its registered viewer: ' + '; '.join(failures), snapshot)
    return snapshot

def wait_for_quiescence(uuid, timeout=10, interval=.5, on_observation=None):
    """Post-child settling only; identical identity/memory/utilization gates.

    Every failed/successful observation is retained. The deadline covers SMI
    queries as well as waits, so an unknown or busy workload cannot hang the
    bench or be silently accepted after a child exits.
    """
    if timeout <= 0 or interval <= 0:
        raise ValueError('quiescence timeout and interval must be positive')
    started = time.monotonic()
    deadline = started + timeout
    attempts = []
    evidence = {'status': 'WAITING', 'timeout_seconds': timeout, 'interval_seconds': interval, 'thresholds_unchanged': True, 'attempts': attempts}
    while True:
        attempt_start = time.monotonic()
        remaining = deadline - attempt_start
        if remaining <= 0:
            evidence['status'] = 'REJECTED_TIMEOUT'
            evidence['elapsed_seconds'] = time.monotonic() - started
            if on_observation:
                on_observation(evidence)
            raise GPUSnapshotError('post-child GPU quiescence deadline exceeded; original gates unchanged', evidence)
        try:
            snapshot = gpu_snapshot(uuid, idle=True, timeout=remaining)
            reason, accepted = None, True
        except GPUSnapshotError as error:
            snapshot, reason, accepted = error.snapshot, error.reason, False
        elapsed = time.monotonic() - started
        failed = not accepted and time.monotonic() >= deadline
        attempts.append({'index': len(attempts), 'unix': time.time(), 'elapsed_seconds': elapsed, 'accepted': accepted, 'reason': reason, 'snapshot': snapshot})
        evidence = {'status': 'QUIESCENT' if accepted else ('REJECTED_TIMEOUT' if failed else 'WAITING'), 'timeout_seconds': timeout, 'interval_seconds': interval, 'elapsed_seconds': elapsed, 'thresholds_unchanged': True, 'attempts': attempts, 'snapshot': snapshot}
        if on_observation:
            on_observation(evidence)
        if accepted:
            return evidence
        if failed:
            raise GPUSnapshotError('post-child GPU did not quiesce within the bounded wait; original gates unchanged', evidence)
        delay = min(max(0, interval - (time.monotonic() - attempt_start)), max(0, deadline - time.monotonic()))
        if delay:
            time.sleep(delay)

class Telemetry:
    def __init__(self, uuid):
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self._check(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.handle)))
        self.samples = []
        self.stop_event = threading.Event()
    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f'NVML call failed: {code}')
    def uint(self, name, *args):
        value = C.c_uint()
        self._check(getattr(self.lib, name)(self.handle, *args, C.byref(value)))
        return value.value
    def sample(self):
        class Utilization(C.Structure):
            _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]
        utilization = Utilization()
        self._check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, C.byref(utilization)))
        return {'unix': time.time(), 'utilization_percent': utilization.gpu, 'memory_utilization_percent': utilization.memory, 'power_w': self.uint('nvmlDeviceGetPowerUsage') / 1000, 'power_limit_w': self.uint('nvmlDeviceGetPowerManagementLimit') / 1000, 'sm_clock_mhz': self.uint('nvmlDeviceGetClockInfo', C.c_uint(1)), 'memory_clock_mhz': self.uint('nvmlDeviceGetClockInfo', C.c_uint(2)), 'temperature_c': self.uint('nvmlDeviceGetTemperature', C.c_uint(0))}
    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(self.sample())
            except BaseException as error:
                self.error = repr(error)
                return
            self.stop_event.wait(.2)
    def start(self):
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.lib.nvmlShutdown()
        if self.error:
            raise RuntimeError(self.error)
    def window(self, start, end):
        import statistics
        ordered = sorted(self.samples, key=lambda s: s['unix'])
        inside = [s for s in ordered if start <= s['unix'] <= end]
        samples = list(inside)
        before = [s for s in ordered if s['unix'] < start]
        after = [s for s in ordered if s['unix'] > end]
        if before: samples.insert(0, before[-1])
        if after: samples.append(after[0])
        if not samples:
            # A sub-200ms smoke may have no internal sample. Standard 175-step
            # windows must retain real samples and refuse unavailable telemetry.
            return {'samples': [], 'available': False, 'clock_observation': 'UNRESOLVED', 'active_clock_sample_count': 0, 'sm_clock_median_mhz': None, 'memory_clock_median_mhz': None}
        energy = sum((b['unix'] - a['unix']) * (a['power_w'] + b['power_w']) / 2 for a, b in zip(samples, samples[1:]))
        active = [s for s in inside if s['utilization_percent'] > 5]
        enough = len(active) >= 2
        return {'available': True, 'samples': samples, 'includes_nearest_boundary_samples': True, 'clock_sample_policy': 'only samples inside this window with GPU utilization >5%; at least two required', 'active_clock_sample_count': len(active), 'clock_observation': 'RESOLVED' if enough else 'UNRESOLVED', 'sm_clock_median_mhz': statistics.median(s['sm_clock_mhz'] for s in active) if enough else None, 'memory_clock_median_mhz': statistics.median(s['memory_clock_mhz'] for s in active) if enough else None, 'mean_power_w': statistics.mean(s['power_w'] for s in samples), 'sampled_energy_j': energy, 'sampled_duration_s': samples[-1]['unix'] - samples[0]['unix'], 'estimated_window_energy_j': statistics.mean(s['power_w'] for s in samples) * (end - start), 'at_power_cap_fraction': sum(s['power_w'] >= .98 * s['power_limit_w'] for s in samples) / len(samples)}
