from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from runtime import GPUSnapshotError, REGISTERED_VIEWERS, Telemetry, classify_processes, gpu_snapshot, wait_for_quiescence

UUID = 'GPU-00000000-0000-0000-0000-000000000000'
VIEWER = f'{UUID}, 101, native-viewer, 1199'

def sample(when, utilization, clock):
    return {'unix': when, 'utilization_percent': utilization, 'sm_clock_mhz': clock, 'memory_clock_mhz': 1500, 'power_w': 100, 'power_limit_w': 400}

class SharedViewerContract(unittest.TestCase):
    def setUp(self):
        fixture = {UUID: {'host_pid': 101, 'container_pid': 202, 'container_start_ticks': 303,
                          'baseline_memory_mib': 1199, 'provenance': 'CPU-only metadata fixture'}}
        self.registry = patch.dict('runtime.REGISTERED_VIEWERS', fixture, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)

    @patch('runtime.proc_start_ticks', return_value=303)
    def test_exact_registered_viewer_allowed_with_observations(self, ticks):
        responses = [subprocess.CompletedProcess([], 0, f'{UUID}, A800, 1199, 0, 210, 1215, 50, 400, 35\n'), subprocess.CompletedProcess([], 0, VIEWER + '\n')]
        with patch('runtime.subprocess.run', side_effect=responses):
            result = gpu_snapshot(UUID, idle=True)
        self.assertTrue(result['registered_viewer_present'])
        self.assertEqual(result['memory_mib'], '1199')
        self.assertEqual(result['parsed_compute_processes'][0]['host_pid'], 101)
        self.assertEqual(result['unknown_compute_processes'], [])
        self.assertIn('UNAVAILABLE', result['formal_benefit'])
    @patch('runtime.proc_start_ticks', return_value=303)
    def test_unknown_compute_not_silently_ignored(self, ticks):
        responses = [subprocess.CompletedProcess([], 0, f'{UUID}, A800, 1199, 0, 210, 1215, 50, 400, 35\n'), subprocess.CompletedProcess([], 0, VIEWER + f'\n{UUID}, 999999, unknown, 1\n')]
        with patch('runtime.subprocess.run', side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, 'unknown compute'):
                gpu_snapshot(UUID, idle=True)
    @patch('runtime.proc_start_ticks', return_value=303)
    def test_failure_retains_raw_and_parsed_resource_observations(self, ticks):
        responses = [subprocess.CompletedProcess([], 0, f'{UUID}, A800, 1199, 77, 1200, 1215, 300, 400, 35\n'), subprocess.CompletedProcess([], 0, VIEWER + '\n')]
        with patch('runtime.subprocess.run', side_effect=responses):
            with self.assertRaises(GPUSnapshotError) as caught:
                gpu_snapshot(UUID, idle=True)
        snapshot = caught.exception.snapshot
        self.assertEqual(snapshot['utilization_percent'], '77')
        self.assertIn('77', snapshot['raw']['gpu_query_raw'])
        self.assertEqual(snapshot['raw']['compute_query_assigned_raw'], VIEWER)
        self.assertEqual(snapshot['parsed_compute_processes'][0]['host_pid'], 101)
        self.assertIn('utilization: 77% > 5%', str(caught.exception))
    @patch('runtime.proc_start_ticks', return_value=123)
    def test_pid_reuse_refused(self, ticks):
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            classify_processes(UUID, [VIEWER])
    def test_host_pid_not_allowlisted_on_another_gpu(self):
        other = 'GPU-other'
        parsed, unknown = classify_processes(other, [f'{other}, 101, native-viewer, 1199'])
        self.assertEqual(len(unknown), 1)
        self.assertFalse(parsed[0]['registered_viewer'])

class ClockSampleContract(unittest.TestCase):
    def test_only_active_inside_samples_determine_clock(self):
        monitor = Telemetry.__new__(Telemetry)
        monitor.samples = [sample(9.9, 100, 100), sample(10.1, 0, 400), sample(10.2, 50, 1200), sample(10.8, 90, 1250), sample(11.1, 100, 100)]
        result = monitor.window(10, 11)
        self.assertEqual(result['active_clock_sample_count'], 2)
        self.assertEqual(result['clock_observation'], 'RESOLVED')
        self.assertEqual(result['sm_clock_median_mhz'], 1225)
        self.assertEqual(len(result['samples']), 5)  # energy still brackets window
    def test_boundary_activity_cannot_resolve_clock(self):
        monitor = Telemetry.__new__(Telemetry)
        monitor.samples = [sample(9.9, 100, 1200), sample(10.2, 6, 1200), sample(10.8, 5, 400), sample(11.1, 100, 1200)]
        result = monitor.window(10, 11)
        self.assertEqual(result['active_clock_sample_count'], 1)
        self.assertEqual(result['clock_observation'], 'UNRESOLVED')
        self.assertIsNone(result['sm_clock_median_mhz'])

class QuiescenceContract(unittest.TestCase):
    def test_short_post_exit_utilization_lag_waits_without_relaxing_gate(self):
        clock = [0.0]
        busy = GPUSnapshotError('utilization: 77% > 5%', {'utilization_percent': '77'})
        settled = {'utilization_percent': '0', 'unknown_compute_processes': []}
        observations = []
        def sleep(seconds): clock[0] += seconds
        with patch('runtime.time.monotonic', side_effect=lambda: clock[0]), patch('runtime.time.sleep', side_effect=sleep), patch('runtime.gpu_snapshot', side_effect=[busy, settled]) as query:
            result = wait_for_quiescence(UUID, timeout=10, interval=.5, on_observation=lambda item: observations.append(item['status']))
        self.assertEqual(result['status'], 'QUIESCENT')
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(clock[0], .5)
        self.assertTrue(all(call.kwargs['idle'] for call in query.call_args_list))
        self.assertEqual(observations, ['WAITING', 'QUIESCENT'])
    def test_persistent_unknown_compute_is_rejected_at_deadline(self):
        clock = [0.0]
        unknown = GPUSnapshotError('unknown compute process observed', {'unknown_compute_processes': [{'host_pid': 999999}]})
        observations = []
        def sleep(seconds): clock[0] += seconds
        with patch('runtime.time.monotonic', side_effect=lambda: clock[0]), patch('runtime.time.sleep', side_effect=sleep), patch('runtime.gpu_snapshot', side_effect=unknown):
            with self.assertRaises(GPUSnapshotError) as caught:
                wait_for_quiescence(UUID, timeout=1, interval=.5, on_observation=lambda item: observations.append(item['status']))
        self.assertEqual(clock[0], 1)
        self.assertEqual(caught.exception.snapshot['status'], 'REJECTED_TIMEOUT')
        self.assertEqual(len(caught.exception.snapshot['attempts']), 2)
        self.assertTrue(all(not item['accepted'] for item in caught.exception.snapshot['attempts']))
        self.assertEqual(observations[-1], 'REJECTED_TIMEOUT')

if __name__ == '__main__':
    unittest.main()
