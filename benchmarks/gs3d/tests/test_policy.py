import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from policy import compare_timing_runs, scan_error_text

def arm(milliseconds):
    return {'status': 'VALID', 'fingerprint': {'smoke': False, 'case': 'fixed'}, 'median_ms': milliseconds, 'relative_window_range': 0, 'windows': [{'ms_per_step': milliseconds, 'telemetry': {'sm_clock_median_mhz': 1200, 'clock_observation': 'RESOLVED', 'active_clock_sample_count': 2}} for _ in range(3)]}

class FrozenTimingPolicy(unittest.TestCase):
    def test_exploratory_never_formal_pass(self):
        result = compare_timing_runs(arm(10), arm(5), arm(10))
        self.assertEqual(result['status'], 'UNAVAILABLE')
        self.assertTrue(result['exploratory_resolved'])
    def test_any_fingerprint_change_refused(self):
        candidate = arm(5)
        candidate['fingerprint']['case'] = 'different'
        self.assertEqual(compare_timing_runs(arm(10), candidate, arm(10))['status'], 'ERROR')
    def test_reference_window_contamination_refused(self):
        baseline = arm(10)
        baseline['windows'][0]['ms_per_step'] = 20
        self.assertEqual(compare_timing_runs(baseline, arm(5), arm(10))['status'], 'INVALID')
    def test_clock_difference_refused(self):
        candidate = arm(5)
        candidate['windows'][0]['telemetry']['sm_clock_median_mhz'] = 900
        self.assertEqual(compare_timing_runs(arm(10), candidate, arm(10))['status'], 'INVALID')
    def test_smoke_cannot_resolve_benefit(self):
        reports = [arm(10), arm(5), arm(10)]
        for report in reports: report['fingerprint']['smoke'] = True
        self.assertEqual(compare_timing_runs(*reports)['reason'], 'smoke is not a speed measurement')
    def test_missing_active_clocks_explicitly_unresolved(self):
        candidate = arm(5)
        candidate['windows'][0]['telemetry'] = {'clock_observation': 'UNRESOLVED', 'active_clock_sample_count': 1, 'sm_clock_median_mhz': None}
        result = compare_timing_runs(arm(10), candidate, arm(10))
        self.assertEqual(result['status'], 'UNAVAILABLE')
        self.assertEqual(result['observation'], 'UNRESOLVED')
        self.assertFalse(result['exploratory_resolved'])
        self.assertIsNone(result['clock_spread'])
    def test_original_error_signatures_cannot_pass_silently(self):
        log = 'CUDA error: an illegal memory access was encountered\nCUDA out of memory\nloss=NaN\nFalling back to eager execution\n'
        self.assertEqual({hit['signature'] for hit in scan_error_text(log)}, {'illegal_memory_access', 'out_of_memory', 'nonfinite', 'eager_fallback'})
    def test_original_eager_path_is_not_a_fallback(self):
        self.assertEqual(scan_error_text('Original eager renderer active; 175 steps completed'), [])

if __name__ == '__main__':
    unittest.main()
