import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gpu import environment

try:
    import torch
    from flow_gather import _flow_gather
except ImportError:
    torch = None


class ReleaseContract(unittest.TestCase):
    def test_generic_numerics_match_pinned_original_bytes(self):
        origin = json.loads((ROOT / 'results/researchboost-numerics-origin.json').read_text())
        for package in ['pidg', 'gs3d']:
            for name, sha in origin['files'].items():
                self.assertEqual(hashlib.sha256((ROOT / 'benchmarks' / package / 'numerics' / name).read_bytes()).hexdigest(), sha)

    def test_one_gpu_environment_preserves_unrelated_variables(self):
        original = {'HOME': '/unchanged-home', 'CUSTOM': 'retained', 'CUDA_VISIBLE_DEVICES': '0,1'}
        result = environment('GPU-example', '8.0', original)
        self.assertEqual(original['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], 'GPU-example')
        self.assertEqual(result['HOME'], original['HOME'])
        self.assertEqual(result['CUSTOM'], 'retained')
        self.assertEqual(result['TORCH_CUDA_ARCH_LIST'], '8.0')

    def test_data_byte_pins_cover_actual_scope(self):
        pidg = json.loads((ROOT / 'results/pidg/data-manifest.json').read_text())
        static = json.loads((ROOT / 'results/gs3d/data-manifest.json').read_text())
        self.assertEqual(len(pidg['files']), 40)
        self.assertEqual(len(static['files']), 683)
        for row in pidg['files'] + static['files']:
            self.assertEqual(len(row['sha256']), 64)
            self.assertFalse(Path(row['path']).is_absolute())
            self.assertNotIn('..', Path(row['path']).parts)

    def test_failed_verdict_remains_failed(self):
        result = json.loads((ROOT / 'results/gs3d/numerics.json').read_text())
        self.assertEqual(result['status'], 'INCORRECT')
        self.assertEqual(len(result['violations']), 11)
        self.assertFalse(json.loads((ROOT / 'results/gs3d/summary.json').read_text())['formal_ResearchBoost_acceptance'])


@unittest.skipIf(torch is None, 'PyTorch required for real CPU autograd tests')
class GatherSemantics(unittest.TestCase):
    def compare(self, incoming):
        table = torch.arange(10, dtype=torch.float64).reshape(5, 2).requires_grad_()
        index = torch.tensor([0, -1, 0, 1])
        expected = torch.autograd.grad(table[index], table, grad_outputs=incoming)[0]
        actual = torch.autograd.grad(_flow_gather(table, index), table, grad_outputs=incoming)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_zero_and_negative_index_accumulation(self):
        self.compare(torch.tensor([[1., 2.], [0., 0.], [-3., 4.], [0., 0.]], dtype=torch.float64))

    def test_nonfinite_addends_are_not_elided(self):
        self.compare(torch.tensor([[float('nan'), 0.], [0., 0.], [2., 3.], [0., float('inf')]], dtype=torch.float64))

    def test_second_derivative_at_zero(self):
        x = torch.tensor([[0., 0.], [1., 2.], [3., 4.]], dtype=torch.float64, requires_grad=True)
        index = torch.tensor([0, 0, 2])
        first = torch.autograd.grad((_flow_gather(x, index) ** 2).sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        torch.testing.assert_close(second, torch.tensor([[4., 4.], [0., 0.], [2., 2.]], dtype=torch.float64), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
