import ast
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from timing_state import BUFFERS, PARAMETERS, FrozenTimingState, _live_tensors, _same_bits

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import numpy as np
    import torch

def cpu_fixture():
    torch.manual_seed(41); random.seed(42); np.random.seed(43)
    holder = torch.nn.Module()
    shapes = {'_xyz': (2, 3), '_features_dc': (2, 1, 3), '_features_rest': (2, 15, 3), '_scaling': (2, 3), '_rotation': (2, 4), '_opacity': (2, 1)}
    for name in PARAMETERS: holder.register_parameter(name, torch.nn.Parameter(torch.arange(np.prod(shapes[name]), dtype=torch.float32).reshape(shapes[name]) / 100))
    for name in BUFFERS: holder.register_buffer(name, torch.zeros((2,) if name == 'max_radii2D' else (2, 1)))
    gaussian = SimpleNamespace(**{name: getattr(holder, name) for name in PARAMETERS + BUFFERS}, active_sh_degree=0, max_sh_degree=3, percent_dense=.01, spatial_lr_scale=1., xyz_scheduler_args=lambda step: .01 / (step + 1))
    group_names = ['xyz', 'f_dc', 'f_rest', 'scaling', 'rotation', 'opacity']
    gaussian.optimizer = torch.optim.Adam([{'params': [getattr(holder, name)], 'name': group, 'lr': .01} for name, group in zip(PARAMETERS, group_names)], eps=1e-15)
    for parameter in holder.parameters(): parameter.grad = torch.ones_like(parameter)
    gaussian.optimizer.step(); gaussian.optimizer.zero_grad(set_to_none=True)
    return SimpleNamespace(gaussian=gaussian, holder=holder, logical_iteration=100, last_camera=None, calls={'render': 7, 'backward': 7, 'adam': 7, 'densification_stats': 7}, observed_adam_steps=7, last_gradients={})

def trajectory(adapter, cursors):
    loss_values = []
    for cursor in cursors:
        adapter.logical_iteration = 101 + cursor % 3; adapter.last_camera = 'camera' + str(cursor % 3)
        adapter.gaussian.optimizer.param_groups[0]['lr'] = adapter.gaussian.xyz_scheduler_args(adapter.logical_iteration)
        target = torch.rand(()) + random.random() + np.random.random()
        loss = sum((parameter - target).square().mean() for parameter in adapter.holder.parameters())
        loss.backward()
        with torch.no_grad():
            for buffer in adapter.holder.buffers(): buffer.add_(cursor + 1)
        adapter.gaussian.optimizer.step(); adapter.gaussian.optimizer.zero_grad(set_to_none=True)
        for key in adapter.calls: adapter.calls[key] += 1
        adapter.observed_adam_steps += 1
        loss_values.append(loss.detach().clone())
    return torch.stack(loss_values)

class RecordPathUnchanged(unittest.TestCase):
    def test_primary_prefix_identical_except_declared_viewer_registration(self):
        root = Path(__file__).resolve().parents[1]
        old = (root / 'prep/worker.before_window_state_reset.py').read_bytes()
        new = (root / 'bench/worker.py').read_bytes()
        # The public port adds only deployment-configured viewer registration.
        # This is explicitly not a claim of the historical evaluator byte hash.
        registration = b"    from runtime import configure_runtime\n    configure_runtime(adapter.c)\n"
        self.assertEqual(new.count(registration), 1)
        new = new.replace(registration, b'')
        boundary = b"    before = gpu_snapshot(adapter.c['gpu_uuid'], idle=True)\n"
        self.assertEqual(old.split(boundary)[0], new.split(boundary)[0])
        self.assertEqual(ast.dump(ast.parse(old.split(boundary)[0])), ast.dump(ast.parse(new.split(boundary)[0])))
        self.assertEqual(hashlib.sha256((root / 'bench/gs3d_adapter.py').read_bytes()).hexdigest(), '2e8181598b6250c3def932685ab0b6596c207ed5c3728ba293776fc542f162b4')
    def test_each_timed_window_restores_before_timer_and_repeats_nonzero_cursor(self):
        source = (Path(__file__).resolve().parents[1] / 'bench/worker.py').read_text()
        tree = ast.parse(source)
        window = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'window')
        self.assertEqual(window.body[0].value.func.id, 'restore_window_state')
        calls = [node for node in ast.walk(window) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'one']
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.dump(calls[0].args[0]), ast.dump(ast.parse('warmup + i', mode='eval').body))

@unittest.skipUnless(HAS_TORCH, 'run CPU tensor fixtures in prepared A/env with CUDA_VISIBLE_DEVICES empty')
class TimingStateCPUFixtures(unittest.TestCase):
    def test_restore_all_values_and_identities_repeats_trajectory_without_clearing_counters(self):
        adapter = cpu_fixture(); frozen = FrozenTimingState(adapter)
        identities = {name: (id(tensor), tensor.untyped_storage().data_ptr()) for name, tensor in _live_tensors(adapter).items()}
        group_ids = [id(group) for group in adapter.gaussian.optimizer.param_groups]
        state_ids = [id(adapter.gaussian.optimizer.state[group['params'][0]]) for group in adapter.gaussian.optimizer.param_groups]
        trajectory(adapter, range(20))  # warmup is discarded physically, not in counters
        before = dict(adapter.calls); observed = adapter.observed_adam_steps
        frozen.restore_and_verify()
        self.assertEqual(adapter.calls, before); self.assertEqual(adapter.observed_adam_steps, observed)
        first_losses = trajectory(adapter, range(20, 25))
        first_state = {name: tensor.detach().clone() for name, tensor in _live_tensors(adapter).items()}
        first_calls = dict(adapter.calls); first_observed = adapter.observed_adam_steps
        adapter.gaussian.active_sh_degree = 3; adapter.gaussian.percent_dense = .9
        receipt = frozen.restore_and_verify()
        self.assertTrue(receipt['bitwise_verified']); self.assertEqual(adapter.gaussian.active_sh_degree, 0)
        self.assertEqual(adapter.calls, first_calls); self.assertEqual(adapter.observed_adam_steps, first_observed)
        self.assertEqual(adapter.gaussian.percent_dense, .01)
        self.assertTrue(all(group['params'][0].grad is None and adapter.gaussian.optimizer.state[group['params'][0]]['step'].device.type == 'cpu' for group in adapter.gaussian.optimizer.param_groups))
        second_losses = trajectory(adapter, range(20, 25))
        self.assertTrue(_same_bits(first_losses, second_losses))
        self.assertTrue(all(_same_bits(tensor, first_state[name]) for name, tensor in _live_tensors(adapter).items()))
        self.assertEqual({name: (id(tensor), tensor.untyped_storage().data_ptr()) for name, tensor in _live_tensors(adapter).items()}, identities)
        self.assertEqual([id(group) for group in adapter.gaussian.optimizer.param_groups], group_ids)
        self.assertEqual([id(adapter.gaussian.optimizer.state[group['params'][0]]) for group in adapter.gaussian.optimizer.param_groups], state_ids)
        self.assertEqual(adapter.calls['adam'], first_calls['adam'] + 5)
    def test_replaced_optimizer_tensor_refused_even_if_values_equal(self):
        adapter = cpu_fixture(); frozen = FrozenTimingState(adapter)
        state = adapter.gaussian.optimizer.state[adapter.gaussian.optimizer.param_groups[0]['params'][0]]
        state['exp_avg'] = state['exp_avg'].clone()
        with self.assertRaisesRegex(RuntimeError, 'object/storage/layout changed'): frozen.restore_and_verify()
    def test_failed_inplace_copy_is_rejected_by_bitwise_verification(self):
        adapter = cpu_fixture(); frozen = FrozenTimingState(adapter)
        with torch.no_grad(): adapter.gaussian._xyz.add_(1)
        with patch('timing_state._copy_tensor', side_effect=lambda target, saved: None):
            with self.assertRaisesRegex(RuntimeError, 'bitwise verification failed'): frozen.restore_and_verify()

if __name__ == '__main__': unittest.main()
