"""Timing-only physical-state reset; never imported on a primary record path."""
import copy
import hashlib
import pickle
import random

PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
BUFFERS = ('max_radii2D', 'xyz_gradient_accum', 'denom')
GAUSSIAN_SCALARS = ('active_sh_degree', 'max_sh_degree', 'percent_dense', 'spatial_lr_scale')
CURSOR_SCALARS = ('logical_iteration', 'last_camera')

def _live_tensors(adapter):
    import torch
    gaussian, holder, optimizer = adapter.gaussian, adapter.holder, adapter.gaussian.optimizer
    parameters, buffers = dict(holder.named_parameters()), dict(holder.named_buffers())
    if set(parameters) != set(PARAMETERS) or set(buffers) != set(BUFFERS):
        raise RuntimeError('timing snapshot requires the original six parameters and three buffers')
    result = {}
    for name in PARAMETERS + BUFFERS:
        tensor = parameters[name] if name in parameters else buffers[name]
        if getattr(gaussian, name) is not tensor:
            raise RuntimeError('GaussianModel/holder tensor alias changed: ' + name)
        if name in parameters and tensor.grad is not None:
            raise RuntimeError('timing boundary has uncleared parameter gradients: ' + name)
        result[('param/' if name in parameters else 'buffer/') + name] = tensor
    seen = set()
    for group in optimizer.param_groups:
        if len(group['params']) != 1 or not any(group['params'][0] is value for value in parameters.values()):
            raise RuntimeError('timing optimizer parameter binding changed')
        name, parameter = group['name'], group['params'][0]
        if name in seen: raise RuntimeError('duplicate optimizer group name')
        seen.add(name)
        for key, value in optimizer.state[parameter].items():
            if torch.is_tensor(value): result[f'optim/{name}/{key}'] = value
        step = optimizer.state[parameter].get('step')
        if not torch.is_tensor(step) or step.device.type != 'cpu':
            raise RuntimeError('native noncapturable Adam step must stay CPU')
    if len(seen) != 6: raise RuntimeError('six native optimizer groups required')
    return result

def _layout(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device, tensor.requires_grad, tensor.untyped_storage().data_ptr(), tensor.storage_offset())

def _copy_tensor(target, frozen):
    target.copy_(frozen)

def _same_bits(first, second):
    import torch
    return torch.equal(first.detach().contiguous().reshape(-1).view(torch.uint8), second.detach().contiguous().reshape(-1).view(torch.uint8))

def _rng_state():
    import numpy as np
    import torch
    return {'python': copy.deepcopy(random.getstate()), 'numpy': copy.deepcopy(np.random.get_state()), 'torch': torch.get_rng_state().clone(), 'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}

def _rng_digest(state):
    checksum = hashlib.sha256(pickle.dumps((state['python'], state['numpy']), protocol=4))
    for tensor in [state['torch'], *state['cuda']]: checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()

class FrozenTimingState:
    """Keep model/state identities; restore only values, LR/SH/cursor and RNG.

    Source state is the actual restored native checkpoint before warmup. The
    bench RNG at that point is captured separately because native checkpoints
    do not contain RNG. Workload/observer counters are deliberately never reset.
    """
    def __init__(self, adapter):
        import torch
        self.adapter, self.gaussian, self.holder, self.optimizer = adapter, adapter.gaussian, adapter.holder, adapter.gaussian.optimizer
        self.live = _live_tensors(adapter)
        self.layouts = {name: _layout(tensor) for name, tensor in self.live.items()}
        self.frozen = {}
        for name, tensor in self.live.items():
            if not torch.isfinite(tensor).all().item(): raise RuntimeError('nonfinite initial timing state: ' + name)
            self.frozen[name] = tensor.detach().clone()
        self.groups_list = self.optimizer.param_groups
        self.groups = list(self.groups_list)
        self.parameter_lists = [group['params'] for group in self.groups]
        self.parameter_bindings = [tuple(group['params']) for group in self.groups]
        self.group_values = [{key: copy.deepcopy(value) for key, value in group.items() if key != 'params'} for group in self.groups]
        self.state_mapping = self.optimizer.state
        self.state_dicts = [self.optimizer.state[binding[0]] for binding in self.parameter_bindings]
        self.state_keys = [set(state) for state in self.state_dicts]
        self.nontensor_state = [{key: copy.deepcopy(value) for key, value in state.items() if not torch.is_tensor(value)} for state in self.state_dicts]
        self.defaults = self.optimizer.defaults
        self.default_values = copy.deepcopy(self.defaults)
        self.gaussian_values = {name: copy.deepcopy(getattr(self.gaussian, name)) for name in GAUSSIAN_SCALARS}
        self.cursor_values = {name: copy.deepcopy(getattr(adapter, name)) for name in CURSOR_SCALARS}
        self.scheduler = self.gaussian.xyz_scheduler_args
        self.rng = _rng_state()
        self.rng_sha256 = _rng_digest(self.rng)

    def _check_identities(self):
        adapter, optimizer = self.adapter, self.optimizer
        if adapter.gaussian is not self.gaussian or adapter.holder is not self.holder or self.gaussian.optimizer is not optimizer:
            raise RuntimeError('timing model/optimizer object replaced')
        if optimizer.param_groups is not self.groups_list or optimizer.state is not self.state_mapping or optimizer.defaults is not self.defaults:
            raise RuntimeError('timing optimizer container replaced')
        if len(optimizer.param_groups) != len(self.groups) or self.gaussian.xyz_scheduler_args is not self.scheduler:
            raise RuntimeError('timing group count or native scheduler identity changed')
        for index, group in enumerate(optimizer.param_groups):
            if group is not self.groups[index] or group['params'] is not self.parameter_lists[index] or len(group['params']) != len(self.parameter_bindings[index]) or any(first is not second for first, second in zip(group['params'], self.parameter_bindings[index])):
                raise RuntimeError('timing optimizer group/parameter identity changed')
            state = optimizer.state[group['params'][0]]
            if state is not self.state_dicts[index] or set(state) != self.state_keys[index] or set(group) != set(self.group_values[index]) | {'params'}:
                raise RuntimeError('timing optimizer state/group schema or dictionary identity changed')
        if set(optimizer.defaults) != set(self.default_values): raise RuntimeError('native optimizer defaults schema changed')
        current = _live_tensors(adapter)
        if set(current) != set(self.live): raise RuntimeError('timing state tensor names changed')
        for name, tensor in current.items():
            if tensor is not self.live[name] or _layout(tensor) != self.layouts[name]:
                raise RuntimeError('timing tensor object/storage/layout changed: ' + name)

    def restore_and_verify(self):
        import numpy as np
        import torch
        self._check_identities()
        counters = {'calls': dict(self.adapter.calls), 'observed_adam_steps': self.adapter.observed_adam_steps}
        with torch.no_grad():
            for name, tensor in self.live.items(): _copy_tensor(tensor, self.frozen[name])
        for group, values in zip(self.groups, self.group_values):
            for key, value in values.items(): group[key] = copy.deepcopy(value)
        for state, values in zip(self.state_dicts, self.nontensor_state):
            for key, value in values.items(): state[key] = copy.deepcopy(value)
        for key, value in self.default_values.items(): self.defaults[key] = copy.deepcopy(value)
        for name, value in self.gaussian_values.items(): setattr(self.gaussian, name, copy.deepcopy(value))
        for name, value in self.cursor_values.items(): setattr(self.adapter, name, copy.deepcopy(value))
        self.adapter.last_gradients.clear()
        random.setstate(self.rng['python']); np.random.set_state(copy.deepcopy(self.rng['numpy']))
        torch.set_rng_state(self.rng['torch'])
        if self.rng['cuda']: torch.cuda.set_rng_state_all(self.rng['cuda'])
        self._check_identities()
        for name, tensor in self.live.items():
            if not _same_bits(tensor, self.frozen[name]): raise RuntimeError('timing restoration bitwise verification failed: ' + name)
        for group, values in zip(self.groups, self.group_values):
            if {key: value for key, value in group.items() if key != 'params'} != values: raise RuntimeError('timing LR/hyperparameter restore failed')
        if self.defaults != self.default_values: raise RuntimeError('timing optimizer defaults restore failed')
        for state, values in zip(self.state_dicts, self.nontensor_state):
            if {key: value for key, value in state.items() if not torch.is_tensor(value)} != values:
                raise RuntimeError('timing non-tensor optimizer state restore failed')
        if any(getattr(self.gaussian, name) != value for name, value in self.gaussian_values.items()) or any(getattr(self.adapter, name) != value for name, value in self.cursor_values.items()):
            raise RuntimeError('timing SH/scalar/cursor restore failed')
        if _rng_digest(_rng_state()) != self.rng_sha256: raise RuntimeError('timing RNG restoration failed')
        if counters != {'calls': dict(self.adapter.calls), 'observed_adam_steps': self.adapter.observed_adam_steps}:
            raise RuntimeError('timing restore cleared or changed workload counters')
        return {'bitwise_verified': True, 'tensor_count': len(self.live), 'object_and_storage_identity_preserved': True, 'cpu_adam_steps_verified': 6, 'rng_verified': True, 'counters_preserved': counters}

    def contract(self):
        return {'source': 'actual restored native checkpoint before any warmup', 'parameters': 6, 'buffers': 3, 'optimizer_tensor_state': len(self.live) - 9, 'all_adam_state_including_cpu_step': True, 'inplace_identity_preserving': True, 'verification': 'all tensor bits, shapes/dtypes/devices/storage aliases, LR/SH/cursor/RNG; workload counters remain cumulative', 'rng_source': 'bench RNG after build; native capture itself does not store RNG', 'native_full_training': False}
