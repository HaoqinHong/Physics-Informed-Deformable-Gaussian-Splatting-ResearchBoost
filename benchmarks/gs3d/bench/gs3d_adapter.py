"""Original GaussianModel/Scene/render/loss/backward/Adam, at frozen real states."""
from argparse import ArgumentParser
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from numerics.adapter import Adapter, Objects
from common import assert_data, assert_source_contract, bind_tree, config, sha256

def checkpoint_to_native_cuda(checkpoint, gaussian, options, expected_iteration):
    """CPU deserialize; preserve native noncapturable Adam's CPU step scalars.

    Only the six trainable Parameters and three Gaussian buffers are explicitly
    moved. Upstream GaussianModel.restore/Adam.load_state_dict migrates moments
    according to parameter policy. Serial storage locations remain auditable.
    """
    import torch
    storage_locations = {}
    def keep_cpu(storage, serialized_location):
        storage_locations[storage._cdata] = serialized_location
        return storage
    state, iteration = torch.load(checkpoint, map_location=keep_cpu, weights_only=False)
    if iteration != expected_iteration or len(state) != 12:
        raise ValueError('native checkpoint iteration or capture layout differs')
    capture = list(state)
    checkpoint_inventory = {}
    for group in capture[10]['param_groups']:
        if group.get('capturable', False) or group.get('fused', False):
            raise ValueError('native workload requires noncapturable, nonfused Adam')
        if len(group['params']) != 1:
            raise ValueError('original six-group Adam layout changed')
        name = group['name']
        saved = capture[10]['state'][group['params'][0]]
        checkpoint_inventory[name] = {}
        for key, value in saved.items():
            if torch.is_tensor(value):
                location = storage_locations.get(value.untyped_storage()._cdata)
                if location is None:
                    raise ValueError('checkpoint tensor storage provenance unavailable')
                checkpoint_inventory[name][key] = {'serialized_device': location, 'loaded_device': str(value.device), 'shape': list(value.shape), 'dtype': str(value.dtype)}
        step = checkpoint_inventory[name].get('step')
        if step is None or step['serialized_device'] != 'cpu' or step['loaded_device'] != 'cpu':
            raise ValueError('original noncapturable Adam step must be serialized and loaded on CPU')
    if len(checkpoint_inventory) != 6:
        raise ValueError('all six native optimizer groups required')
    for index in range(1, 7):
        parameter = capture[index]
        if not isinstance(parameter, torch.nn.Parameter):
            raise ValueError('native capture trainable tensor is not nn.Parameter')
        capture[index] = torch.nn.Parameter(parameter.detach().to(device='cuda'), requires_grad=parameter.requires_grad)
    for index in [7, 8, 9]:
        capture[index] = capture[index].to(device='cuda')
    gaussian.restore(tuple(capture), options)
    restored_inventory = {}
    for group in gaussian.optimizer.param_groups:
        name, parameter = group['name'], group['params'][0]
        if group.get('capturable', False) or group.get('fused', False) or parameter.device.type != 'cuda':
            raise ValueError('restored native optimizer policy changed')
        restored_inventory[name] = {}
        for key, value in gaussian.optimizer.state[parameter].items():
            if torch.is_tensor(value):
                saved = checkpoint_inventory[name][key]
                expected_device = 'cpu' if key == 'step' else parameter.device.type
                if value.device.type != expected_device or list(value.shape) != saved['shape'] or str(value.dtype) != saved['dtype']:
                    raise ValueError('Adam state device/dtype/shape differs from native restore policy')
                restored_inventory[name][key] = {'device': str(value.device), 'shape': list(value.shape), 'dtype': str(value.dtype)}
        if set(restored_inventory[name]) != set(checkpoint_inventory[name]):
            raise ValueError('restored optimizer tensor-state names changed')
    return iteration, {'checkpoint': checkpoint_inventory, 'restored': restored_inventory, 'policy': 'CPU deserialize; six CUDA nn.Parameters + three CUDA buffers; upstream restore migrates moments; noncapturable step stays CPU'}

class GS3DAdapter(Adapter):
    name = 'original3dgs-fixed-N-real-phase'
    def __init__(self, config_path, tree, scratch, timing=False, lr_control=False):
        self.config_path, self.tree, self.scratch = config_path, str(Path(tree).resolve()), str(Path(scratch).resolve())
        self.c = config(config_path)
        self.timing = timing
        self.lr_control = lr_control
    def configure(self, case):
        self.case = case
    def build(self, seed, golden=False):
        if golden:
            raise ValueError('original CUDA raster has no faithful fp64 gold; arbitration unavailable')
        self.source = assert_source_contract(self.tree, self.c['reference'])
        self.data_sha = assert_data(self.c, self.case['scene'])
        if sha256(self.case['checkpoint']) != self.case['checkpoint_sha256']:
            raise ValueError('frozen native checkpoint changed')
        self.binding = bind_tree(self.tree, self.c)
        import torch
        from arguments import ModelParams, PipelineParams, OptimizationParams
        from scene import GaussianModel, Scene
        from gaussian_renderer import render
        from utils.loss_utils import l1_loss, ssim
        parser = ArgumentParser()
        mp, pp, op = ModelParams(parser), PipelineParams(parser), OptimizationParams(parser)
        scene_config = self.c['scenes'][self.case['scene']]
        Path(self.scratch).mkdir(parents=True, exist_ok=True)
        args = parser.parse_args(['--source_path', scene_config['path'], '--images', scene_config['images'], '--model_path', self.scratch, '--eval'])
        self.pipe, self.opt = pp.extract(args), op.extract(args)
        self.gaussian = GaussianModel(args.sh_degree)
        self.scene = Scene(mp.extract(args), self.gaussian, shuffle=False)
        iteration, self.optimizer_state_device_inventory = checkpoint_to_native_cuda(self.case['checkpoint'], self.gaussian, self.opt, self.case['iteration'])
        self.cameras = {v.image_name: v for v in self.scene.getTrainCameras()}
        if not set(self.case['cameras']).issubset(self.cameras):
            raise ValueError('frozen camera is not in real native train split')
        from numerics.record import tensor_sha
        self.camera_hashes = {name: {key: tensor_sha(getattr(self.cameras[name], key)) for key in ['original_image', 'world_view_transform', 'full_proj_transform', 'camera_center']} for name in self.case['cameras']}
        self.render, self.l1_loss, self.ssim = render, l1_loss, ssim
        self.background = torch.zeros(3, dtype=torch.float32, device='cuda')
        self.calls = {'render': 0, 'backward': 0, 'adam': 0, 'densification_stats': 0}
        self.observed_adam_steps = 0
        def observed_adam_step(_optimizer, _args, _kwargs):
            self.observed_adam_steps += 1
        self.adam_counter_hook = self.gaussian.optimizer.register_step_pre_hook(observed_adam_step)
        self.last_gradients = {}
        class Holder(torch.nn.Module):
            def __init__(holder, gaussian):
                super().__init__()
                for name in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
                    holder.register_parameter(name, getattr(gaussian, name))
                for name in ['max_radii2D', 'xyz_gradient_accum', 'denom']:
                    holder.register_buffer(name, getattr(gaussian, name))
                for name in ['image', 'viewspace', 'first_radii', 'first_visibility', 'post_radii', 'post_visibility', 'l1', 'ssim_value', 'viewspace_grad', 'xyz_after', 'features_dc_after', 'features_rest_after', 'scaling_after', 'rotation_after', 'opacity_after']:
                    holder.add_module(name, torch.nn.Identity())
        self.holder = Holder(self.gaussian)
        self.logical_iteration = iteration
        self.last_camera = None
    def objects(self):
        return Objects(models={'gaussian': self.holder}, optimizers={'adam': self.gaussian.optimizer}, sync={'gaussian': 'local'}, capture_outputs=('gaussian.*',), count_forwards=False)
    def phases(self):
        return ['i100', 'i7000', 'i29900']
    def set_phase(self, phase, step):
        if phase != self.case['phase']:
            raise ValueError('phase mismatch')
        # Timing is a fixed phase, cycling three schedule positions. It does not
        # masquerade as a compressed or resumed native full training schedule.
        self.logical_iteration = self.case['iteration'] + 1 + (step % 3 if self.timing else step)
        self.gaussian.update_learning_rate(self.logical_iteration)
        if self.lr_control:
            # Formulation-only negative control, recorded in the exact LR items.
            # Set every group's upstream LR first so the factor never compounds.
            rates = [self.gaussian.xyz_scheduler_args(self.logical_iteration), self.opt.feature_lr, self.opt.feature_lr / 20, self.opt.opacity_lr, self.opt.scaling_lr, self.opt.rotation_lr]
            for group, rate in zip(self.gaussian.optimizer.param_groups, rates):
                group['lr'] = rate * 1.02
        if self.logical_iteration % 1000 == 0:
            self.gaussian.oneupSHdegree()
    def batch(self, step, rank):
        if rank != 0:
            raise ValueError('one assigned GPU only')
        return self.cameras[self.case['cameras'][step % 3]]
    def run(self, step, camera, mode='step'):
        import torch
        # Attribution mode retains only the preceding step's gradient storage
        # until the next call, allowing a boundary audit without timed syncs.
        # Clear it before rendering so older gradient allocations are reusable.
        self.last_gradients.clear()
        self.last_camera = camera.image_name
        gaussian, h = self.gaussian, self.holder
        package = self.render(camera, gaussian, self.pipe, self.background)
        self.calls['render'] += 1
        image = h.image(package['render'])
        viewspace = h.viewspace(package['viewspace_points'])
        visibility, radii = package['visibility_filter'], package['radii']
        getattr(h, 'first_radii' if step == 0 else 'post_radii')(radii)
        getattr(h, 'first_visibility' if step == 0 else 'post_visibility')(visibility)
        if step == 0:
            from numerics.record import tensor_sha
            self.initial_discrete = {'radii_sha256': tensor_sha(radii), 'visibility_sha256': tensor_sha(visibility)}
        truth = camera.original_image.cuda()
        l1, structural = h.l1(self.l1_loss(image, truth)), h.ssim_value(self.ssim(image, truth))
        loss = (1 - self.opt.lambda_dssim) * l1 + self.opt.lambda_dssim * (1 - structural)
        if mode == 'forward':
            return {'total': loss, 'l1': l1, 'ssim': structural}
        loss.backward()
        self.calls['backward'] += 1
        h.viewspace_grad(viewspace.grad)
        if mode == 'forward-backward':
            self.last_gradients = {name: parameter.grad for name, parameter in self.holder.named_parameters()}
        with torch.no_grad():
            if mode == 'step':
                if self.logical_iteration < self.opt.densify_until_iter:
                    gaussian.max_radii2D[visibility] = torch.max(gaussian.max_radii2D[visibility], radii[visibility])
                    gaussian.add_densification_stats(viewspace, visibility)
                    self.calls['densification_stats'] += 1
                gaussian.optimizer.step()
                self.calls['adam'] += 1
                for name in ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']:
                    getattr(h, name + '_after')(getattr(gaussian, '_' + name))
            gaussian.optimizer.zero_grad(set_to_none=True)
        return {'total': loss, 'l1': l1, 'ssim': structural}
    def step(self, step, batch):
        return self.run(step, batch, mode='step')
    def invariants(self):
        return {'scene': self.case['scene'], 'checkpoint_iteration': self.case['iteration'], 'logical_iteration': self.logical_iteration, 'camera': self.last_camera, 'camera_set': self.case['cameras'], 'camera_tensor_sha256': self.camera_hashes[self.last_camera], 'optimizer_state_device_inventory': self.optimizer_state_device_inventory, 'gaussians': int(self.gaussian.get_xyz.shape[0]), 'active_sh_degree': self.gaussian.active_sh_degree, 'max_sh_degree': self.gaussian.max_sh_degree, 'spatial_lr_scale': float(self.gaussian.spatial_lr_scale), 'lambda_dssim': self.opt.lambda_dssim, 'event_densification': 0, 'event_opacity_reset': 0, 'observed_adam_steps': self.observed_adam_steps, **self.initial_discrete, **self.calls}
    def save_state(self, path):
        import random
        import numpy as np
        import torch
        torch.save({'capture': self.gaussian.capture(), 'iteration': self.logical_iteration, 'calls': self.calls, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, path)
    def load_state(self, path):
        # Frozen cases restore their real native checkpoint in build(). No
        # alternate adapter state is accepted by this workload contract.
        raise ValueError('alternate state restore prohibited; use frozen native case checkpoint')
