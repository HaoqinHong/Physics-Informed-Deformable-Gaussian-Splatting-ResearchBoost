"""Actual PIDG module-group adapter. Imported only in a GPU worker process.

No renderer, flow, geometry, velocity network or loss is reimplemented here.
The identity deformation is an explicitly scoped initialization fixture, not a
replacement for the training deformation/physics objectives.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

from common import POLICY, RASTER, file_sha, value_sha, write_json
from numerics.adapter import Adapter, Objects
from numerics.record import seed_everything, tensor_sha

GAUSSIAN_PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
TAPPED_OUTPUTS = ("render", "depth", "alpha", "radii", "proj_means_2D", "conic_2D", "conic_2D_inv",
                  "weight_per_gs_pixel", "x_mu", "means3D_deform")


def load_upstream(selection):
    base = str(Path(selection["base_tree"]).resolve())
    raster = str(Path(selection["raster_root"]).resolve())
    sys.path[:0] = [raster, base]
    # Normal imports execute the real package initializers. Any hashencoder build
    # belongs to startup, in this worker's private working directory.
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.dataset_readers import fetchPly, readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from arguments import OptimizationParams
    flow_spec = importlib.util.spec_from_file_location("utils.flow_utils", selection["flow_file"])
    flow = importlib.util.module_from_spec(flow_spec)
    sys.modules["utils.flow_utils"] = flow
    flow_spec.loader.exec_module(flow)
    observed = SimpleNamespace(visualization_writes=0)
    original_save = flow.save_vis_flow_tofile

    def observed_save(*args, **kwargs):
        observed.visualization_writes += 1
        return original_save(*args, **kwargs)

    flow.save_vis_flow_tofile = observed_save
    raster_module = importlib.import_module("diff_gaussian_rasterization")
    extension = importlib.import_module("diff_gaussian_rasterization._C")
    if Path(extension.__file__).resolve() != Path(selection["extension"]).resolve():
        raise RuntimeError(f"wrong raster extension loaded: {extension.__file__}; expected {selection['extension']}")
    if Path(raster_module.__file__).resolve().parent != Path(selection["raster_root"]).resolve() / "diff_gaussian_rasterization":
        raise RuntimeError("raster Python wrapper is not tree-local")
    parser = argparse.ArgumentParser(add_help=False)
    optimization = OptimizationParams(parser).extract(parser.parse_args([]))
    return SimpleNamespace(GaussianModel=GaussianModel, fetchPly=fetchPly, readCameras=readCamerasFromTransforms,
                           cameraList=cameraList_from_camInfos, render=render, l1=l1_loss, ssim=ssim,
                           flow=flow.motion_flow_velocity, optimization=optimization,
                           observed=observed,
                           velocity_source=importlib.import_module(GaussianModel.__module__).__file__,
                           extension=extension.__file__)


def camera_description(camera):
    result = {"image_name": camera.image_name, "width": camera.image_width, "height": camera.image_height,
              "FoVx": float(camera.FoVx), "FoVy": float(camera.FoVy), "fid": camera.fid.cpu().tolist()}
    for name in ("original_image", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center", "k", "w2c"):
        value = getattr(camera, name)
        result[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}
    for index, value in enumerate(camera.flow or []):
        result[f"flow_{index}"] = None if value is None else {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}
    for name in ("depth", "motion_mask"):
        value = getattr(camera, name)
        result[name] = None if value is None else {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(torch.as_tensor(value))}
    return result


class ModuleGroup(nn.Module):
    def __init__(self, gaussian, cameras, upstream, workload, out, config):
        super().__init__()
        self.gaussian, self.cameras, self.upstream = gaussian, cameras, upstream
        self.workload, self.out, self.config = workload, Path(out), config
        for name in GAUSSIAN_PARAMETERS:
            self.register_parameter(name, getattr(gaussian, name))
        self.velocity_net = gaussian.velocity_net
        self.register_parameter("force_sigma", gaussian.force_sigma)
        self.taps = nn.ModuleDict({name: nn.Identity() for name in TAPPED_OUTPUTS})
        self.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self.background = torch.zeros(3, dtype=torch.float32, device="cuda")
        self.render_calls = self.backward_calls = self.velocity_calls = self.flow_calls = self.optimizer_calls = 0
        self.last_packages, self.deformation_inputs = [], []

    def identity(self):
        warp = torch.zeros((self._xyz.shape[0], 7), dtype=self._xyz.dtype, device=self._xyz.device)
        warp[:, 0] = 1.0
        return warp.requires_grad_(True)

    def _render(self, warp, flow, rotation=0.0, scaling=0.0):
        # train_pidg.py renders all three packages from the CURRENT viewpoint.
        package = self.upstream.render(self.cameras[0], self.gaussian, self.pipe, self.background,
                                       warp, rotation, scaling, enable_flow_grad=flow)
        self.render_calls += 1
        for name, tap in self.taps.items():
            tap(package[name])
        self.last_packages.append(package)
        return package

    def forward(self, iteration):
        self.last_packages, self.deformation_inputs = [], []
        is_triple = self.workload == "triple-flow"
        if is_triple:
            current_warp, next_warp = self.identity(), self.identity()
            # Deformation-output gradients are observed, while the real
            # Gaussian/velocity parameters use the upstream Adam optimizer.
            self.deformation_inputs = [current_warp, next_warp]
            package = self._render(current_warp, True)
        else:
            # The original warm-up render accepts scalar zero deformation.
            package = self._render(0.0, self.workload == "raster-flow")
        image = package["render"]
        gt = self.cameras[0].original_image.cuda()
        l1 = self.upstream.l1(image, gt)
        dssim = 1.0 - self.upstream.ssim(image, gt)
        reconstruction = (1.0 - self.config["lambda_dssim"]) * l1 + self.config["lambda_dssim"] * dssim
        losses = {"l1": l1, "dssim": dssim, "reconstruction": reconstruction}
        total = reconstruction
        if is_triple:
            next_package = self._render(next_warp, True)
            velocity, _stress = self.gaussian.get_velocity_and_stress(self.cameras[0].fid)
            self.velocity_calls += 1
            dt = self.cameras[0].next_cam.fid - self.cameras[0].fid
            xyz_t_velocity = current_warp[..., 4:, None] + velocity.unsqueeze(2) * dt
            velocity_warp = torch.cat([current_warp[..., :4], xyz_t_velocity.squeeze(-1)], dim=-1)
            velocity_package = self._render(velocity_warp, True)
            flow = self.upstream.flow(iteration, self.gaussian, self.cameras[0], package, next_package,
                                      velocity_package, str(self.out), dataset_type="PIDG")
            self.flow_calls += 1
            losses["flow"] = flow
            total = total + self.config["flow_loss_weight"] * flow
        losses["total"] = total
        return losses


class PIDGAdapter(Adapter):
    name = "pidg_exploratory_module_group"

    def __init__(self, selection, fixture, scene, workload, out, config, initial_state=None):
        self.selection, self.fixture, self.scene = selection, Path(fixture), scene
        self.workload, self.out, self.config = workload, Path(out), config
        self.initial_state = Path(initial_state) if initial_state else None

    def build(self, seed, golden=False):
        if golden:
            raise NotImplementedError("upstream CUDA rasterizer has no fp64 gold path; no synthetic arbitration is provided")
        seed_everything(seed)
        self.upstream = load_upstream(self.selection)
        scene_path = self.fixture / self.scene
        infos, test_infos = self.upstream.readCameras(str(scene_path), "transforms_train.json", "transforms_test.json", False)
        if len(infos) != 2 or test_infos:
            raise ValueError("require the frozen two-frame derivative, not original full transforms")
        args = SimpleNamespace(resolution=-1, data_device="cuda", load2gpu_on_the_fly=False)
        cameras = self.upstream.cameraList(infos, 1.0, args)
        camera = cameras[0]
        if camera.next_cam is None or camera.flow[0] is None or camera.next_cam.flow[1] is None or camera.next_cam.depth is None:
            raise ValueError("require original current forward flow, next backward flow and next depth")
        for info, loaded in zip(infos, cameras):
            expected = min(1600, info["cam"].image.width)
            if loaded.image_width != expected:
                raise ValueError("upstream resolution=-1 did not resolve to its original 1600-width policy")
        gaussian = self.upstream.GaussianModel(self.config["sh_degree"])
        if self.initial_state:
            frozen = torch.load(self.initial_state, map_location="cpu", weights_only=True)
            for name in GAUSSIAN_PARAMETERS:
                setattr(gaussian, name, nn.Parameter(frozen["parameters"][name].cuda().clone().requires_grad_(True)))
            gaussian.velocity_net.load_state_dict(frozen["velocity_net"])
            gaussian.force_sigma = nn.Parameter(frozen["force_sigma"].cuda().clone().requires_grad_(True))
            gaussian.particle_indices = frozen["particle_indices"].cuda().clone()
            gaussian.dynamic_mask = frozen["dynamic_mask"].cuda().clone()
            gaussian.max_radii2D = frozen["max_radii2D"].cuda().clone()
            gaussian.active_sh_degree = frozen["active_sh_degree"]
            gaussian.spatial_lr_scale = frozen["spatial_lr_scale"]
        else:
            pcd = self.upstream.fetchPly(str(scene_path / "points3d.ply"))
            if len(pcd.points) != self.config["expected_points"]:
                raise ValueError("public PLY point count changed")
            gaussian.create_from_pcd(pcd, spatial_lr_scale=5)
        if gaussian.get_xyz.shape != (100000, 3):
            raise ValueError("original 100000 Gaussian count/shape changed")
        if not all(torch.isfinite(getattr(gaussian, name)).all() for name in GAUSSIAN_PARAMETERS):
            raise ValueError("nonfinite actual Gaussian initialization")
        gaussian.training_setup(self.upstream.optimization)
        self.group = ModuleGroup(gaussian, cameras, self.upstream, self.workload, self.out, self.config)
        self.optimizer = gaussian.optimizer
        self.camera_config = [camera_description(c) for c in cameras]
        self.runtime = {
            "scene": self.scene, "workload": self.workload, "points": 100000,
            "active_sh_degree": gaussian.active_sh_degree, "max_sh_degree": gaussian.max_sh_degree,
            "camera_sha256": value_sha(self.camera_config), "cameras": self.camera_config,
            "input_state_sha256": file_sha(self.initial_state) if self.initial_state else None,
            "parameter_inventory": {n: {"shape": list(p.shape), "dtype": str(p.dtype)} for n, p in self.group.named_parameters()},
            "upstream_optimizer": [{"name": g.get("name"), "lr": g["lr"], "eps": g.get("eps")} for g in self.optimizer.param_groups],
            "flow_visualization": "five original PNG writes per triple-flow invocation, retained inside timing",
            "scope": "module group; excludes deformation model, TV regularizers, densification and Cauchy physics",
            "original_scene_mode": self.config["original_scene_modes"][self.scene],
            "flow_case_scope": ("hypothetical enabled-flow component coverage; original rubber_duck training use_flow=false"
                                if self.scene == "rubber_duck" and self.workload != "raster-regular"
                                else "explicit phase-specific module coverage, not the full original training schedule"),
        }

    def save_initial(self, path):
        gaussian = self.group.gaussian
        state = {"parameters": {name: getattr(gaussian, name).detach().cpu() for name in GAUSSIAN_PARAMETERS},
                 "velocity_net": {n: t.detach().cpu() for n, t in gaussian.velocity_net.state_dict().items()},
                 "force_sigma": gaussian.force_sigma.detach().cpu(),
                 "particle_indices": gaussian.particle_indices.cpu(), "dynamic_mask": gaussian.dynamic_mask.cpu(),
                 "max_radii2D": gaussian.max_radii2D.cpu(), "active_sh_degree": gaussian.active_sh_degree,
                 "spatial_lr_scale": gaussian.spatial_lr_scale}
        torch.save(state, path)

    def objects(self):
        return Objects(models={"module": self.group}, optimizers={"adam": self.optimizer},
                       sync={"module": "local"}, capture_outputs=("module.taps.*",))

    def phases(self):
        return [self.workload]

    def set_phase(self, phase, step):
        if phase != self.workload:
            raise ValueError("workload phase changed")

    def batch(self, step, rank):
        if rank != 0:
            raise ValueError("this is a single assigned GPU benchmark")
        return self.group.cameras[0]

    def execute(self, iteration, backward=True, update=False):
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.group(iteration)
        if backward:
            losses["total"].backward()
            self.group.backward_calls += 1
        if update:
            self.assert_finite_before_update(losses, iteration)
            self.optimizer.step()
            self.group.optimizer_calls += 1
        return losses

    def assert_finite_before_update(self, losses, step):
        tensors = {"loss/" + n: t for n, t in losses.items()}
        for name, p in self.group.named_parameters():
            tensors["parameter/" + name] = p
            if p.grad is not None:
                tensors["gradient/" + name] = p.grad
        for index, package in enumerate(self.group.last_packages):
            for name in TAPPED_OUTPUTS:
                tensors[f"render{index}/{name}"] = package[name]
            if package["viewspace_points"].grad is not None:
                tensors[f"gradient/render{index}/viewspace_points"] = package["viewspace_points"].grad
        for index, value in enumerate(self.group.deformation_inputs):
            if value.grad is not None:
                tensors[f"gradient/deformation{index}"] = value.grad
        failures = []
        for name, value in tensors.items():
            bad = ~torch.isfinite(value.detach())
            count = int(bad.sum().item())
            if count:
                failures.append({"tensor": name, "shape": list(value.shape), "dtype": str(value.dtype),
                                 "nonfinite_count": count, "elements": value.numel(),
                                 "first_nonfinite_flat_indices": torch.nonzero(bad.reshape(-1), as_tuple=False)[:16].flatten().cpu().tolist()})
        if failures:
            report = {"status": "BASELINE_OR_CANDIDATE_NONFINITE", "step": step, "failures": failures,
                      "no_masking_or_gate_relaxation": True,
                      "scope": "full tensors include culled Gaussians and the next-conic SVD gradient path"}
            write_json(self.out.parent / "numerics-finiteness-error.json", report)
            raise RuntimeError(f"nonfinite tensor before diagnostic Adam update; see {self.out.parent / 'numerics-finiteness-error.json'}")

    def step(self, step, batch):
        # These are diagnostic module-group updates, not full PIDG train steps.
        losses = self.execute(step, backward=True, update=True)
        for index, package in enumerate(self.group.last_packages):
            value = package["viewspace_points"].grad
            if value is not None:
                losses[f"probe_gradient/render{index}/viewspace_points"] = value.detach()
        for index, value in enumerate(self.group.deformation_inputs):
            if value.grad is not None:
                losses[f"probe_gradient/deformation{index}"] = value.grad.detach()
        return losses

    def canonical(self, kind, name, tensor):
        prefix = "loss/probe_gradient/"
        if name.startswith(prefix):
            return {"grad/probe/" + name[len(prefix):]: tensor}
        return {name: tensor}

    def invariants(self):
        group = self.group
        actual_writes = self.upstream.observed.visualization_writes
        if actual_writes != 5 * group.flow_calls:
            raise RuntimeError(f"flow visualization workload changed: actual writes={actual_writes}, expected={5 * group.flow_calls}")
        return {"scene": self.scene, "workload": self.workload, "points": 100000,
                "camera_sha256": self.runtime["camera_sha256"], "render_calls": group.render_calls,
                "velocity_calls": group.velocity_calls, "flow_calls": group.flow_calls,
                "backward_calls": group.backward_calls, "optimizer_calls": group.optimizer_calls,
                "physics_calls": 0, "deformation_model_calls": 0, "resolution": -1,
                "flow_visualization_writes": actual_writes,
                "flow_visualization_expected_writes": 5 * group.flow_calls,
                "gs_per_pixel": [{"shape": list(p["gs_per_pixel"].shape),
                                  "dtype": str(p["gs_per_pixel"].dtype),
                                  "sha256": tensor_sha(p["gs_per_pixel"])} for p in group.last_packages]}
