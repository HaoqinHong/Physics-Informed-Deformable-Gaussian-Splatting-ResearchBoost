"""Full held-out evaluation at native 7k/30k checkpoints, with baseline metrics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from common import assert_data, assert_source_contract, bind_tree, bounded_run, child_environment, config, digest, new_output, sha256, write_json

def frozen_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def evaluate(a, c, out):
    import numpy as np
    binding = bind_tree(a.tree, c)
    import torch
    from PIL import Image
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams
    from scene import GaussianModel, Scene
    import scene.gaussian_model as gaussian_module
    from gaussian_renderer import render
    from artifact_checks import scoped_gaussian_ply_reader
    losses = frozen_module(Path(c['reference']) / 'utils/loss_utils.py', 'frozen_quality_loss')
    images = frozen_module(Path(c['reference']) / 'utils/image_utils.py', 'frozen_quality_images')
    parser = ArgumentParser()
    mp, pp = ModelParams(parser), PipelineParams(parser)
    s = c['scenes'][a.scene]
    args = parser.parse_args(['--source_path', s['path'], '--images', s['images'], '--model_path', str(Path(a.model).resolve()), '--eval'])
    saved_paths = [Path(a.model) / 'point_cloud' / f'iteration_{iteration}/point_cloud.ply' for iteration in [7000, 30000]]
    results = {'binding': binding, 'metrics_source': c['reference'], 'metrics': ['PSNR', 'SSIM', 'L1'], 'LPIPS': 'not requested; no weights downloaded', 'iterations': {}, 'sequential_model_ply_reads': {}}
    for iteration in [7000, 30000]:
        gaussian = GaussianModel(args.sh_degree)
        with scoped_gaussian_ply_reader(gaussian_module, saved_paths) as receipts:
            scene = Scene(mp.extract(args), gaussian, load_iteration=iteration, shuffle=False)
        results['sequential_model_ply_reads'].update(receipts)
        cameras = scene.getTestCameras()
        if not cameras:
            raise ValueError('held-out split empty')
        chosen = {0, len(cameras) // 2, len(cameras) - 1}
        frames, grid_rows = [], []
        with torch.no_grad():
            for i, camera in enumerate(cameras):
                predicted = torch.clamp(render(camera, gaussian, pp.extract(args), torch.zeros(3, dtype=torch.float32, device='cuda'))['render'], 0, 1)
                truth = torch.clamp(camera.original_image.to('cuda'), 0, 1)
                if not torch.isfinite(predicted).all().item():
                    raise ValueError('nonfinite held-out render')
                row = {'index': i, 'image_name': camera.image_name, 'height': camera.image_height, 'width': camera.image_width, 'PSNR': float(images.psnr(predicted, truth).mean()), 'SSIM': float(losses.ssim(predicted, truth)), 'L1': float(losses.l1_loss(predicted, truth))}
                if not all(np.isfinite(row[k]) for k in ['PSNR', 'SSIM', 'L1']):
                    raise ValueError('nonfinite quality metric')
                frames.append(row)
                if i in chosen:
                    strip = torch.cat([truth, predicted, (predicted - truth).abs()], dim=2)
                    array = (strip.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    image = Image.fromarray(array)
                    image.thumbnail((1200, 500))
                    grid_rows.append(image)
        canvas = Image.new('RGB', (max(im.width for im in grid_rows), sum(im.height for im in grid_rows)), (16, 16, 16))
        y = 0
        for image in grid_rows:
            canvas.paste(image, (0, y)); y += image.height
        canvas.save(out / f'samples_{iteration}.png')
        results['iterations'][str(iteration)] = {'count': len(frames), 'frames': frames, 'mean': {k: float(np.mean([r[k] for r in frames])) for k in ['PSNR', 'SSIM', 'L1']}, 'worst': {'PSNR': min(r['PSNR'] for r in frames), 'SSIM': min(r['SSIM'] for r in frames), 'L1': max(r['L1'] for r in frames)}, 'grid_indices': sorted(chosen), 'grid_columns': ['ground truth', 'render', 'absolute error']}
        del scene, gaussian, cameras
        torch.cuda.empty_cache()
    write_json(out / 'quality_metrics.json', results)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--tree', required=True)
    p.add_argument('--scene', choices=['bonsai', 'bicycle', 'garden'], required=True)
    p.add_argument('--model', required=True, help='native model directory, including point_cloud/')
    p.add_argument('--native-manifest', help='explicit full_train.recovered.json; default is original full_train.json')
    p.add_argument('--out', required=True)
    p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    c = config(a.config)
    if a.worker:
        evaluate(a, c, Path(a.out))
        return
    source, data = assert_source_contract(a.tree, c['reference']), assert_data(c, a.scene)
    model = Path(a.model).resolve()
    native_path = Path(a.native_manifest).resolve() if a.native_manifest else model.parent / 'full_train.json'
    if native_path.parent != model.parent:
        raise ValueError('native manifest must belong to this exact model run')
    native = json.loads(native_path.read_text())
    recovered_artifacts = None
    if native['status'] == 'complete_postcheck_recovered':
        if not a.native_manifest: raise ValueError('recovered native manifest requires explicit --native-manifest')
        from artifact_checks import verify_recovered_manifest
        native, recovered_artifacts = verify_recovered_manifest(native_path)
    if native['status'] not in ['complete', 'complete_postcheck_recovered'] or native['scene'] != a.scene or native['source']['sha256'] != source['sha256'] or native['data_sha256'] != data or native['config_sha256'] != digest(c):
        raise ValueError('quality model is not a completed native run of this exact tree/data/scene')
    model_hashes = {str(iteration): sha256(model / 'point_cloud' / f'iteration_{iteration}/point_cloud.ply') for iteration in [7000, 30000]}
    if recovered_artifacts and any(model_hashes[key] != recovered_artifacts['ply'][key]['sha256'] for key in ['7000', '30000']):
        raise ValueError('quality saved PLY differs from strict recovery proof')
    out = new_output(a.out)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--config', str(Path(a.config).resolve()), '--tree', str(Path(a.tree).resolve()), '--scene', a.scene, '--model', str(Path(a.model).resolve()), '--out', str(out)]
    report = {'status': 'running', 'kind': 'all_native_heldout_frames', 'source': source, 'data_sha256': data, 'config_sha256': digest(c), 'seed': native['seed'], 'model_ply_sha256': model_hashes, 'native_run_manifest': str(native_path), 'native_run_status': native['status'], 'native_run_manifest_sha256': sha256(native_path)}
    write_json(out / 'quality.json', report)
    try:
        report['process'] = bounded_run(command, cwd=out, env=child_environment(out), log=out / 'quality.log', timeout=a.timeout)
        if not (out / 'quality_metrics.json').is_file():
            raise ValueError('quality artifact missing')
        if source != assert_source_contract(a.tree, c['reference']) or data != assert_data(c, a.scene):
            raise ValueError('source/data changed during quality evaluation')
        if model_hashes != {str(iteration): sha256(model / 'point_cloud' / f'iteration_{iteration}/point_cloud.ply') for iteration in [7000, 30000]}:
            raise ValueError('saved model changed during quality evaluation')
        if report['native_run_manifest_sha256'] != sha256(native_path): raise ValueError('native manifest changed during quality evaluation')
        if recovered_artifacts: verify_recovered_manifest(native_path)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='error', error=repr(error)); raise
    finally:
        write_json(out / 'quality.json', report)

if __name__ == '__main__':
    main()
