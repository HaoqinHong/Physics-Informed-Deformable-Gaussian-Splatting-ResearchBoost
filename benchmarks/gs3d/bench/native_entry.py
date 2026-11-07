"""Run unmodified upstream train.py; the only wrapper is matched seed injection."""
import argparse
from pathlib import Path
import random
import runpy
import sys
from common import bind_tree, config, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--tree', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    c = config(args.config)
    if args.seed not in c['seeds']:
        raise ValueError('seed outside frozen matched set')
    binding = bind_tree(args.tree, c)
    import numpy as np
    import torch
    import utils.general_utils as general
    original_safe_state = general.safe_state
    def matched_safe_state(silent):
        original_safe_state(silent)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    general.safe_state = matched_safe_state
    s = c['scenes'][args.scene]
    sys.argv = [str(Path(args.tree) / 'train.py'), '--source_path', s['path'], '--images', s['images'], '--model_path', str(Path(args.out) / 'model'), '--eval', '--iterations', '30000', '--port', '0', '--test_iterations', '7000', '30000', '--save_iterations', '7000', '30000', '--checkpoint_iterations', '100', '7000', '29900', '30000']
    write_json(Path(args.out) / 'native_binding.json', {'binding': binding, 'argv': sys.argv, 'seed': args.seed, 'seed_wrapper': 'upstream safe_state followed by random/numpy/torch seed; no other patch'})
    runpy.run_path(sys.argv[0], run_name='__main__')
    # Exit zero alone does not establish a completed or finite training run.
    import json
    import os
    from common import bounded_run, sha256
    out = Path(args.out)
    artifact_report = out / 'native_artifacts.json'
    if artifact_report.exists():
        raise FileExistsError('native postcheck report already exists; never overwrite')
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable, '-X', 'faulthandler', str(Path(__file__).with_name('artifact_checks.py')), 'validate', '--model', str(out / 'model'), '--out', str(artifact_report)]
    process = bounded_run(command, cwd=out, env=environment, log=out / 'cpu_artifact_validation.log', timeout=7200)
    artifacts = json.loads(artifact_report.read_text())
    if artifacts.get('schema') != 'gs3d-native-artifacts@1' or artifacts.get('validation_complete') is not True:
        raise ValueError('fresh CPU child did not produce complete native artifact validation')
    artifacts['postcheck_cpu_process'] = {'fresh_exec': True, 'CUDA_VISIBLE_DEVICES': '', 'faulthandler': True, 'parent_pid': os.getpid(), 'process': process, 'artifact_report': str(artifact_report), 'artifact_report_sha256': sha256(artifact_report)}
    write_json(out / 'native_complete.json', artifacts)

if __name__ == '__main__':
    main()
