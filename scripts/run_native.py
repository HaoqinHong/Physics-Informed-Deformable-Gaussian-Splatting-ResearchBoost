"""Sequential original 30k train + all-heldout quality; no replacement timer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', choices=['baseline', 'candidate'], required=True)
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    p.add_argument('--scene', choices=['all', 'bonsai', 'bicycle', 'garden'], default='all')
    p.add_argument('--seed', choices=['all', '0', '1'], default='all')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise RuntimeError('invoke through scripts/gpu.py')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    workspace = a.workspace.resolve()
    harness = workspace / 'benchmarks/gs3d'
    config = harness / 'task/config.json'
    tree = workspace / 'gs3d' / a.tree
    inputs = {scene: {} for scene in ['bonsai', 'bicycle', 'garden']}
    for seed in ([0, 1] if a.seed == 'all' else [int(a.seed)]):
        for scene in (['bonsai', 'bicycle', 'garden'] if a.scene == 'all' else [a.scene]):
            run, quality = out / f'{scene}_s{seed}', out / f'quality_{scene}_s{seed}'
            subprocess.run([sys.executable, str(harness / 'bench/full_train.py'), '--config', str(config), '--tree', str(tree), '--scene', scene, '--seed', str(seed), '--out', str(run)], check=True)
            subprocess.run([sys.executable, str(harness / 'bench/quality.py'), '--config', str(config), '--tree', str(tree), '--scene', scene, '--model', str(run / 'model'), '--out', str(quality)], check=True)
            inputs[scene][str(seed)] = str(quality / 'quality_metrics.json')
            (out / 'quality-inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')


if __name__ == '__main__':
    main()
