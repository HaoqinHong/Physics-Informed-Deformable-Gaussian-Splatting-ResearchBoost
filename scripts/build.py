"""Build tree-local extensions; invoke only through scripts/gpu.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['pidg', 'gs3d'])
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    p.add_argument('--include-rejected', action='store_true')
    a = p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise RuntimeError('invoke via scripts/gpu.py with one allocated GPU')
    root = a.workspace.resolve() / a.experiment
    base, candidate = root / 'baseline', root / 'candidate'
    raster = 'flow-based-diff-gaussian-rasterization' if a.experiment == 'pidg' else 'diff-gaussian-rasterization'
    records = []

    def build(tree, package):
        directory = tree / 'submodules' / package
        log = root / (tree.name + '-' + package + '.build.log')
        with log.open('w') as stream:
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True)
        records.append({'tree': tree.name, 'package': package, 'log': str(log)})

    build(base, 'simple-knn')
    build(base, raster)
    for target in [candidate] + ([root / 'rejected-both'] if a.include_rejected else []):
        for package in ['simple-knn'] + ([raster] if a.experiment == 'pidg' and target == candidate else []):
            for binary in (base / 'submodules' / package).rglob('_C*.so'):
                destination = target / binary.relative_to(base)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(binary, destination)
        if a.experiment == 'gs3d' or target.name == 'rejected-both':
            build(target, raster)
    if a.experiment == 'pidg':
        subprocess.run([sys.executable, '-c', 'import hashencoder'], cwd=base, check=True)
    binaries = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob('*.so') if 'build' not in path.parts}
    (root / 'build.json').write_text(json.dumps({'builds': records, 'binaries': binaries,
        'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'arch': os.environ['TORCH_CUDA_ARCH_LIST'],
        'status': 'built_not_numerically_certified'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
