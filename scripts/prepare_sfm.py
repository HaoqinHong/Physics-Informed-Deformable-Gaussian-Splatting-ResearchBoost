"""Create native COLMAP PLY caches on CPU before freezing the static scene data."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    path = a.workspace.resolve() / 'gs3d/baseline/scene/colmap_loader.py'
    spec = importlib.util.spec_from_file_location('native_colmap_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dtype = [(name, 'f4') for name in ['x', 'y', 'z', 'nx', 'ny', 'nz']] + [(name, 'u1') for name in ['red', 'green', 'blue']]
    for scene in ['bonsai', 'bicycle', 'garden']:
        sparse = a.data.resolve() / scene / 'sparse/0'
        target = sparse / 'points3D.ply'
        if target.exists():
            raise FileExistsError('prepare caches exactly once before data freeze: ' + str(target))
        xyz, rgb, _ = module.read_points3D_binary(str(sparse / 'points3D.bin'))
        values = np.concatenate((xyz, np.zeros_like(xyz), rgb), axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, values))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(target)
        print(target)


if __name__ == '__main__':
    main()
