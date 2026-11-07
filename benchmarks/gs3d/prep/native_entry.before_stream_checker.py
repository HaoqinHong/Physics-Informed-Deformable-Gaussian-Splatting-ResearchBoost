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
    finite = {}
    for iteration in c['schedule']['checkpoint']:
        checkpoint = Path(args.out) / 'model' / f'chkpnt{iteration}.pth'
        state, actual_iteration = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if actual_iteration != iteration:
            raise ValueError('checkpoint iteration mismatch')
        def check(value):
            if torch.is_tensor(value):
                if not torch.isfinite(value).all().item():
                    raise ValueError(f'nonfinite saved checkpoint {iteration}')
            elif isinstance(value, dict):
                for v in value.values(): check(v)
            elif isinstance(value, (list, tuple)):
                for v in value: check(v)
        check(state)
        finite[str(iteration)] = {'gaussians': int(state[1].shape[0]), 'finite': True}
    from plyfile import PlyData
    for iteration in c['schedule']['save']:
        path = Path(args.out) / 'model/point_cloud' / f'iteration_{iteration}/point_cloud.ply'
        ply = PlyData.read(path)
        for prop in ply['vertex'].properties:
            if not np.isfinite(ply['vertex'][prop.name]).all():
                raise ValueError('nonfinite saved PLY')
    write_json(Path(args.out) / 'native_complete.json', finite)

if __name__ == '__main__':
    main()
