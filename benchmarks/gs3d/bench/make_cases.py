"""CPU-only case preparation after baseline native checkpoints exist."""
import argparse
import json
from pathlib import Path
from common import config, digest, sha256, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--models', required=True, help='JSON scene -> baseline seed0 native model directory')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    c = config(a.config)
    models = json.loads(Path(a.models).read_text())
    cases = []
    for scene in ['bonsai', 'bicycle', 'garden']:
        model = Path(models[scene]).resolve()
        cameras = json.loads((model / 'cameras.json').read_text())
        # Upstream Scene writes test cameras first, then train cameras, before shuffle.
        # Native COLMAP eval holds out sorted camera indices i % 8 == 0.
        test_count = (len(cameras) + 7) // 8
        train = cameras[test_count:]
        names = [train[i]['img_name'] for i in [0, len(train) // 2, len(train) - 1]]
        if len(set(names)) != 3:
            raise ValueError('three distinct train cameras required')
        for iteration in [100, 7000, 29900]:
            checkpoint = model / f'chkpnt{iteration}.pth'
            cases.append({'name': f'{scene}_i{iteration}', 'scene': scene, 'phase': f'i{iteration}', 'seed': 0, 'steps': 3, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'iteration': iteration, 'cameras': names, 'model_dir': str(model), 'contract': 'fixed-N non-event real native step; actual densification/pruning/reset events excluded'})
    write_json(a.out, cases)
    print(json.dumps({'cases': len(cases), 'sha256': digest(cases)}))

if __name__ == '__main__':
    main()
