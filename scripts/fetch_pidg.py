"""Fetch the exact public two-frame module inputs, validating recorded SHA256 bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'workspace/data/pidg')
    a = p.parse_args()
    target = a.output.resolve()
    public, fixture = target / 'public-pidg-fixture', target / 'two-frame-fixture'
    manifest = json.loads((ROOT / 'results/pidg/data-manifest.json').read_text())
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe relative data filename')
        path = public / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = path.read_bytes() if path.exists() else None
        if blob is None or hashlib.sha256(blob).hexdigest() != row['sha256']:
            with urllib.request.urlopen(row['url'], timeout=180) as stream:
                blob = stream.read()
            if len(blob) != row['bytes'] or hashlib.sha256(blob).hexdigest() != row['sha256']:
                raise ValueError('public input changed: ' + str(relative))
            path.write_bytes(blob)
    for scene in ['dry_ice', 'rubber_duck', 'balls_reaction', 'mechanics_cloth', 'motion_kuro']:
        source, dest = public / scene, fixture / scene
        dest.mkdir(parents=True, exist_ok=True)
        for name in ['train', 'flows_flo', 'depth_distill', 'points3d.ply']:
            link = dest / name
            if not link.exists():
                link.symlink_to(source / name, target_is_directory=(source / name).is_dir())
        data = json.loads((source / 'transforms_train.json').read_text())
        data['frames'] = data['frames'][:2]
        if [Path(frame['file_path']).name for frame in data['frames']] != ['0002', '0003']:
            raise ValueError('public train ordering differs from recorded inputs')
        (dest / 'transforms_train.json').write_text(json.dumps(data))
        data['frames'] = []
        (dest / 'transforms_test.json').write_text(json.dumps(data))
    print(target)


if __name__ == '__main__':
    main()
