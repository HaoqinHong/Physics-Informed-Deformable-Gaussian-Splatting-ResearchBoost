"""Create deployment-local harness copies before collecting/fixing new reference records."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['pidg', 'gs3d'])
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--viewer', help='explicit existing HOST_PID:CONTAINER_PID:BASELINE_GPU_MEMORY_MIB; gs3d only')
    a = p.parse_args()
    workspace, data = a.workspace.resolve(), a.data.resolve()
    uuid = subprocess.check_output(['nvidia-smi', '-i', a.gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if not uuid.startswith('GPU-') or '\n' in uuid or ',' in uuid:
        raise ValueError('one allocated GPU required')
    target = workspace / 'benchmarks' / a.experiment
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'benchmarks' / a.experiment, target, ignore=shutil.ignore_patterns('__pycache__'))
    task = target / 'task'
    task.mkdir(exist_ok=True)
    if a.experiment == 'pidg':
        c = json.loads((task / 'bench.json').read_text())
        c.update(baseline=str(workspace / 'pidg/baseline'), fixture=str(data / 'two-frame-fixture'),
                 public_fixture=str(data / 'public-pidg-fixture'), gpu_uuid=uuid)
        (task / 'bench.json').write_text(json.dumps(c, indent=2) + '\n')
    else:
        c = {'reference': str(workspace / 'gs3d/baseline'), 'data_manifest': str(task / 'data_manifest.json'),
             'gpu_uuid': uuid, 'precision': 'upstream', 'seeds': [0, 1],
             'extension_roots': ['submodules/diff-gaussian-rasterization', 'submodules/simple-knn'],
             'schedule': {'iterations': 30000, 'test': [7000, 30000], 'save': [7000, 30000], 'checkpoint': [100, 7000, 29900, 30000]},
             'scenes': {scene: {'path': str(data / scene), 'images': scale} for scene, scale in [('bonsai', 'images_2'), ('bicycle', 'images_4'), ('garden', 'images_4')]}}
        if a.viewer:
            host, container, memory = map(int, a.viewer.split(':'))
            ticks = int(Path(f'/proc/{container}/stat').read_text().rsplit(')', 1)[1].split()[19])
            c['registered_viewer'] = {'host_pid': host, 'container_pid': container, 'container_start_ticks': ticks,
                                      'baseline_memory_mib': memory, 'provenance': 'explicit owner registration before new baseline freeze'}
        (task / 'config.json').write_text(json.dumps(c, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
