"""Fixed 30k native training driver. Launch through the assigned gpu_exec.py."""
import argparse
from pathlib import Path
import sys
from common import assert_data, assert_source_contract, bounded_run, child_environment, config, digest, new_output, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--tree', required=True)
    p.add_argument('--scene', required=True, choices=['bonsai', 'bicycle', 'garden'])
    p.add_argument('--seed', required=True, type=int, choices=[0, 1])
    p.add_argument('--out', required=True)
    p.add_argument('--timeout', type=float, default=21600)
    a = p.parse_args()
    c = config(a.config)
    source = assert_source_contract(a.tree, c['reference'])
    data_hash = assert_data(c, a.scene)
    out = new_output(a.out)
    command = [sys.executable, str(Path(__file__).with_name('native_entry.py')), '--config', str(Path(a.config).resolve()), '--tree', str(Path(a.tree).resolve()), '--scene', a.scene, '--seed', str(a.seed), '--out', str(out)]
    metadata = {'kind': 'full_native_30000', 'source': source, 'data_sha256': data_hash, 'config_sha256': digest(c), 'scene': a.scene, 'seed': a.seed, 'gpu_uuid': c['gpu_uuid'], 'status': 'running'}
    write_json(out / 'full_train.json', metadata)
    try:
        metadata['process'] = bounded_run(command, cwd=str(out), env=child_environment(out), log=out / 'train.log', timeout=a.timeout)
        if not (out / 'native_complete.json').is_file():
            raise RuntimeError('native completion artifact missing')
        if assert_source_contract(a.tree, c['reference']) != source or assert_data(c, a.scene) != data_hash:
            raise RuntimeError('source or data changed during full training')
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'error'
        metadata['error'] = repr(error)
        raise
    finally:
        write_json(out / 'full_train.json', metadata)

if __name__ == '__main__':
    main()
