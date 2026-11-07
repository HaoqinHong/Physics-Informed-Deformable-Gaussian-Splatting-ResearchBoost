"""CPU final contract manifest. Run before any optimizing session/candidate."""
import argparse
import json
from pathlib import Path
from common import config, digest, sha256, write_json

def evaluator_files(root):
    files = {}
    for sub in ['bench', 'numerics']:
        for p in sorted((Path(root) / sub).rglob('*.py')):
            if '__pycache__' not in p.parts:
                files[p.relative_to(root).as_posix()] = sha256(p)
    return files

def verify_manifest(root, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if evaluator_files(root) != manifest['evaluator_files']:
        raise ValueError('frozen evaluator changed')
    for path, expected in manifest['input_files'].items():
        if sha256(path) != expected:
            raise ValueError('frozen task input changed: ' + path)
    return manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--numerics-frozen', required=True)
    p.add_argument('--quality-frozen', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    c = config(a.config)
    numerical = Path(a.numerics_frozen)
    acceptance = json.loads((numerical / 'acceptance.json').read_text())
    if acceptance['status'] != 'READY_TO_FREEZE':
        raise ValueError('numerics formulation has not passed baseline acceptance')
    if Path(a.out).exists():
        raise FileExistsError('final formulation manifest must be new')
    root = Path(__file__).resolve().parents[1]
    evaluator = evaluator_files(root)
    inputs = [Path(a.config), Path(a.cases), Path(c['data_manifest']), numerical / 'acceptance.json', numerical / 'thresholds.json', numerical / 'generic_selftest.json', Path(a.quality_frozen)]
    thresholds = json.loads((numerical / 'thresholds.json').read_text())
    # Reference recording metadata and sketch payloads are frozen judge inputs.
    for record_dir in [thresholds['meta']['reference'], *thresholds['meta']['variants']]:
        inputs.extend(p for p in sorted(Path(record_dir).iterdir()) if p.suffix in {'.json', '.pt'})
    roles = {'config': str(Path(a.config).resolve()), 'cases': str(Path(a.cases).resolve()), 'thresholds': str((numerical / 'thresholds.json').resolve()), 'reference_records': str(Path(thresholds['meta']['reference']).resolve())}
    manifest = {'schema': 'original3dgs-formulation@1', 'evaluator_files': evaluator, 'evaluator_sha256': digest(evaluator), 'input_files': {str(p.resolve()): sha256(p) for p in inputs}, 'roles': roles, 'candidate_seen': False, 'formal_benefit': 'unavailable_shared_machine', 'precision': 'upstream', 'full_training_seeds': [0, 1], 'full_training_equivalence': 'not statistically established by two seeds', 'policy': 'Gates cannot be loosened after candidate results.'}
    write_json(a.out, manifest)
    print(json.dumps({'evaluator_sha256': manifest['evaluator_sha256'], 'inputs': len(inputs)}))

if __name__ == '__main__':
    main()
