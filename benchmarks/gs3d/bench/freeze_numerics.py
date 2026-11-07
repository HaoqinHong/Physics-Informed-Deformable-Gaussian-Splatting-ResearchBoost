"""CPU baseline-self/LOO/mutation checks, then write immutable pre-candidate gates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import new_output, write_json
from policy import POST_DISCRETE, strict_compare

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True, help='bench baseline output containing r0..r3/records and lr_control/records')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from numerics.judge import calibrate, manifests, thresholds_sha
    from numerics.selftest import run as generic_selftest
    root, out = Path(a.baseline), new_output(a.out)
    ref = root / 'r0/records'
    variants = [root / f'r{i}/records' for i in [1, 2, 3]]
    records = [manifests(path) for path in [ref, *variants]]
    keys = set(records[0])
    if len(keys) != 9 or any(set(r) != keys for r in records):
        raise ValueError('four independent complete nine-case baseline records required')
    thresholds = calibrate(ref, variants)
    loo = []
    for held in variants:
        th = calibrate(ref, [v for v in variants if v != held])
        result = strict_compare(ref, held, th)
        loo.append({'held': str(held), 'status': result['status'], 'violations': result['violations'], 'exact_failures': result['exact_failures'], 'errors': result['errors']})
    self_comparisons = [strict_compare(ref, variant, thresholds) for variant in variants]
    # Generic code is untouched; test its own complete failure paths on a
    # bitwise self-record. Real independently repeated baseline acceptance is
    # separately established by the task LOO checks above.
    generic = generic_selftest(ref, ref, thresholds, out=out / 'generic_selftest.json')
    control = strict_compare(ref, root / 'lr_control/records', thresholds)
    accepted = all(r['status'] == 'PASS' for r in self_comparisons) and all(r['status'] == 'PASS' for r in loo) and generic['ok'] and control['status'] == 'INCORRECT' and bool(control['exact_failures'])
    evidence = {'status': 'READY_TO_FREEZE' if accepted else 'REJECTED_FORMULATION', 'baseline_self_statuses': [r['status'] for r in self_comparisons], 'bitwise_baseline_reproducible': all(r['checked'] == r['bitwise_identical'] for r in self_comparisons), 'loo': loo, 'generic_mutations_ok': generic['ok'], 'lr_control_status': control['status'], 'lr_control_exact_failures': control['exact_failures'], 'post_discrete_policy': POST_DISCRETE, 'fp64_arbitration': 'unavailable: no faithful original raster fp64 path', 'thresholds_sha256': thresholds_sha(thresholds), 'constraint': 'No candidate has been measured. Gates may not be widened after candidate results.'}
    write_json(out / 'acceptance.json', evidence)
    write_json(out / 'lr_control.json', control)
    if not accepted:
        raise RuntimeError('baseline-self, LOO or negative control failed; do not freeze or optimize')
    write_json(out / 'thresholds.json', thresholds)
    print(json.dumps({'status': evidence['status'], 'thresholds_sha256': evidence['thresholds_sha256']}))

if __name__ == '__main__':
    main()
