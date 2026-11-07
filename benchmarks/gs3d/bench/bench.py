"""Frozen CLI: baseline / record / time / profile / check (all GPU work via gpu_exec)."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import assert_data, assert_source_contract, bounded_run, child_environment, config, digest, new_output, sha256, write_json
from policy import compare_timing_runs, scan_error_text, strict_compare
from runtime import GPUSnapshotError, gpu_snapshot, wait_for_quiescence
from freeze_task import verify_manifest

def cases_from(a):
    cases = json.loads(Path(a.cases).read_text())
    return [c for c in cases if not a.case or c['name'] == a.case]

def launch(a, c, case, tree, action, out, control=False):
    out.mkdir()
    try:
        before = gpu_snapshot(c['gpu_uuid'], idle=True)
    except GPUSnapshotError as error:
        write_json(out / 'gpu_before.json', {'status': 'REJECTED', 'reason': error.reason, 'snapshot': error.snapshot})
        raise
    write_json(out / 'gpu_before.json', before)
    source, data = assert_source_contract(tree, c['reference']), assert_data(c, case['scene'])
    command = [sys.executable, str(Path(__file__).with_name('worker.py')), action, '--config', str(Path(a.config).resolve()), '--tree', str(Path(tree).resolve()), '--cases', str(Path(a.cases).resolve()), '--case', case['name'], '--out', str(out), '--mode', a.mode]
    if a.smoke: command.append('--smoke')
    if a.detailed: command.append('--detailed')
    if control: command.append('--lr-control')
    if action == 'profile': command.extend(['--profile-path', a.profile_path])
    try:
        process = bounded_run(command, cwd=out, env=child_environment(out), log=out / 'worker.log', timeout=a.timeout)
    finally:
        signatures = scan_error_text((out / 'worker.log').read_text(errors='replace')) if (out / 'worker.log').exists() else [{'signature': 'missing_worker_log'}]
        write_json(out / 'signature_scan.json', {'valid': not signatures, 'hits': signatures})
    if signatures:
        raise ValueError('invalid worker error signature; inspect signature_scan.json')
    post = wait_for_quiescence(c['gpu_uuid'], timeout=10, interval=.5, on_observation=lambda evidence: write_json(out / 'gpu_post_quiescence.json', evidence))
    after = post['snapshot']
    artifact = {'record': 'record_complete.json', 'record-dtype': 'record_complete.json', 'time': 'timing.json', 'profile': 'profile_complete.json'}[action]
    if not (out / artifact).is_file():
        raise ValueError('expected worker artifact missing after exit zero')
    if source != assert_source_contract(tree, c['reference']) or data != assert_data(c, case['scene']):
        raise ValueError('source/data changed during subprocess')
    write_json(out / 'process.json', {'process': process, 'source': source, 'data_sha256': data, 'gpu_after': after, 'post_quiescence_attempts': len(post['attempts']), 'config_sha256': digest(c)})
    if action == 'profile':
        bounded_run([sys.executable, str(Path(__file__).with_name('kernel_util.py')), str(out / 'trace.json'), '--out', str(out / 'kernel_util.md'), '--json', str(out / 'kernel_util.json')], cwd=out, env=__import__('os').environ.copy(), log=out / 'kernel_util.log', timeout=300)
    return json.loads((out / artifact).read_text())

def record_set(a, c, cases, tree, out, control=False, dtype_supplement=False):
    out.mkdir()
    records = out / 'records'; records.mkdir()
    frozen_source = assert_source_contract(tree, c['reference'])
    supplements = {}
    for case in cases:
        work = out / case['name']
        primary = launch(a, c, case, tree, 'record', work, control=control)
        if primary['dtype_trace'] or primary['source'] != frozen_source:
            raise ValueError('primary record was traced or source changed across cases')
        for f in (work / 'records').iterdir():
            shutil.copyfile(f, records / f.name)
        if dtype_supplement:
            diagnostic = out / (case['name'] + '_dtype_supplement')
            extra = launch(a, c, case, tree, 'record-dtype', diagnostic)
            if not extra['dtype_trace'] or extra['source'] != frozen_source:
                raise ValueError('dtype supplement role/source mismatch')
            manifest = json.loads((diagnostic / 'records' / f'{case["name"]}.r0.json').read_text())
            supplements[case['name']] = {'dtype_ops': manifest['dtype_ops'], 'math_mode': manifest['steps'][0]['math_mode'], 'manifest_sha256': sha256(diagnostic / 'records' / f'{case["name"]}.r0.json'), 'completion_sha256': sha256(diagnostic / 'record_complete.json'), 'source_sha256': frozen_source['sha256'], 'primary_record_dtype_trace': False, 'supplement_dtype_trace': True}
    if dtype_supplement:
        # This filename does not match the generic *.r*.json case glob. It is
        # nevertheless frozen by freeze_task's JSON input hashing, alongside
        # primary record payloads. Diagnostic values never enter numerics.
        write_json(records / 'dtype_supplement.json', {'schema': 'fresh-dtype-supplement@1', 'role': 'op dtype inventory only; traced values are not precision or timing evidence', 'process_separation': 'each primary and each supplement is a distinct bounded fresh worker with private cold caches', 'cases': supplements})
    return records

def compare_dtype_supplements(reference_records, candidate_records):
    ref_path, candidate_path = [Path(root) / 'dtype_supplement.json' for root in [reference_records, candidate_records]]
    if not ref_path.is_file() or not candidate_path.is_file():
        return {'status': 'UNAVAILABLE', 'reason': 'independent baseline/candidate dtype supplement missing'}
    reference, candidate = [json.loads(path.read_text())['cases'] for path in [ref_path, candidate_path]]
    if set(reference) != set(candidate):
        return {'status': 'INCORRECT', 'reason': 'dtype supplement cases differ'}
    changes = []
    for case in reference:
        # Op names/mixes may change under a real optimization. The dtype set
        # itself must remain at original precision; internal CUDA review is
        # still required because opaque extension accumulators are not traced.
        first, second = [item[case]['dtype_ops'] for item in [reference, candidate]]
        if first is None or second is None:
            return {'status': 'UNAVAILABLE', 'reason': 'dtype op inventory absent'}
        old, new = [{key.rsplit(':', 1)[1] for key in inventory} for inventory in [first, second]]
        if old != new:
            changes.append({'case': case, 'baseline_dtypes': sorted(old), 'candidate_dtypes': sorted(new)})
        if reference[case]['math_mode'] != candidate[case]['math_mode']:
            changes.append({'case': case, 'math_mode_changed': True})
    return {'status': 'INCORRECT' if changes else 'SUPPORTED_DIAGNOSTIC_ONLY', 'changes': changes, 'limitation': 'TorchDispatchMode can force compile eager fallback; these are supplemental dtype observations, never actual compiled-path numeric evidence.'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['baseline', 'record', 'time', 'profile', 'check'])
    p.add_argument('--config', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--tree', help='selected tree for record/profile; baseline defaults to reference')
    p.add_argument('--candidate', help='candidate tree for B-C-B time/check')
    p.add_argument('--case', help='one frozen case; omit for all nine')
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['forward', 'forward-backward', 'step'], default='step')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--detailed', action='store_true')
    p.add_argument('--profile-path', choices=['timing', 'record'], default='timing', help='profile actual timing path or untraced primary Recorder path in a separate fresh process')
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--thresholds', help='frozen thresholds.json')
    p.add_argument('--reference-records', help='frozen r0/records')
    p.add_argument('--skip-timing', action='store_true')
    p.add_argument('--formulation', help='frozen formulation.json; mandatory for check')
    a = p.parse_args()
    c = config(a.config)
    from runtime import configure_runtime
    configure_runtime(c)
    cases = cases_from(a)
    if not cases:
        p.error('no selected frozen case')
    if a.action in ['time', 'check'] and not a.candidate:
        p.error('--candidate required')
    if a.action == 'check' and (not a.thresholds or not a.reference_records or a.case or a.smoke or a.mode != 'step'):
        p.error('check requires complete cases, original step, frozen --thresholds and --reference-records')
    if a.action == 'check' and not a.formulation:
        p.error('check requires --formulation to verify evaluator and task hashes')
    if a.formulation:
        frozen = verify_manifest(Path(__file__).resolve().parents[1], a.formulation)
        for name in ['config', 'cases', 'thresholds', 'reference_records']:
            value = getattr(a, name, None)
            if value and str(Path(value).resolve()) != frozen['roles'][name]:
                raise ValueError('CLI points outside frozen input role: ' + name)
    out = new_output(a.out)
    if a.action == 'baseline':
        if len(cases) != 9 or a.case:
            p.error('baseline formulation requires all nine cases')
        for i in range(4): record_set(a, c, cases, c['reference'], out / f'r{i}', dtype_supplement=i == 0)
        record_set(a, c, cases, c['reference'], out / 'lr_control', control=True)
        write_json(out / 'baseline_complete.json', {'independent_records': 4, 'cases': len(cases), 'actual_lr_control': 1.02, 'primary_dtype_trace': False, 'r0_fresh_process_dtype_supplement': True})
    elif a.action == 'record':
        record_set(a, c, cases, a.tree or c['reference'], out / 'record_set', dtype_supplement=True)
    elif a.action == 'profile':
        for case in cases: launch(a, c, case, a.tree or c['reference'], 'profile', out / case['name'])
    elif a.action in ['time', 'check']:
        timings = {}
        if not a.skip_timing:
            for case in cases:
                arms = [launch(a, c, case, tree, 'time', out / f'{case["name"]}_{label}') for label, tree in [('b1', c['reference']), ('c', a.candidate), ('b2', c['reference'])]]
                timings[case['name']] = compare_timing_runs(*arms)
            write_json(out / 'timing_comparison.json', timings)
        if a.action == 'check':
            # Shared-machine benefit is intentionally unavailable. Continue
            # numerical diagnosis by explicit task policy; never issue PASS.
            records = record_set(a, c, cases, a.candidate, out / 'candidate_records', dtype_supplement=True)
            result = strict_compare(a.reference_records, records, json.loads(Path(a.thresholds).read_text()))
            dtype_result = compare_dtype_supplements(a.reference_records, records)
            if dtype_result['status'] == 'INCORRECT':
                result['status'] = 'INCORRECT'
                result['dtype_changes'].extend(dtype_result.get('changes', []))
                result['measured_tier'] = 'N2'
            write_json(out / 'numerics_comparison.json', result)
            write_json(out / 'dtype_supplement_comparison.json', dtype_result)
            write_json(out / 'check.json', {'status': 'INCORRECT' if result['status'] != 'PASS' else 'UNAVAILABLE', 'reason': 'numerics failed' if result['status'] != 'PASS' else 'formal benefit unavailable; full native quality and actual fast-path profile review still required', 'numerics_status': result['status'], 'measured_tier': result['measured_tier'], 'timing': timings, 'dtype_supplement': dtype_result, 'primary_record_dtype_trace': False, 'actual_fast_path_profile_proof': 'UNVERIFIED', 'compile_candidate_acceptance': 'PROHIBITED until a reviewer verifies actual fast-path kernels in BOTH --profile-path record and --profile-path timing traces for these exact source/binding/cases. Compiler counters, traced dtype values, or an eager numeric PASS are insufficient.'})
            if result['status'] != 'PASS': sys.exit(2)
    if a.formulation:
        verify_manifest(Path(__file__).resolve().parents[1], a.formulation)

if __name__ == '__main__':
    main()
