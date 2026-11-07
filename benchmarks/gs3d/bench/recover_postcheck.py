"""Explicit CPU-only postcheck recovery. Never rewrite an original error receipt."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from common import assert_data, assert_source_contract, bounded_run, config, digest, new_output, sha256, write_json
from artifact_checks import NATIVE_NAMES, ROW_BYTES, _file_sha256, verify_recovered_manifest

KNOWN_OLD_NATIVE_ENTRIES = {
    '1ffa843221a76568f1f2f0810cfe9bc9e4999844d268e4950a02f8336bcc2f29': 'legacy_inprocess_plyfile_postcheck',
    'da3ca410dc483424f6f794f9d743174e510784acf4364bfe079d3f75cbc25c99': 'stream_postcheck_inside_training_cuda_process'
}
POSTCHECK_MARKER = b'    # Exit zero alone does not establish a completed or finite training run.\n'

def archived_checker_identity(old_entry, current_entry):
    old_entry, current_entry = Path(old_entry).resolve(), Path(current_entry).resolve()
    old_hash = sha256(old_entry)
    if old_hash not in KNOWN_OLD_NATIVE_ENTRIES:
        raise ValueError('archived native wrapper is not a strictly allowlisted prior version')
    original_prefix = old_entry.read_bytes().split(POSTCHECK_MARKER)[0]
    if original_prefix != current_entry.read_bytes().split(POSTCHECK_MARKER)[0]:
        raise ValueError('native CLI/seed/binding/runpy prefix changed')
    return {'archived_path': str(old_entry), 'sha256': old_hash, 'version': KNOWN_OLD_NATIVE_ENTRIES[old_hash], 'unchanged_native_prefix_sha256': hashlib.sha256(original_prefix).hexdigest(), 'current_native_entry_sha256': sha256(current_entry)}

def checker_log_correspondence(identity, log_path):
    marker = 'Artifact validation: checkpoint 100 (CPU, mmap=False)'
    present = marker in Path(log_path).read_text(errors='replace')
    expected = identity['version'] == 'stream_postcheck_inside_training_cuda_process'
    if present != expected:
        raise ValueError('archived checker version does not match original postcheck log evidence')
    return {'stream_inprocess_marker_present': present, 'marker': marker, 'matches_archived_version': True}

def native_argv(tree, c, scene, run):
    item = c['scenes'][scene]
    return [str(Path(tree).resolve() / 'train.py'), '--source_path', item['path'], '--images', item['images'], '--model_path', str(Path(run).resolve() / 'model'), '--eval', '--iterations', '30000', '--port', '0', '--test_iterations', '7000', '30000', '--save_iterations', '7000', '30000', '--checkpoint_iterations', '100', '7000', '29900', '30000']

def complete_native_log(path):
    text = Path(path).read_text(errors='replace')
    markers = ['[ITER 100] Saving Checkpoint', '[ITER 7000] Evaluating test:', '[ITER 7000] Saving Gaussians', '[ITER 7000] Saving Checkpoint', '[ITER 29900] Saving Checkpoint', '[ITER 30000] Evaluating test:', '[ITER 30000] Saving Gaussians', '[ITER 30000] Saving Checkpoint', 'Training complete.']
    positions = [text.find(marker) for marker in markers]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError('native log lacks the complete ordered 30k eval/save/checkpoint/completion events')
    eval_lines = [line for line in text.splitlines() if '[ITER 30000] Evaluating test:' in line]
    if not eval_lines or 'PSNR' not in eval_lines[0] or re.search(r'\b(?:nan|inf|infinity)\b', eval_lines[0], re.IGNORECASE):
        raise ValueError('native 30k test evaluation is missing or nonfinite')
    return {'markers': dict(zip(markers, positions)), 'test_30000_line': eval_lines[0][-1000:], 'training_complete_before_postcheck_failure': True}

def rehash_existing_artifacts(report, model):
    """Reuse strict finite/bitwise validation only when every input byte matches."""
    model = Path(model).resolve()
    if report.get('schema') != 'gs3d-native-artifacts@1' or report.get('validation_complete') is not True or set(report['checkpoints']) != {'100', '7000', '29900', '30000'} or set(report['ply']) != {'7000', '30000'}:
        raise ValueError('cached strict artifact validation is incomplete')
    rehashed = {}
    for group, iterations in [('checkpoints', [100, 7000, 29900, 30000]), ('ply', [7000, 30000])]:
        for iteration in iterations:
            item = report[group][str(iteration)]
            expected = model / (f'chkpnt{iteration}.pth' if group == 'checkpoints' else f'point_cloud/iteration_{iteration}/point_cloud.ply')
            if Path(item['path']).resolve() != expected or item.get('finite') is not True or item.get('mmap') is not False:
                raise ValueError('cached artifact path/finite/reader policy differs')
            if group == 'checkpoints':
                if item['iteration'] != iteration or item['gaussians'] <= 0: raise ValueError('cached checkpoint identity differs')
            else:
                if item['property_names'] != list(NATIVE_NAMES) or item['property_count'] != 62 or item['row_bytes'] != ROW_BYTES or item['format'] != 'binary_little_endian 1.0' or item['scalar_dtype'] != '<f4' or item['expected_file_bytes'] != item['file_identity']['size_bytes'] or not item['all_62_fields_single_pass']:
                    raise ValueError('cached native PLY schema/size/scan differs')
            print(f'Recovery rehash: {group}/{iteration} (CPU sequential, no mmap)', flush=True)
            actual, identity = _file_sha256(expected)
            if actual != item['sha256'] or identity != item['file_identity']:
                raise ValueError('artifact bytes or model identity changed since strict validation: ' + str(expected))
            rehashed[f'{group}/{iteration}'] = {'sha256': actual, 'file_identity': identity}
    if report['ply']['30000'].get('capture_bitwise_equal') is not True or report['ply']['30000']['vertex_count'] != report['checkpoints']['30000']['gaussians']:
        raise ValueError('cached proof lacks final native PLY/capture bitwise equivalence')
    if report['ply']['7000'].get('capture_bitwise_equal') is not False:
        raise ValueError('7k pre-event PLY must not be judged against post-event checkpoint')
    return rehashed

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--tree', required=True)
    p.add_argument('--scene', choices=['bonsai', 'bicycle', 'garden'], required=True)
    p.add_argument('--run', required=True, help='existing completed-training directory with original error full_train.json')
    p.add_argument('--out', required=True, help='new, unique recovery proof directory')
    p.add_argument('--artifacts', help='existing strict artifact_checks.py validate result; all six artifacts are rehashed before reuse')
    p.add_argument('--old-native-entry', required=True, help='exact archived legacy1ffa843... or stream-inprocess da3ca410... native_entry.py')
    p.add_argument('--timeout', type=float, default=7200)
    a = p.parse_args()
    c, run, tree = config(a.config), Path(a.run).resolve(), Path(a.tree).resolve()
    original_path = run / 'full_train.json'
    original_hash = sha256(original_path)
    old = json.loads(original_path.read_text())
    if old['status'] != 'error': raise ValueError('recovery is permitted only for an original error manifest')
    failed_child = re.search(r'child failed \((-?\d+)\)', old.get('error', ''))
    if not failed_child or int(failed_child.group(1)) == 0 or old.get('process', {}).get('returncode') == 0:
        raise ValueError('recovery requires an evidenced failed native postcheck child, not a source/data failure after a successful child')
    original_returncode = int(failed_child.group(1))
    recovered_path = run / 'full_train.recovered.json'
    if recovered_path.exists(): raise FileExistsError('recovery manifest already exists; never overwrite')
    source, data = assert_source_contract(tree, c['reference']), assert_data(c, a.scene)
    if old['source'] != source or old['data_sha256'] != data or old['config_sha256'] != digest(c) or old['scene'] != a.scene or old['seed'] not in c['seeds'] or old['gpu_uuid'] != c['gpu_uuid']:
        raise ValueError('original source/data/config/scene/seed/GPU binding does not match')
    native = json.loads((run / 'native_binding.json').read_text())
    if native['argv'] != native_argv(tree, c, a.scene, run) or native['seed'] != old['seed'] or native['seed_wrapper'] != 'upstream safe_state followed by random/numpy/torch seed; no other patch':
        raise ValueError('original exact CLI or matched seed wrapper differs')
    for name in ['raster_python', 'raster_binary', 'knn_binary']:
        item = native['binding'][name]
        path = Path(item['path']).resolve()
        if not path.is_relative_to(tree) or sha256(path) != item['sha256']:
            raise ValueError('original tree-local implementation binding changed: ' + name)
    old_entry = Path(a.old_native_entry).resolve()
    current_entry = Path(__file__).with_name('native_entry.py')
    checker_identity = archived_checker_identity(old_entry, current_entry)
    log_evidence = complete_native_log(run / 'train.log')
    checker_identity['log_correspondence'] = checker_log_correspondence(checker_identity, run / 'train.log')
    input_hashes = {'full_train.json': original_hash, 'native_binding.json': sha256(run / 'native_binding.json'), 'train.log': sha256(run / 'train.log'), 'old_native_entry.py': sha256(old_entry), 'current_native_entry.py': sha256(current_entry), 'artifact_checks.py': sha256(Path(__file__).with_name('artifact_checks.py'))}
    out = new_output(a.out)
    attempt = {'status': 'running', 'original_exit_success': False, 'original_child_returncode': original_returncode, 'original_error': old.get('error'), 'original_manifest_sha256': original_hash}
    write_json(out / 'recovery_attempt.json', attempt)
    try:
        if a.artifacts:
            artifact_input = Path(a.artifacts).resolve()
            artifact_input_hash = sha256(artifact_input)
            artifacts = json.loads(artifact_input.read_text())
            validation_process = None
        else:
            artifact_input = out / 'strict_artifacts.json'
            environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = ''; environment['PYTHONDONTWRITEBYTECODE'] = '1'
            validation_process = bounded_run([sys.executable, str(Path(__file__).with_name('artifact_checks.py')), 'validate', '--model', str(run / 'model'), '--out', str(artifact_input)], cwd=out, env=environment, log=out / 'cpu_validation.log', timeout=a.timeout)
            artifact_input_hash = sha256(artifact_input)
            artifacts = json.loads(artifact_input.read_text())
        if artifacts['torch'] != native['binding']['torch']:
            raise ValueError('strict artifact validation used a different Torch version')
        rehashed = rehash_existing_artifacts(artifacts, run / 'model')
        if sha256(artifact_input) != artifact_input_hash: raise ValueError('strict artifact result changed during recovery')
        if source != assert_source_contract(tree, c['reference']) or data != assert_data(c, a.scene): raise ValueError('source/data changed during recovery')
        for name in ['full_train.json', 'native_binding.json', 'train.log']:
            if sha256(run / name) != input_hashes[name]: raise ValueError('original recovery input changed: ' + name)
        write_json(out / 'artifacts.json', artifacts)
        binding = {'scene': a.scene, 'seed': old['seed'], 'source_sha256': source['sha256'], 'data_sha256': data, 'config_sha256': digest(c), 'original_manifest_sha256': original_hash}
        proof = {'status': 'strict_postcheck_passed', 'binding': binding, 'original_exit_success': False, 'original_child_returncode': original_returncode, 'original_error': old.get('error'), 'original_native_checker': checker_identity, 'strict_artifact_checker_sha256': input_hashes['artifact_checks.py'], 'recovery_scope': 'Only post-training artifact validation recovered. Original child failure is retained; no claim of original exit 0 or training replay.', 'input_hashes': input_hashes, 'native_argv': native['argv'], 'native_log': log_evidence, 'artifacts_path': str(out / 'artifacts.json'), 'artifacts_sha256': sha256(out / 'artifacts.json'), 'strict_validation_input': {'path': str(artifact_input), 'sha256': artifact_input_hash}, 'validation_process': validation_process, 'rehash_receipts': rehashed, 'checker_mmap': False}
        write_json(out / 'recovery_proof.json', proof)
        recovered = copy.deepcopy(old)
        recovered.update(status='complete_postcheck_recovered', original_status='error', original_exit_success=False, original_child_returncode=original_returncode, original_manifest_sha256=original_hash, recovery_proof_path=str(out / 'recovery_proof.json'), recovery_proof_sha256=sha256(out / 'recovery_proof.json'))
        # A separate filename; the old error receipt is never opened for write.
        if recovered_path.exists(): raise FileExistsError('recovery manifest appeared concurrently')
        write_json(recovered_path, recovered)
        verify_recovered_manifest(recovered_path)
        attempt.update(status='complete_postcheck_recovered', recovered_manifest=str(recovered_path), recovered_manifest_sha256=sha256(recovered_path), recovery_proof_sha256=sha256(out / 'recovery_proof.json'))
        print(json.dumps({'status': attempt['status'], 'manifest': str(recovered_path), 'original_error_receipt_preserved': sha256(original_path) == original_hash}))
    except BaseException as error:
        attempt.update(status='error', error=repr(error)); raise
    finally:
        write_json(out / 'recovery_attempt.json', attempt)

if __name__ == '__main__': main()
