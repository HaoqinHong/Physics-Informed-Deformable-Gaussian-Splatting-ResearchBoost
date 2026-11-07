import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import textwrap
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from artifact_checks import NATIVE_NAMES, ROW_BYTES, _file_sha256, verify_recovered_manifest
from common import sha256, write_json
from recover_postcheck import archived_checker_identity, checker_log_correspondence, complete_native_log, rehash_existing_artifacts

class RecoveryContracts(unittest.TestCase):
    def test_only_known_archived_checker_and_corresponding_log_allowed(self):
        root = Path(__file__).resolve().parents[1]
        current = root / 'bench/native_entry.py'
        legacy = archived_checker_identity(root / 'prep/native_entry.before_stream_checker.py', current)
        stream = archived_checker_identity(root / 'prep/native_entry.before_cpu_subprocess.py', current)
        self.assertEqual(legacy['version'], 'legacy_inprocess_plyfile_postcheck')
        self.assertEqual(stream['version'], 'stream_postcheck_inside_training_cuda_process')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log'; path.write_text('Training complete.\n')
            checker_log_correspondence(legacy, path)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                checker_log_correspondence(stream, path)
            path.write_text('Training complete.\nArtifact validation: checkpoint 100 (CPU, mmap=False)\n')
            checker_log_correspondence(stream, path)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                checker_log_correspondence(legacy, path)
            fake = Path(tmp) / 'not_native.py'; fake.write_text('unrecognized')
            with self.assertRaisesRegex(ValueError, 'allowlisted'):
                archived_checker_identity(fake, current)
    def test_native_postcheck_exec_is_cpu_only_and_does_not_publish_failure(self):
        root = Path(__file__).resolve().parents[1]
        entry = root / 'bench/native_entry.py'
        source = entry.read_text().split('    # Exit zero alone does not establish a completed or finite training run.\n')[1].split('\nif __name__')[0]
        postcheck = compile(textwrap.dedent(source), str(entry), 'exec')
        for success in [True, False]:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                def fake_cpu_child(command, *, cwd, env, log, timeout):
                    self.assertEqual(command[1:3], ['-X', 'faulthandler'])
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                    self.assertEqual(timeout, 7200)
                    self.assertEqual(command[-1], str(out / 'native_artifacts.json'))
                    write_json(out / 'native_artifacts.json', {'schema': 'gs3d-native-artifacts@1', 'validation_complete': success})
                    return {'command': command, 'returncode': 0}
                context = {'__file__': str(entry), 'args': SimpleNamespace(out=tmp), 'Path': Path, 'sys': SimpleNamespace(executable='/cpu/python'), 'write_json': write_json}
                with patch('common.bounded_run', side_effect=fake_cpu_child):
                    if success:
                        exec(postcheck, context)
                        receipt = __import__('json').loads((out / 'native_complete.json').read_text())
                        self.assertTrue(receipt['postcheck_cpu_process']['fresh_exec'])
                        self.assertEqual(receipt['postcheck_cpu_process']['CUDA_VISIBLE_DEVICES'], '')
                    else:
                        with self.assertRaisesRegex(ValueError, 'did not produce complete'):
                            exec(postcheck, context)
                        self.assertFalse((out / 'native_complete.json').exists())
    def test_full_native_log_required_in_original_order(self):
        markers = ['[ITER 100] Saving Checkpoint', '[ITER 7000] Evaluating test: L1 0.1 PSNR 25', '[ITER 7000] Saving Gaussians', '[ITER 7000] Saving Checkpoint', '[ITER 29900] Saving Checkpoint', '[ITER 30000] Evaluating test: L1 0.05 PSNR 30', '[ITER 30000] Saving Gaussians', '[ITER 30000] Saving Checkpoint', 'Training complete.']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.log'
            path.write_text('\n'.join(markers))
            self.assertTrue(complete_native_log(path)['training_complete_before_postcheck_failure'])
            for invalid in [markers[:-1], markers[:5] + markers[6:], list(reversed(markers))]:
                path.write_text('\n'.join(invalid))
                with self.assertRaises(ValueError): complete_native_log(path)
    def test_cached_validation_reuse_rehashes_all_six_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model'; model.mkdir()
            report = {'schema': 'gs3d-native-artifacts@1', 'validation_complete': True, 'checkpoints': {}, 'ply': {}}
            for group, iterations in [('checkpoints', [100, 7000, 29900, 30000]), ('ply', [7000, 30000])]:
                for iteration in iterations:
                    path = model / (f'chkpnt{iteration}.pth' if group == 'checkpoints' else f'point_cloud/iteration_{iteration}/point_cloud.ply')
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f'previously strictly checked {group}/{iteration}'.encode())
                    checksum, identity = _file_sha256(path)
                    item = {'path': str(path), 'sha256': checksum, 'file_identity': identity, 'finite': True, 'mmap': False}
                    item.update({'iteration': iteration, 'gaussians': 7} if group == 'checkpoints' else {'property_names': list(NATIVE_NAMES), 'property_count': 62, 'row_bytes': ROW_BYTES, 'format': 'binary_little_endian 1.0', 'scalar_dtype': '<f4', 'expected_file_bytes': identity['size_bytes'], 'all_62_fields_single_pass': True, 'capture_bitwise_equal': iteration == 30000, 'vertex_count': 7 if iteration == 30000 else 6})
                    report[group][str(iteration)] = item
            self.assertEqual(len(rehash_existing_artifacts(report, model)), 6)
            (model / 'chkpnt7000.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes or model identity changed'):
                rehash_existing_artifacts(report, model)
    def test_proof_cannot_replace_or_edit_original_error_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {'status': 'error', 'error': 'child failed (-7)'}
            write_json(root / 'full_train.json', original)
            for name in ['train.log', 'native_binding.json']: (root / name).write_text(name)
            artifacts = {'validation_complete': True, 'checkpoints': {str(i): {} for i in [100, 7000, 29900, 30000]}, 'ply': {'7000': {}, '30000': {'capture_bitwise_equal': True}}}
            write_json(root / 'artifacts.json', artifacts)
            recovered = {'status': 'complete_postcheck_recovered', 'original_exit_success': False, 'scene': 'bicycle', 'seed': 0, 'config_sha256': 'config', 'data_sha256': 'data', 'source': {'sha256': 'source'}, 'original_manifest_sha256': sha256(root / 'full_train.json')}
            binding = {key: recovered[key] for key in ['scene', 'seed', 'config_sha256', 'data_sha256', 'original_manifest_sha256']}; binding['source_sha256'] = 'source'
            proof = {'status': 'strict_postcheck_passed', 'binding': binding, 'original_exit_success': False, 'artifacts_path': str(root / 'artifacts.json'), 'artifacts_sha256': sha256(root / 'artifacts.json'), 'input_hashes': {name: sha256(root / name) for name in ['train.log', 'native_binding.json']}}
            write_json(root / 'proof.json', proof)
            recovered.update(recovery_proof_path=str(root / 'proof.json'), recovery_proof_sha256=sha256(root / 'proof.json'))
            write_json(root / 'full_train.recovered.json', recovered)
            verify_recovered_manifest(root / 'full_train.recovered.json')
            self.assertEqual((root / 'full_train.json').read_text().find('"status": "error"') >= 0, True)
            (root / 'proof.json').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'proof hash mismatch'):
                verify_recovered_manifest(root / 'full_train.recovered.json')

if __name__ == '__main__': unittest.main()
