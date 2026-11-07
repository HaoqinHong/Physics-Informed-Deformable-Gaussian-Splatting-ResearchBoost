"""CPU-only, sequential native-3DGS artifact validation; never mmap a model PLY."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path

CHUNK_BYTES = 16 << 20
NATIVE_NAMES = tuple(['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)] + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
ROW_BYTES = 4 * len(NATIVE_NAMES)

def _file_identity(stat):
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

def parse_native_header(stream):
    """Strict scalar-float, binary_little_endian, vertex-only, SH-degree-3 schema."""
    header = bytearray()
    lines = []
    while True:
        line = stream.readline(65537)
        header.extend(line)
        if not line or len(header) > 65536:
            raise ValueError('missing/oversized native PLY header')
        try:
            text = line.decode('ascii').rstrip('\r\n')
        except UnicodeDecodeError as error:
            raise ValueError('native PLY header must be ASCII') from error
        lines.append(text)
        if text == 'end_header': break
    if lines[:2] != ['ply', 'format binary_little_endian 1.0']:
        raise ValueError('native PLY requires binary_little_endian 1.0')
    elements, names = [], []
    for line in lines[2:-1]:
        fields = line.split()
        if fields and fields[0] == 'comment': continue
        if len(fields) == 3 and fields[:2] == ['element', 'vertex'] and not elements and not names:
            try: count = int(fields[2])
            except ValueError as error: raise ValueError('invalid native vertex count') from error
            if count <= 0: raise ValueError('native vertex count must be positive')
            elements.append(count)
        elif len(fields) == 3 and fields[:2] == ['property', 'float'] and len(elements) == 1:
            names.append(fields[2])
        else:
            raise ValueError('unexpected native PLY element/property/header: ' + line)
    if len(elements) != 1 or tuple(names) != NATIVE_NAMES:
        raise ValueError('native PLY requires exactly 62 ordered scalar-float properties and one vertex element')
    info = {'vertex_count': elements[0], 'property_names': list(names), 'property_count': len(names), 'format': 'binary_little_endian 1.0', 'scalar_dtype': '<f4', 'row_bytes': ROW_BYTES, 'header_bytes': len(header), 'header_sha256': hashlib.sha256(header).hexdigest(), 'expected_file_bytes': len(header) + elements[0] * ROW_BYTES}
    return info, bytes(header)

def _capture_shape_check(capture):
    import numpy as np
    if len(capture) != 12: raise ValueError('native capture must contain 12 entries')
    count = int(capture[1].shape[0])
    shapes = {1: (count, 3), 2: (count, 1, 3), 3: (count, 15, 3), 4: (count, 3), 5: (count, 4), 6: (count, 1), 7: (count,), 8: (count, 1), 9: (count, 1)}
    for index, shape in shapes.items():
        value = capture[index]
        if tuple(value.shape) != shape or value.device.type != 'cpu' or value.detach().numpy().dtype != np.dtype('<f4'):
            raise ValueError(f'native capture entry {index} has unexpected shape/device/precision')
    if count <= 0: raise ValueError('empty native Gaussian model')
    return count

def _capture_rows(capture, first, last):
    import numpy as np
    result = np.zeros((last - first, len(NATIVE_NAMES)), dtype='<f4')
    def array(index): return capture[index][first:last].detach().numpy()
    result[:, 0:3] = array(1)
    result[:, 6:9] = array(2).transpose(0, 2, 1).reshape(last - first, 3)
    result[:, 9:54] = array(3).transpose(0, 2, 1).reshape(last - first, 45)
    result[:, 54:55] = array(6)
    result[:, 55:58] = array(4)
    result[:, 58:62] = array(5)
    return result

def scan_native_ply(path, *, capture=None, collect=False):
    """One sequential pass computes SHA and finite checks over all 62 fields.

    collect=True additionally keeps one in-memory structured vertex array for
    the original GaussianModel loader. capture is allowed only for the 30k
    native PLY, whose save and checkpoint share the final (no-Adam) state.
    """
    import numpy as np
    path = Path(path).resolve()
    with open(path, 'rb', buffering=CHUNK_BYTES) as stream:
        before = _file_identity(os.fstat(stream.fileno()))
        info, header = parse_native_header(stream)
        if before['size_bytes'] != info['expected_file_bytes']:
            raise ValueError('native PLY exact file size/count/schema mismatch')
        count = info['vertex_count']
        if capture is not None and _capture_shape_check(capture) != count:
            raise ValueError('30k PLY/checkpoint Gaussian count differs')
        data = np.empty((count, len(NATIVE_NAMES)), dtype='<f4') if collect else None
        checksum = hashlib.sha256(header)
        rows_per_chunk = max(1, CHUNK_BYTES // ROW_BYTES)
        chunks = 0
        for first in range(0, count, rows_per_chunk):
            last = min(count, first + rows_per_chunk)
            raw = stream.read((last - first) * ROW_BYTES)
            if len(raw) != (last - first) * ROW_BYTES:
                raise ValueError('short sequential native PLY read')
            checksum.update(raw)
            block = np.frombuffer(raw, dtype='<f4').reshape(last - first, len(NATIVE_NAMES))
            finite = np.isfinite(block)
            if not finite.all():
                row, field = np.argwhere(~finite)[0]
                raise ValueError(f'nonfinite native PLY vertex {first + int(row)} property {NATIVE_NAMES[int(field)]}')
            if capture is not None:
                expected = _capture_rows(capture, first, last)
                if not np.array_equal(block.view('<u4'), expected.view('<u4')):
                    row, field = np.argwhere(block.view('<u4') != expected.view('<u4'))[0]
                    raise ValueError(f'30k PLY/capture bit mismatch at vertex {first + int(row)} property {NATIVE_NAMES[int(field)]}')
            if collect: data[first:last] = block
            chunks += 1
        if stream.read(1): raise ValueError('native PLY has trailing payload')
        if _file_identity(os.fstat(stream.fileno())) != before:
            raise ValueError('native PLY changed during sequential validation')
    report = {**info, 'path': str(path), 'file_identity': before, 'sha256': checksum.hexdigest(), 'finite': True, 'sequential_chunk_bytes': CHUNK_BYTES, 'chunks': chunks, 'mmap': False, 'all_62_fields_single_pass': True, 'capture_bitwise_equal': capture is not None}
    if collect:
        structured = data.view(np.dtype([(name, '<f4') for name in NATIVE_NAMES])).reshape(count)
        return report, structured
    return report

def read_native_ply(path):
    """Return genuine PlyData/PlyElement backed by sequentially read RAM data."""
    from plyfile import PlyData, PlyElement
    report, vertices = scan_native_ply(path, collect=True)
    return PlyData([PlyElement.describe(vertices, 'vertex')], text=False, byte_order='<'), report

@contextmanager
def scoped_gaussian_ply_reader(gaussian_module, allowed_paths):
    """Only intercept the two explicitly selected saved model PLY paths."""
    allowed = {str(Path(path).resolve()) for path in allowed_paths}
    original = gaussian_module.PlyData
    receipts = {}
    class ScopedPlyData:
        @staticmethod
        def read(path, *args, **kwargs):
            resolved = str(Path(path).resolve())
            if resolved not in allowed:
                return original.read(path, *args, **kwargs)
            value, report = read_native_ply(resolved)
            receipts[resolved] = report
            return value
    gaussian_module.PlyData = ScopedPlyData
    try:
        yield receipts
    finally:
        gaussian_module.PlyData = original

def _file_sha256(path):
    checksum = hashlib.sha256()
    with open(path, 'rb', buffering=CHUNK_BYTES) as stream:
        before = _file_identity(os.fstat(stream.fileno()))
        for block in iter(lambda: stream.read(CHUNK_BYTES), b''): checksum.update(block)
        if _file_identity(os.fstat(stream.fileno())) != before: raise ValueError('artifact changed while hashing')
    return checksum.hexdigest(), before

def validate_checkpoint(path, iteration):
    import numpy as np
    import torch
    checksum, identity = _file_sha256(path)
    capture, actual = torch.load(path, map_location='cpu', weights_only=False, mmap=False)
    if actual != iteration: raise ValueError('saved checkpoint iteration mismatch')
    count = _capture_shape_check(capture)
    counters = {'tensors': 0, 'tensor_elements': 0, 'floating_scalars': 0}
    def finite(value):
        if torch.is_tensor(value):
            if value.device.type != 'cpu' or not torch.isfinite(value).all().item():
                raise ValueError(f'nonfinite/non-CPU saved checkpoint {iteration}')
            counters['tensors'] += 1; counters['tensor_elements'] += value.numel()
        elif isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value: finite(item)
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value): raise ValueError('nonfinite checkpoint scalar')
            counters['floating_scalars'] += 1
    finite(capture)
    if _file_identity(Path(path).stat()) != identity: raise ValueError('checkpoint changed during validation')
    return capture, {'path': str(Path(path).resolve()), 'iteration': iteration, 'gaussians': count, 'finite': True, 'sha256': checksum, 'file_identity': identity, 'mmap': False, **counters}

def validate_native_artifacts(model_path):
    import torch
    model = Path(model_path).resolve()
    result = {'schema': 'gs3d-native-artifacts@1', 'checkpoints': {}, 'ply': {}, 'torch': torch.__version__, 'validation_complete': False}
    final_capture = None
    for iteration in [100, 7000, 29900, 30000]:
        print(f'Artifact validation: checkpoint {iteration} (CPU, mmap=False)', flush=True)
        capture, report = validate_checkpoint(model / f'chkpnt{iteration}.pth', iteration)
        result['checkpoints'][str(iteration)] = report
        if iteration == 30000: final_capture = capture
        del capture
        gc.collect()
    for iteration in [7000, 30000]:
        print(f'Artifact validation: saved PLY {iteration} (16MiB sequential, all 62 fields)', flush=True)
        path = model / 'point_cloud' / f'iteration_{iteration}/point_cloud.ply'
        result['ply'][str(iteration)] = scan_native_ply(path, capture=final_capture if iteration == 30000 else None)
        result['ply'][str(iteration)]['native_relation'] = 'bitwise equal to final capture, including normals/DC/rest transpose-flatten order' if iteration == 30000 else 'saved before native densification/Adam; deliberately not compared to post-event 7k checkpoint count or values'
    result['validation_complete'] = True
    return result

def verify_recovered_manifest(path):
    """Verify immutable proof linkage; caller separately binds source/data/config."""
    from common import sha256
    path = Path(path).resolve()
    recovered = json.loads(path.read_text())
    if recovered['status'] != 'complete_postcheck_recovered' or recovered.get('original_exit_success') is not False:
        raise ValueError('not an explicit postcheck-only recovery manifest')
    original = path.parent / 'full_train.json'
    if sha256(original) != recovered['original_manifest_sha256'] or json.loads(original.read_text())['status'] != 'error':
        raise ValueError('original error manifest changed or was overwritten')
    proof_path = Path(recovered['recovery_proof_path']).resolve()
    if sha256(proof_path) != recovered['recovery_proof_sha256']: raise ValueError('recovery proof hash mismatch')
    proof = json.loads(proof_path.read_text())
    expected = {key: recovered[key] for key in ['scene', 'seed', 'config_sha256', 'data_sha256', 'original_manifest_sha256']}
    expected['source_sha256'] = recovered['source']['sha256']
    if proof['status'] != 'strict_postcheck_passed' or proof['binding'] != expected or proof['original_exit_success'] is not False:
        raise ValueError('recovery proof does not bind this exact recovered run')
    artifacts_path = Path(proof['artifacts_path']).resolve()
    if sha256(artifacts_path) != proof['artifacts_sha256']: raise ValueError('recovered artifact proof hash mismatch')
    artifacts = json.loads(artifacts_path.read_text())
    if not artifacts['validation_complete'] or set(artifacts['checkpoints']) != {'100', '7000', '29900', '30000'} or set(artifacts['ply']) != {'7000', '30000'}:
        raise ValueError('incomplete recovered artifact proof')
    if not artifacts['ply']['30000']['capture_bitwise_equal']: raise ValueError('recovery lacks final PLY/capture bitwise proof')
    for name in ['train.log', 'native_binding.json']:
        if sha256(path.parent / name) != proof['input_hashes'][name]: raise ValueError('recovery input changed: ' + name)
    return recovered, artifacts

def selftest():
    import tempfile
    import numpy as np
    from plyfile import PlyData, PlyElement
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory(prefix='gs3d-ply-selftest-') as directory:
        root = Path(directory)
        rows = np.arange(7 * 62, dtype='<f4').reshape(7, 62) / np.float32(17)
        rows[0, 0] = np.float32(-0.)
        structured = rows.view(np.dtype([(name, '<f4') for name in NATIVE_NAMES])).reshape(7)
        valid = root / 'native.ply'
        PlyData([PlyElement.describe(structured, 'vertex')], text=False, byte_order='<').write(valid)
        original = PlyData.read(valid, mmap='c')
        streamed, receipt = read_native_ply(valid)
        assert original.elements[0].data.tobytes() == streamed.elements[0].data.tobytes()
        assert [p.name for p in original.elements[0].properties] == list(NATIVE_NAMES)
        assert receipt['sha256'] == hashlib.sha256(valid.read_bytes()).hexdigest()
        module = SimpleNamespace(PlyData=PlyData)
        with scoped_gaussian_ply_reader(module, [valid]) as receipts:
            assert module.PlyData.read(valid).elements[0].data.tobytes() == structured.tobytes()
            assert str(valid.resolve()) in receipts
        assert module.PlyData is PlyData
        for label, payload in [('truncated', valid.read_bytes()[:-1]), ('extra', valid.read_bytes() + b'x'), ('schema', valid.read_bytes().replace(b'property float x\n', b'property float q\n', 1))]:
            path = root / (label + '.ply'); path.write_bytes(payload)
            try: scan_native_ply(path)
            except ValueError: pass
            else: raise AssertionError(label + ' was accepted')
        broken = rows.copy(); broken[-1, -1] = np.nan
        nan_path = root / 'nan.ply'
        PlyData([PlyElement.describe(broken.view(structured.dtype).reshape(7), 'vertex')], text=False, byte_order='<').write(nan_path)
        try: scan_native_ply(nan_path)
        except ValueError as error: assert 'nonfinite' in str(error)
        else: raise AssertionError('NaN accepted')
    print(json.dumps({'status': 'PASS', 'mmap_vs_stream_all_field_bytes': True, 'scoped_shim': True, 'negative_controls': ['truncated', 'trailing', 'schema', 'NaN']}))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['selftest', 'validate', 'scan-ply'])
    p.add_argument('--model')
    p.add_argument('--ply')
    p.add_argument('--out')
    a = p.parse_args()
    if a.action == 'selftest': selftest(); return
    from common import write_json
    if not a.out: p.error('--out required')
    if Path(a.out).exists(): raise FileExistsError('artifact report must be new')
    if a.action == 'validate':
        if not a.model: p.error('--model required')
        result = validate_native_artifacts(a.model)
    else:
        if not a.ply: p.error('--ply required')
        result = scan_native_ply(a.ply)
    write_json(a.out, result)

if __name__ == '__main__': main()
