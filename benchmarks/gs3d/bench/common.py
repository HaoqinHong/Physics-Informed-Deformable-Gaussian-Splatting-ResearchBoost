"""CPU-only contract and launch helpers for the frozen original-2023 3DGS bench."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

IGNORED_DIRS = {'.git', '__pycache__', 'build', 'dist', '.pytest_cache', '.idea', '.vscode'}
SOURCE_SUFFIXES = {'.py', '.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.hpp', '.inl', '.json', '.toml', '.yaml', '.yml', '.so'}
ALLOWED_PREFIXES = ('gaussian_renderer/', 'submodules/diff-gaussian-rasterization/')
ALLOWED_FILES = {'utils/loss_utils.py'}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)

def config(path):
    c = json.loads(Path(path).read_text())
    if c['schedule'] != {'iterations': 30000, 'test': [7000, 30000], 'save': [7000, 30000], 'checkpoint': [100, 7000, 29900, 30000]}:
        raise ValueError('native schedule contract changed')
    if c['seeds'] != [0, 1] or c['precision'] != 'upstream':
        raise ValueError('seed/precision contract changed')
    if set(c['scenes']) != {'bonsai', 'bicycle', 'garden'}:
        raise ValueError('scene contract changed')
    for name, image_dir in [('bonsai', 'images_2'), ('bicycle', 'images_4'), ('garden', 'images_4')]:
        if c['scenes'][name]['images'] != image_dir:
            raise ValueError('official image scale contract changed')
    return c

def sources(tree):
    tree = Path(tree).resolve()
    result = {}
    for root, dirs, files in os.walk(tree):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.endswith('.egg-info'))
        for name in sorted(files):
            p = Path(root, name)
            if p.suffix in SOURCE_SUFFIXES or name in {'CMakeLists.txt', 'Makefile'}:
                if not p.resolve().is_relative_to(tree):
                    raise ValueError(f'source escapes tree: {p}')
                result[p.relative_to(tree).as_posix()] = sha256(p)
    return result

def assert_source_contract(tree, reference):
    current, original = sources(tree), sources(reference)
    changes = [p for p in sorted(set(current) | set(original)) if current.get(p) != original.get(p)]
    denied = [p for p in changes if p == 'gaussian_renderer/network_gui.py' or (p not in ALLOWED_FILES and not p.startswith(ALLOWED_PREFIXES))]
    if denied:
        raise ValueError('source outside allowed implementation scope changed: ' + ', '.join(denied))
    return {'files': current, 'sha256': digest(current), 'changed': changes}

def data_files(c, scene):
    s = c['scenes'][scene]
    root = Path(s['path']).resolve()
    # Native reader writes this cache if absent; prevent any frozen-data mutation.
    if not (root / 'sparse/0/points3D.ply').is_file():
        raise ValueError('pre-create native sparse/0/points3D.ply before freezing data')
    files = {}
    for sub in (s['images'], 'sparse'):
        for p in sorted((root / sub).rglob('*')):
            if p.is_file():
                if not p.resolve().is_relative_to(root):
                    raise ValueError(f'data symlink escapes scene: {p}')
                files[p.relative_to(root).as_posix()] = sha256(p)
    if not any(k.startswith(s['images'] + '/') for k in files):
        raise ValueError('official images missing')
    return files

def assert_data(c, scene):
    manifest_path = Path(c['data_manifest'])
    manifest = json.loads(manifest_path.read_text())
    if sources(c['reference']) != manifest['reference_sources']:
        raise ValueError('frozen reference source changed')
    actual = data_files(c, scene)
    if actual != manifest['scenes'][scene]:
        raise ValueError(f'frozen scene data changed: {scene}')
    return digest(actual)

def new_output(path):
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=False)
    (p / '.gs3d_bench_owned').write_text('original3dgs frozen bench\n')
    return p

def child_environment(out):
    env = os.environ.copy()
    cache = Path(out) / 'cache'
    cache.mkdir()
    for key, sub in [('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda'), ('TORCH_EXTENSIONS_DIR', 'extensions')]:
        env[key] = str(cache / sub)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env

def bounded_run(command, *, cwd, env, log, timeout):
    started = time.time()
    with open(log, 'w') as stream:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code:
        raise RuntimeError(f'child failed ({code}); see {log}')
    return {'command': command, 'started_unix': started, 'ended_unix': time.time(), 'returncode': code}

def bind_tree(tree, c):
    """Call only inside the GPU child, before importing any upstream package."""
    tree = Path(tree).resolve()
    roots = [tree] + [tree / r for r in c['extension_roots']]
    for p in roots:
        if not p.is_dir():
            raise ValueError(f'tree-local import root absent: {p}')
    sys.path[:0] = [str(p) for p in roots]
    import torch
    import diff_gaussian_rasterization
    import diff_gaussian_rasterization._C as raster_c
    import simple_knn._C as knn_c
    modules = {'raster_python': diff_gaussian_rasterization, 'raster_binary': raster_c, 'knn_binary': knn_c}
    binding = {}
    for name, mod in modules.items():
        p = Path(mod.__file__).resolve()
        if not p.is_relative_to(tree):
            raise ValueError(f'{name} imported outside selected tree: {p}')
        binding[name] = {'path': str(p), 'sha256': sha256(p)}
    if torch.cuda.device_count() != 1:
        raise ValueError('exactly one assigned GPU must be visible')
    binding['torch'] = torch.__version__
    binding['cuda'] = torch.version.cuda
    binding['device_name'] = torch.cuda.get_device_name(0)
    binding['math'] = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'default_dtype': str(torch.get_default_dtype())}
    return binding
