"""Run a command on exactly one explicitly selected GPU, with a shared file lock."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys


def environment(uuid, arch, environ):
    result = dict(environ)
    result.update(CUDA_VISIBLE_DEVICES=uuid, TORCH_CUDA_ARCH_LIST=arch,
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                  MAX_JOBS='4', PYTHONDONTWRITEBYTECODE='1')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True, help='one allocated GPU index or UUID')
    p.add_argument('--arch', default='8.0', help='A800 is 8.0; explicitly change for other architectures')
    p.add_argument('--lock', type=Path, help='use the existing deployment lock when one is already in service')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('a command is required after --')
    uuid = subprocess.check_output(['nvidia-smi', '-i', a.gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if not uuid.startswith('GPU-') or len(uuid.splitlines()) != 1 or ',' in uuid:
        raise ValueError('exactly one GPU UUID required')
    lock = a.lock or Path('/tmp') / f'researchboost-{os.getuid()}-{uuid}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        raise SystemExit(subprocess.call(command, env=environment(uuid, a.arch, os.environ)))


if __name__ == '__main__':
    main()
