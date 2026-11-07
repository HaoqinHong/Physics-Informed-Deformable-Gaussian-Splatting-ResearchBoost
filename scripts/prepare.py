"""Fetch pinned public sources and create separate compatible reference/candidate trees."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'pidg': ('SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting', 'b77fd2107a50c875d7219bb4f82388d5e5ea1319'),
    'flow-raster': ('HaoqinHong/flow-based-diff-gaussian-rasterization', '2146a1c0dc3558644af420f20c21a0ad07ffa67a'),
    'gs3d': ('graphdeco-inria/gaussian-splatting', 'b2ada78a779ba0455dfdc2b718bdf1726b05a1b6'),
    'raster': ('graphdeco-inria/diff-gaussian-rasterization', '8064f52ca233942bdec2d1a1451c026deedd320b'),
    'knn': ('camenduru/simple-knn', '44f764299fa305faf6ec5ebd99939e0508331503'),
    'glm': ('g-truc/glm', '5c46b9c07008ae65cb81ab79cd677ecc1934b903'),
}
IGNORE = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '*.egg-info', 'build', 'tmp_build')


def fetch(name, cache):
    repo, commit = PINS[name]
    archive = cache / (name + '-' + commit + '.tar.gz')
    cache.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
        with urllib.request.urlopen(url, timeout=180) as stream:
            blob = stream.read()
        temporary = archive.with_suffix('.part')
        temporary.write_bytes(blob)
        temporary.replace(archive)
    target = cache / (name + '-' + commit)
    if not target.exists():
        target.mkdir()
        with tarfile.open(archive) as package:
            for member in package:
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                if '..' in parts or Path(*parts).is_absolute() or not (member.isfile() or member.isdir()):
                    raise ValueError('unexpected archive member: ' + member.name)
                path = target.joinpath(*parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(package.extractfile(member).read())
    return target


def overlay(source, destination):
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError('nonempty dependency destination: ' + str(destination))
        destination.rmdir()
    shutil.copytree(source, destination, ignore=IGNORE)


def apply(tree, name):
    with (ROOT / 'patches' / name).open('rb') as patch:
        subprocess.run(['patch', '--batch', '--forward', '-p1'], cwd=tree, stdin=patch, check=True)


def prepare(experiment, workspace, get_source, include_rejected=False):
    target = workspace / experiment
    target.mkdir(parents=True, exist_ok=False)
    base, candidate = target / 'baseline', target / 'candidate'
    shutil.copytree(get_source(experiment), base, ignore=IGNORE)
    if experiment == 'pidg':
        overlay(get_source('flow-raster'), base / 'submodules/flow-based-diff-gaussian-rasterization')
        apply(base, 'pidg-compat.patch')
        apply(base, 'pidg-build.patch')
    else:
        overlay(get_source('raster'), base / 'submodules/diff-gaussian-rasterization')
        overlay(get_source('knn'), base / 'submodules/simple-knn')
        overlay(get_source('glm'), base / 'submodules/diff-gaussian-rasterization/third_party/glm')
        apply(base, 'gs3d-compat.patch')
    shutil.copytree(base, candidate, ignore=IGNORE)
    apply(candidate, 'pidg-flow.patch' if experiment == 'pidg' else 'gs3d-fullwarp-experimental.patch')
    if experiment == 'pidg' and include_rejected:
        rejected = target / 'rejected-both'
        shutil.copytree(candidate, rejected, ignore=IGNORE)
        apply(rejected, 'pidg-raster-rejected.patch')
    receipt = {'experiment': experiment, 'source_pins': PINS,
               'baseline': str(base.resolve()), 'candidate': str(candidate.resolve()),
               'status': 'sources_prepared_not_built_or_benchmarked',
               'numerical_acceptance': 'must be measured in the new environment'}
    (target / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['pidg', 'gs3d'])
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    p.add_argument('--include-rejected', action='store_true')
    a = p.parse_args()
    workspace = a.workspace.resolve()
    target = prepare(a.experiment, workspace, lambda name: fetch(name, workspace / 'source-cache'), a.include_rejected)
    print(target)


if __name__ == '__main__':
    main()
