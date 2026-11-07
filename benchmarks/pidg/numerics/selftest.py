"""Mutation tests for the judge on the task's own records (negative self-test evidence for Benchmark acceptance).

Starting from a legitimate variant that must PASS, each mutation injects one known defect into a copy of its
records and checks that the judge reaches the expected verdict. A judge whose failure paths do not fire
(e.g. an arbitration result that never reaches the final status) fails this test.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from . import judge


def _copy(src, dst):
    """Mutable view of a record set: manifests are copied, tensor files are links (a mutation that edits a tensor
    file replaces the link with its own file). Full copies of large-model records would fill the disk."""
    src, dst = Path(src).resolve(), Path(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.suffix == '.json':
            shutil.copyfile(f, dst / f.name)
        else:
            os.symlink(f, dst / f.name)
    return dst


def _first(records):
    case, rank = sorted(records)[0]
    return case, rank, records[(case, rank)]


def _edit_manifest(root, case, rank, fn):
    path = Path(root) / f'{case}.r{rank}.json'
    m = json.loads(path.read_text())
    fn(m)
    path.write_text(json.dumps(m, indent=1) + '\n')


def _edit_tensor(root, case, rank, step, name, fn):
    """Replace a stored tensor (full or sketch) and keep its manifest entry consistent."""
    path = Path(root) / f'{case}.r{rank}.s{step}.pt'
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    old = tensors[name]
    new = fn(old)
    tensors[name] = new
    if path.is_symlink():
        path.unlink()                       # never write through to the source records
    torch.save(tensors, path)
    from .record import tensor_sha

    def update(m):
        d = m['steps'][step]['tensors'][name]
        old_norm = float(old.double().norm())
        new_norm = float(new.double().norm())
        if d.get('stored', 'full') == 'full':
            d['norm'] = new_norm
        elif d['norm'] is not None and old_norm > 0 and new_norm == new_norm:
            d['norm'] = d['norm'] * new_norm / old_norm
        d['dtype'] = str(new.dtype).replace('torch.', '')
        d['sha256'] = tensor_sha(new)
        d['finite'] = bool(torch.isfinite(new).all())
    _edit_manifest(root, case, rank, update)


def _largest(entry, kinds):
    candidates = [(d['norm'] or 0.0, n) for n, d in entry['tensors'].items()
                  if d['kind'] in kinds and d['norm'] and d['shape']]
    return max(candidates)[1] if candidates else None


def _break_cross_rank(mm):
    """Inject the defect the sync declaration forbids: silently synchronize a 'local' model, or desynchronize a
    tensor that is equal across ranks in the reference. Single-rank records get an extra pattern entry."""
    pattern = mm['steps'][0].get('xrank_equal')
    if not pattern:
        mm['steps'][0]['xrank_equal'] = {'__mutation__': False}
        return
    sync = mm.get('sync') or {}
    local = [n for n in pattern if sync.get(judge.model_of(n)) == 'local']
    if local and not all(pattern[n] for n in local):
        for n in local:
            pattern[n] = True
        return
    equal = sorted(n for n in pattern if pattern[n])
    if equal:
        pattern[equal[0]] = False
    else:
        pattern['__mutation__'] = False


def run(ref, variant, thresholds, out=None):
    gates = thresholds['gates']
    records = judge.manifests(variant)
    case, rank, m = _first(records)
    entry = m['steps'][0]
    target = _largest(entry, ('grad',)) or _largest(entry, ('delta', 'optim', 'ema', 'loss', 'out'))
    loss = next((n for n in entry['tensors'] if n.startswith('loss/')), None)
    gate = gates[judge.key(case, rank, 0, target)]['gate'] if target else None
    results = []
    tmp = Path(tempfile.mkdtemp(prefix='numerics-selftest-'))

    def check(label, expect, mutate, golden_fn=None):
        cand = _copy(variant, tmp / label)
        golden = None
        try:
            mutate(cand)
            if golden_fn:
                golden = _copy(ref, tmp / f'{label}-golden')
                golden_fn(golden)
            rep = judge.compare(ref, cand, thresholds, golden=golden)
            got = {'status': rep['status'], 'measured_tier': rep['measured_tier'],
                   'arbitrated': len(rep['arbitration']), 'unresolved': len(rep['unresolved'])}
            ok = all(got.get(k) == v for k, v in expect.items())
        except Exception as error:  # a crash is a failed self-test, not a pass
            got, ok = {'exception': f'{type(error).__name__}: {error}'}, False
        finally:
            shutil.rmtree(tmp / label, ignore_errors=True)
            shutil.rmtree(tmp / f'{label}-golden', ignore_errors=True)
        results.append({'mutation': label, 'expect': expect, 'got': got, 'ok': ok})

    check('unmodified_variant', {'status': 'PASS'}, lambda c: None)
    if target:
        check('gradient_beyond_gate', {'status': 'INCORRECT'},
              lambda c: _edit_tensor(c, case, rank, 0, target, lambda t: t * (1 + 10 * gate + 1e-2)))
    check('call_counter', {'status': 'INCORRECT'},
          lambda c: _edit_manifest(c, case, rank, lambda mm: mm['steps'][0]['counters'].__setitem__(
              'forward/__mutation__', 1)))
    check('rng_stream', {'status': 'INCORRECT'},
          lambda c: _edit_manifest(c, case, rank, lambda mm: mm['steps'][0].__setitem__('rng_after', 'mutated')))
    check('invariant', {'status': 'INCORRECT'},
          lambda c: _edit_manifest(c, case, rank, lambda mm: mm['steps'][0]['invariants'].__setitem__(
              '__mutation__', 1)))
    check('phase', {'status': 'INCORRECT'},
          lambda c: _edit_manifest(c, case, rank, lambda mm: mm['steps'][0].__setitem__('phase', '__mutation__')))
    if loss:
        check('nan_loss', {'status': 'INCORRECT'},
              lambda c: _edit_tensor(c, case, rank, 0, loss, lambda t: torch.full_like(t, float('nan'))))
    if target:
        check('missing_tensor', {'status': 'ERROR'},
              lambda c: _edit_manifest(c, case, rank, lambda mm: mm['steps'][0]['tensors'].pop(target)))
        check('dtype_change', {'measured_tier': 'N2'},
              lambda c: _edit_tensor(c, case, rank, 0, target, lambda t: t.to(torch.bfloat16)))
    check('cross_rank_pattern', {'status': 'INCORRECT'},
          lambda c: _edit_manifest(c, case, rank, _break_cross_rank))
    if target:
        rt = torch.load(Path(ref) / f'{case}.r{rank}.s0.pt', map_location='cpu', weights_only=True)[target].double()
        gen = torch.Generator().manual_seed(0)
        u = torch.randn(rt.shape, generator=gen, dtype=torch.float64)
        norm = float(m['steps'][0]['tensors'][target]['norm'] or rt.norm())
        # |D| must exceed the gate (a violation) and dominate the pooled reference-to-fp64 yardstick, so that
        # |c - g| = 5|D| is unexplained while |c - g| = |D| is explained.
        pooled = thresholds.get('ref_to_fp64', {}).get(f'{judge.family(target)}|s0', 0.0)
        D = u / u.norm() * max(gate, 2 * pooled) * norm

        def set_to(offset):
            return lambda t: (rt + offset).to(t.dtype)
        # outside the gate, explained by fp64: |c-g| = |r-g| -> must be resolved and PASS
        check('arbitration_resolves', {'status': 'PASS', 'arbitrated': 1, 'unresolved': 0},
              lambda c: _edit_tensor(c, case, rank, 0, target, set_to(2 * D)),
              golden_fn=lambda g: _edit_tensor(g, case, rank, 0, target, set_to(D)))
        # outside the gate, not explained: |c-g| = 5|r-g| > k|r-g| -> must stay INCORRECT
        check('arbitration_rejects', {'status': 'INCORRECT', 'arbitrated': 1, 'unresolved': 1},
              lambda c: _edit_tensor(c, case, rank, 0, target, set_to(4 * D)),
              golden_fn=lambda g: _edit_tensor(g, case, rank, 0, target, set_to(-D)))
    shutil.rmtree(tmp, ignore_errors=True)
    evidence = {'schema': 'numerics-selftest@1', 'reference': str(ref), 'variant': str(variant),
                'thresholds_sha256': judge.thresholds_sha(thresholds), 'target_tensor': target,
                'results': results, 'ok': all(r['ok'] for r in results)}
    if out:
        Path(out).write_text(json.dumps(evidence, indent=1) + '\n')
    return evidence
