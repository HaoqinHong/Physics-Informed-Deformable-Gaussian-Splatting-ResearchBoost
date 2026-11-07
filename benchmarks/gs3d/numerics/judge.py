"""Precision judge: calibrate gates from legitimate variants, then compare a candidate to the reference.

Verdict rules (docs/numerics.md):
- exact checks (phase, RNG fingerprints, call counters, invariants incl. learning rates, tensor names and
  shapes, finiteness, cross-rank pattern) must all hold, otherwise INCORRECT. The cross-rank pattern follows the
  adapter's sync declaration: in a 'replicated' model every tensor equal across ranks in the reference must stay
  equal; in a 'local' model per-tensor equality is coincidence (rounding noise, Adam's first step ~ lr*sign(g)),
  so only the model-level pattern is checked (all equal vs not, and no large jump in the equal fraction);
- every tensor must be bitwise identical or within its gate = max(k x envelope, floor). The envelope is pooled:
  the largest relative L2 distance of any legitimate variant from the reference over the same tensor family
  (same name, call indices dropped), all cases and ranks, at this step or earlier. One gate per tensor from a
  handful of samples rejects legitimate variants it has not seen (150k gates = 150k tests);
- a tensor outside its gate is resolved only by fp64 arbitration: ||c - g|| <= k x max(||r - g||, R x ||g||,
  floor x ||g||), where R is the reference's own pooled relative distance to fp64 for that family and step. A single
  ||r - g|| is one draw: for scalars and tiny tensors the reference can be close to fp64 by luck;
  the arbitration outcome is part of the verdict;
- parameters whose fp64 gradient is negligible next to the reference's (true gradient zero, e.g. attention key
  bias under softmax shift invariance) are excluded from grad/delta/optim comparison and listed;
- measured tier: N0 if everything is bitwise identical with identical dtypes and math modes (and every case has a step
  recorded without the dtype trace: compiled code falls back to eager under the trace's dispatch mode), N2 if any dtype
  (tensor or traced op output) or math mode (TF32, reduced-precision reductions, SDPA backends) differs, else N1.
  N3+ is a reviewer's call.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import torch

from .record import FWD_KINDS

TINY = 1e-300


def manifests(root):
    root = Path(root)
    found = {}
    for path in sorted(root.glob('*.r*.json')):
        m = json.loads(path.read_text())
        found[(m['case']['name'], m['rank'])] = m
    if not found:
        raise FileNotFoundError(f'no record manifests in {root}')
    return found


def load_step(root, case, rank, step):
    return torch.load(Path(root) / f'{case}.r{rank}.s{step}.pt', map_location='cpu', weights_only=True)


def rel(c, r, r_norm=None):
    """Relative L2 distance; works on full tensors or on CountSketches (pass the exact reference norm)."""
    c64, r64 = c.double(), r.double()
    diff = float((c64 - r64).norm())
    base = float(r64.norm()) if r_norm is None else r_norm
    return diff / max(base, TINY), diff


def stored_mode(desc):
    return desc.get('stored', 'full')


def kind_of(name):
    return name.split('/', 1)[0]


def param_of(name):
    kind = kind_of(name)
    return name.split('/')[2] if kind in ('grad', 'delta', 'optim') else None


def key(case, rank, step, name):
    return f'{case}|r{rank}|s{step}|{name}'


def family(name):
    """Tensor family: the name without optimizer/forward call indices (grad/gen#3/x -> grad/gen/x)."""
    return re.sub(r'#\d+', '', name)


def model_of(name):
    """Model that owns a param-class tensor: grad/opt#k/<model>.<p>, optim/opt/<model>.<p>/k, buffer/<model>.<b>."""
    parts = name.split('/')
    path = parts[2] if parts[0] in ('grad', 'delta', 'optim') and len(parts) > 2 else parts[-1]
    return path.split('.', 1)[0]


def _running_max(pooled):
    """(family, step) -> max over steps <= step; legitimate divergence only grows along a trajectory."""
    out, steps = {}, sorted({s for _, s in pooled})
    for fam in {f for f, _ in pooled}:
        best = 0.0
        for s in steps:
            best = max(best, pooled.get((fam, s), 0.0))
            out[f'{fam}|s{s}'] = best
    return out


def calibrate(ref, variants, k_fwd=2.0, k_bwd=3.0, floor_fwd=1e-6, floor_bwd=1e-5, golden=None):
    refs = manifests(ref)
    golds = manifests(golden) if golden else {}
    own, pooled, ref_fp64 = {}, defaultdict(float), defaultdict(float)
    for (case, rank), m in refs.items():
        if golden:
            problem = golden_problem(m, golds.get((case, rank)))
            if problem:
                raise ValueError(f'{case} r{rank}: fp64 reference is not the same computation: {problem}')
        for entry in m['steps']:
            s = entry['step']
            rt = load_step(ref, case, rank, s)
            env = {n: 0.0 for n in entry['tensors']}
            for v in variants:
                vt = load_step(v, case, rank, s)
                missing = set(rt) ^ set(vt)
                if missing:
                    raise ValueError(f'variant {v} does not line up with the reference: {sorted(missing)[:5]}')
                for n, r in rt.items():
                    if r.is_floating_point() and r.shape == vt[n].shape:
                        env[n] = max(env[n], rel(vt[n], r, entry['tensors'][n]['norm'])[0])
            for n, e in env.items():
                own[(case, rank, s, n)] = e
                pooled[(family(n), s)] = max(pooled[(family(n), s)], e)
            if golden:
                gt = load_step(golden, case, rank, s)
                gsteps = golds[(case, rank)]['steps'][s]['tensors']
                for n, r in rt.items():
                    g_norm = gsteps.get(n, {}).get('norm') or 0.0
                    if n in gt and r.is_floating_point() and r.shape == gt[n].shape and g_norm > 0:
                        d = float((r.double() - gt[n].double()).norm()) / g_norm
                        ref_fp64[(family(n), s)] = max(ref_fp64[(family(n), s)], d)
    family_env = _running_max(pooled)
    gates = {}
    for (case, rank, s, n), e in own.items():
        fwd = kind_of(n) in FWD_KINDS
        env = max(e, family_env[f'{family(n)}|s{s}'])
        gates[key(case, rank, s, n)] = {'env': e, 'pooled_env': env,
                                        'gate': max((k_fwd if fwd else k_bwd) * env, floor_fwd if fwd else floor_bwd)}
    meta = {'reference': str(ref), 'variants': [str(v) for v in variants], 'golden': str(golden) if golden else None,
            'k_fwd': k_fwd, 'k_bwd': k_bwd, 'floor_fwd': floor_fwd, 'floor_bwd': floor_bwd,
            'pooling': 'tensor family (call indices dropped) x all cases and ranks x steps up to this one'}
    return {'meta': meta, 'gates': gates, 'ref_to_fp64': _running_max(ref_fp64)}


def thresholds_sha(thresholds):
    return hashlib.sha256(json.dumps(thresholds, sort_keys=True).encode()).hexdigest()


def golden_problem(rm, gm):
    """The fp64 run must replay exactly the reference's steps: same phases, call counts, invariants and
    random draws. Otherwise ||r - g|| measures a different computation and arbitration becomes lenient."""
    if gm is None:
        return 'missing'
    if len(gm['steps']) != len(rm['steps']):
        return f'{len(gm["steps"])} steps vs {len(rm["steps"])}'
    for r, g in zip(rm['steps'], gm['steps']):
        for field in ('phase', 'rng_before', 'rng_after', 'counters', 'invariants'):
            if r.get(field) != g.get(field):
                return f'step {r["step"]} {field} differs (e.g. random numbers drawn in fp64 instead of cast up)'
        if set(r['tensors']) - set(g['tensors']):
            return f'step {r["step"]} lacks tensors'
    return None


def compare(ref, cand, thresholds, golden=None, exclude=(), zero_ratio=1e-3, local_margin=0.25):
    """Return a report dict; status is PASS, INCORRECT or ERROR."""
    meta, gates = thresholds['meta'], thresholds['gates']
    ref_to_fp64 = thresholds.get('ref_to_fp64', {})
    patterns = [(re.compile(p), why) for p, why in exclude]
    refs, cands = manifests(ref), manifests(cand)
    golds = manifests(golden) if golden else {}
    report = {'status': 'PASS', 'thresholds_sha256': thresholds_sha(thresholds), 'errors': [], 'exact_failures': [],
              'violations': [], 'arbitration': [], 'unresolved': [], 'excluded': {}, 'dtype_changes': [],
              'precision_mode_changes': [], 'checked': 0, 'bitwise_identical': 0, 'closest': []}
    ratios = []
    for (case, rank), rm in refs.items():
        cm = cands.get((case, rank))
        if cm is None:
            report['errors'].append(f'candidate has no record for case {case} rank {rank}')
            continue
        if len(cm['steps']) != len(rm['steps']):
            report['errors'].append(f'{case} r{rank}: {len(cm["steps"])} steps vs {len(rm["steps"])}')
            continue
        if rm.get('dtype_ops') and cm.get('dtype_ops'):
            # A different op mix (e.g. SDPA instead of matmul+softmax) is not a precision change by itself;
            # a dtype that appears or disappears among op outputs is.
            rdt = {k.rsplit(':', 1)[1] for k in rm['dtype_ops']}
            cdt = {k.rsplit(':', 1)[1] for k in cm['dtype_ops']}
            if rdt != cdt:
                report['dtype_changes'].append({'where': f'{case} r{rank} traced op outputs',
                                                'dtypes': [sorted(rdt), sorted(cdt)]})
            ops = sorted(set(rm['dtype_ops']) ^ set(cm['dtype_ops']))
            if ops:
                report.setdefault('op_changes', []).append({'where': f'{case} r{rank}', 'ops': ops[:30]})
        gm = golds.get((case, rank))
        if golden:
            problem = golden_problem(rm, gm)
            if problem:
                report['errors'].append(f'{case} r{rank}: fp64 reference is not the same computation: {problem}')
                continue
        for re_, ce in zip(rm['steps'], cm['steps']):
            s = re_['step']
            where = f'{case} r{rank} s{s}'
            for field in ('phase', 'rng_before', 'rng_after', 'counters', 'invariants'):
                if re_.get(field) != ce.get(field):
                    report['exact_failures'].append({'where': where, 'check': field,
                                                     'reference': re_.get(field), 'candidate': ce.get(field)})
            rmode, cmode = re_.get('math_mode'), ce.get('math_mode')
            if rmode is not None and cmode is not None and rmode != cmode:
                report['precision_mode_changes'].append(
                    {'where': where, 'changed': {k: [rmode.get(k), cmode.get(k)] for k in sorted(set(rmode) | set(cmode))
                                                 if rmode.get(k) != cmode.get(k)}})
            rnames, cnames = set(re_['tensors']), set(ce['tensors'])
            if rnames != cnames:
                item = {'where': where, 'missing': sorted(rnames - cnames)[:20], 'unexpected': sorted(cnames - rnames)[:20]}
                if re_.get('counters') != ce.get('counters'):
                    # different work (e.g. a skipped forward) -> wrong candidate, not an unreadable one
                    report['exact_failures'].append(dict(item, check='structure'))
                else:
                    # same work, different names (e.g. a layout change without a canonical view)
                    report['errors'].append(item)
                continue
            rt, ct = load_step(ref, case, rank, s), load_step(cand, case, rank, s)
            gt = load_step(golden, case, rank, s) if gm else None
            zero_params = set()
            if gt is not None:
                for n in rnames:
                    if kind_of(n) == 'grad' and n in gt:
                        rn = re_['tensors'][n]['norm'] or 0.0
                        gn = gm['steps'][s]['tensors'][n]['norm'] or 0.0
                        if rn > 0 and gn <= zero_ratio * rn:
                            zero_params.add(param_of(n))
            rx, cx = re_.get('xrank_equal', {}), ce.get('xrank_equal', {})
            sync = rm.get('sync') or {}
            if set(rx) != set(cx):
                report['exact_failures'].append({'where': where, 'check': 'cross_rank_pattern',
                                                 'tensors': sorted(set(rx) ^ set(cx))[:20]})
            groups = defaultdict(list)
            for n in sorted(set(rx) | set(cx)):
                if param_of(n) not in zero_params:          # equality of rounding noise means nothing
                    groups[(model_of(n), kind_of(n))].append(n)
            for (model, kind), group in sorted(groups.items()):
                mode = sync.get(model, 'replicated')
                if mode == 'local':
                    r_frac = sum(bool(rx.get(n)) for n in group) / len(group)
                    c_frac = sum(bool(cx.get(n)) for n in group) / len(group)
                    if (r_frac == 1.0) != (c_frac == 1.0) or c_frac > r_frac + local_margin:
                        report['exact_failures'].append(
                            {'where': where, 'check': 'cross_rank_pattern', 'model': model, 'kind': kind, 'sync': mode,
                             'reference_equal_fraction': r_frac, 'candidate_equal_fraction': c_frac})
                else:
                    broken = [n for n in group if rx.get(n) and not cx.get(n)]
                    if broken:
                        report['exact_failures'].append({'where': where, 'check': 'cross_rank_pattern', 'model': model,
                                                         'sync': mode, 'tensors': broken[:20]})
            for n in sorted(rnames):
                rd, cd = re_['tensors'][n], ce['tensors'][n]
                if rd['shape'] != cd['shape']:
                    report['errors'].append({'where': where, 'tensor': n, 'shape': [rd['shape'], cd['shape']]})
                    continue
                if rd['dtype'] != cd['dtype']:
                    report['dtype_changes'].append({'where': where, 'tensor': n, 'dtype': [rd['dtype'], cd['dtype']]})
                if rd['finite'] and not cd['finite']:
                    report['exact_failures'].append({'where': where, 'check': 'finite', 'tensor': n})
                    continue
                reason = next((why for p, why in patterns if p.search(n)), None)
                if reason is None and param_of(n) in zero_params:
                    reason = 'fp64 gradient negligible (true gradient is zero; values are rounding noise)'
                if reason:
                    report['excluded'].setdefault(reason, []).append(f'{where} {n}')
                    continue
                report['checked'] += 1
                if rd['sha256'] == cd['sha256']:
                    report['bitwise_identical'] += 1
                    continue
                if stored_mode(rd) != stored_mode(cd):
                    report['errors'].append({'where': where, 'tensor': n, 'error': 'stored as full vs sketch'})
                    continue
                gate = gates.get(key(case, rank, s, n))
                if gate is None:
                    report['errors'].append({'where': where, 'tensor': n, 'error': 'no calibrated gate'})
                    continue
                err = rel(ct[n], rt[n], rd['norm'])[0]
                ratios.append((err / gate['gate'], where, n, err, gate['gate']))
                if err <= gate['gate']:
                    continue
                violation = {'where': where, 'tensor': n, 'rel_err': err, 'gate': gate['gate']}
                report['violations'].append(violation)
                if gt is None or n not in gt:
                    report['unresolved'].append(dict(violation, arbitration='no fp64 reference'))
                    continue
                g = gt[n].double()
                fwd = kind_of(n) in FWD_KINDS
                k = meta['k_fwd'] if fwd else meta['k_bwd']
                g_norm = gm['steps'][s]['tensors'][n]['norm'] or 0.0
                floor = (meta['floor_fwd'] if fwd else meta['floor_bwd']) * g_norm
                cand_err = float((ct[n].double() - g).norm())
                ref_err = float((rt[n].double() - g).norm())
                ref_pooled = ref_to_fp64.get(f'{family(n)}|s{s}', 0.0) * g_norm
                ok = cand_err <= k * max(ref_err, ref_pooled, floor)
                verdict = dict(violation, cand_to_fp64=cand_err, ref_to_fp64=ref_err, ref_to_fp64_pooled=ref_pooled,
                               k=k, resolved=ok)
                report['arbitration'].append(verdict)
                if not ok:
                    report['unresolved'].append(verdict)
    report['closest'] = [{'ratio': r, 'where': w, 'tensor': n, 'rel_err': e, 'gate': g}
                         for r, w, n, e, g in sorted(ratios, reverse=True)[:10]]
    for reason, items in report['excluded'].items():
        report['excluded'][reason] = {'count': len(items), 'examples': items[:5]}
    if report['errors']:
        report['status'] = 'ERROR'
    elif report['exact_failures'] or report['unresolved']:
        report['status'] = 'INCORRECT'
    identical = report['checked'] == report['bitwise_identical']
    # A dtype-traced step ran under a dispatch mode, where compiled regions fall back to eager. Bitwise identity
    # proves N0 only if every case has at least one untraced step.
    report['traced_only_cases'] = sorted(f'{case} r{rank}' for (case, rank), cm in cands.items()
                                         if cm['steps'] and all(st.get('dtype_traced') for st in cm['steps']))
    report['bitwise_trusted'] = not report['traced_only_cases']
    if not report['bitwise_trusted']:
        identical = False
    changed = report['dtype_changes'] or report['precision_mode_changes']
    report['measured_tier'] = 'N2' if changed else ('N0' if identical else 'N1')
    return report


def leave_one_out(ref, variants, golden=None, **calibration):
    """Judge each legitimate variant with gates calibrated on the others. A judge that rejects legitimate
    variants it has not seen measures its calibration set, not correctness; every variant must PASS."""
    results = []
    for held in variants:
        th = calibrate(ref, [v for v in variants if v != held], golden=golden, **calibration)
        rep = compare(ref, held, th, golden=golden)
        results.append({'variant': str(held), 'status': rep['status'], 'measured_tier': rep['measured_tier'],
                        'violations': len(rep['violations']), 'unresolved': len(rep['unresolved']),
                        'exact_failures': rep['exact_failures'][:5], 'errors': rep['errors'][:5],
                        'unresolved_examples': [(u['where'], u['tensor']) for u in rep['unresolved'][:5]]})
    return {'ok': all(r['status'] == 'PASS' for r in results), 'results': results}
