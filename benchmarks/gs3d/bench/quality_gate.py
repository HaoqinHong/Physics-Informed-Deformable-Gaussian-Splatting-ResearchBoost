"""Freeze descriptive quality bands from two baseline seeds, then compare pairs.

Two seeds provide a noise yardstick, not a statistical equivalence claim.
"""
import argparse
import json
from pathlib import Path
import statistics
from common import digest, new_output, sha256, write_json

METRICS = ['PSNR', 'SSIM', 'L1']

def load_inputs(path):
    paths = json.loads(Path(path).read_text())
    if set(paths) != {'bonsai', 'bicycle', 'garden'}:
        raise ValueError('all three scenes required')
    result, hashes = {}, {}
    for scene, seeds in paths.items():
        if set(seeds) != {'0', '1'}:
            raise ValueError('matched seeds 0/1 required')
        result[scene], hashes[scene] = {}, {}
        for seed, filename in seeds.items():
            p = Path(filename)
            outer = json.loads((p.parent / 'quality.json').read_text())
            if outer['status'] != 'complete' or outer['seed'] != int(seed):
                raise ValueError('incomplete quality run')
            result[scene][seed] = json.loads(p.read_text())['iterations']
            hashes[scene][seed] = {'metrics': sha256(p), 'quality_manifest': sha256(p.parent / 'quality.json'), 'data_sha256': outer['data_sha256'], 'config_sha256': outer['config_sha256']}
    return result, hashes

def values(iteration):
    values = {}
    for group in ['mean', 'worst']:
        for metric in METRICS:
            values[f'{group}/{metric}'] = iteration[group][metric]
    for frame in iteration['frames']:
        for metric in METRICS:
            values[f'frame/{frame["image_name"]}/{metric}'] = frame[metric]
    return values

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze', 'compare'])
    p.add_argument('--inputs', required=True, help='JSON scene -> seed string -> quality_metrics.json')
    p.add_argument('--frozen', help='quality_frozen.json, required for compare')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    inputs, hashes = load_inputs(a.inputs)
    out = new_output(a.out)
    if a.action == 'freeze':
        frozen = {'schema': 'gs3d-quality-two-seed-band@1', 'baseline': inputs, 'input_hashes': hashes, 'bands': {}, 'seeds': [0, 1], 'iterations': [7000, 30000], 'policy': 'Per endpoint band = 2 x sample standard deviation of two original baseline seeds. Descriptive noise yardstick only; no statistical equivalence claim with n=2. Every mean, worst, and per-frame metric is retained. Candidate improvements beyond band also require review.'}
        for scene in inputs:
            frozen['bands'][scene] = {}
            for iteration in ['7000', '30000']:
                first, second = [values(inputs[scene][str(seed)][iteration]) for seed in [0, 1]]
                if set(first) != set(second):
                    raise ValueError('baseline held-out frame identities differ')
                frozen['bands'][scene][iteration] = {key: {'baseline_mean': statistics.mean([first[key], second[key]]), 'delta': 2 * statistics.stdev([first[key], second[key]])} for key in first}
        frozen['sha256'] = digest(frozen)
        write_json(out / 'quality_frozen.json', frozen)
        return
    if not a.frozen:
        p.error('--frozen required for compare')
    frozen = json.loads(Path(a.frozen).read_text())
    comparisons, outside = {}, []
    for scene in inputs:
        comparisons[scene] = {}
        for iteration in ['7000', '30000']:
            baseline_values = [values(frozen['baseline'][scene][str(seed)][iteration]) for seed in [0, 1]]
            candidate_values = [values(inputs[scene][str(seed)][iteration]) for seed in [0, 1]]
            if any(set(v) != set(baseline_values[0]) for v in candidate_values):
                raise ValueError('candidate held-out frame identities differ')
            endpoints = {}
            for key, band in frozen['bands'][scene][iteration].items():
                paired = [candidate_values[seed][key] - baseline_values[seed][key] for seed in [0, 1]]
                mean = statistics.mean(paired)
                within = abs(mean) <= band['delta']
                endpoints[key] = {'paired_differences': paired, 'mean_difference': mean, 'delta': band['delta'], 'within_baseline_band': within}
                if not within:
                    outside.append({'scene': scene, 'iteration': iteration, 'endpoint': key, 'mean_difference': mean, 'delta': band['delta']})
            comparisons[scene][iteration] = endpoints
        for seed in ['0', '1']:
            for key in ['data_sha256', 'config_sha256']:
                if hashes[scene][seed][key] != frozen['input_hashes'][scene][seed][key]:
                    raise ValueError('candidate quality data/config differs')
    write_json(out / 'quality_comparison.json', {'status': 'OUTSIDE_BASELINE_BAND' if outside else 'WITHIN_BASELINE_BAND', 'statistical_equivalence': 'UNAVAILABLE: two seeds are insufficient for a supported equivalence claim', 'outside': outside, 'comparisons': comparisons, 'frozen_sha256': frozen['sha256'], 'candidate_input_hashes': hashes})

if __name__ == '__main__':
    main()
