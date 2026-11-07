"""Task-specific predeclared policy around the unchanged generic numerics."""
POST_DISCRETE = [(r'^out/gaussian\.post_(radii|visibility)#', 'After native atomic backward + Adam, discrete raster visibility/radii values are diagnostic; names, shape, dtype and finiteness remain checked. Initial visibility/radii are exact invariant hashes.')]

ERROR_SIGNATURES = {
    'illegal_memory_access': r'illegal\s+(?:memory\s+)?access|CUDA_ERROR_ILLEGAL_ADDRESS',
    'out_of_memory': r'CUDA[^\n]*out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED',
    'nonfinite': r'\bnan\b|\bnon[- ]?finite\b',
    'eager_fallback': r'(?:falling|falls?|fallback)\s+back[^\n]{0,80}\beager\b|fallback\s+to\s+eager|eager\s+fallback',
    'compile_failure': r'(?:compilation|compile)[^\n]{0,60}\bfailed\b|failed\s+to\s+compile|graph\s+capture[^\n]{0,60}\bfailed\b'
}

def scan_error_text(content):
    import re
    hits = []
    for number, line in enumerate(content.splitlines(), 1):
        for signature, pattern in ERROR_SIGNATURES.items():
            if re.search(pattern, line, flags=re.IGNORECASE):
                hits.append({'signature': signature, 'line_number': number, 'line': line[:500]})
    return hits

def strict_compare(ref, candidate, thresholds):
    from numerics.judge import compare
    report = compare(ref, candidate, thresholds, exclude=POST_DISCRETE)
    if report['dtype_changes'] or report['precision_mode_changes']:
        report['status'] = 'INCORRECT'
        report['exact_failures'].append({'check': 'original_precision_required'})
    return report

def compare_timing_runs(first, candidate, last):
    import statistics
    reports = [first, candidate, last]
    if any(r['status'] != 'VALID' for r in reports):
        return {'status': 'INVALID', 'reason': 'invalid arm'}
    if any(r['fingerprint'] != first['fingerprint'] for r in reports[1:]):
        return {'status': 'ERROR', 'reason': 'fingerprint mismatch'}
    if first['fingerprint']['smoke']:
        return {'status': 'UNAVAILABLE', 'reason': 'smoke is not a speed measurement'}
    medians = [r['median_ms'] for r in reports]
    bmean = (medians[0] + medians[2]) / 2
    gap = abs(medians[0] - medians[2]) / min(medians[0], medians[2])
    threshold = max(1.02, 1 + 2 * gap)
    noise = max(r['relative_window_range'] for r in reports)
    base_windows = first['windows'] + last['windows']
    # Each window must also agree with the reference distribution, so symmetric
    # contamination cannot hide behind matching reference medians.
    ref_median = statistics.median(w['ms_per_step'] for w in base_windows)
    if any(abs(w['ms_per_step'] / ref_median - 1) > .10 for w in base_windows):
        return {'status': 'INVALID', 'reason': 'individual reference window exceeds 10% deviation'}
    observed = bmean / medians[1]
    windows = [w for r in reports for w in r['windows']]
    insufficient = [{'arm': arm, 'window': index, 'active_clock_sample_count': w['telemetry'].get('active_clock_sample_count', 0)} for arm, report in zip(['b1', 'candidate', 'b2'], reports) for index, w in enumerate(report['windows']) if w['telemetry'].get('clock_observation') != 'RESOLVED' or w['telemetry'].get('active_clock_sample_count', 0) < 2 or w['telemetry'].get('sm_clock_median_mhz') is None]
    if insufficient:
        return {'status': 'UNAVAILABLE', 'reason': 'formal benefit unavailable on shared machine; active in-window clock samples insufficient', 'observation': 'UNRESOLVED', 'exploratory_resolved': False, 'exploratory_ratio': observed, 'clock_spread': None, 'unresolved_clock_windows': insufficient, 'reference_gap': gap, 'max_window_range': noise, 'arm_medians_ms': medians}
    clocks = [w['telemetry']['sm_clock_median_mhz'] for w in windows]
    clock_spread = (max(clocks) - min(clocks)) / statistics.median(clocks)
    if clock_spread > .05:
        return {'status': 'INVALID', 'reason': 'active in-window SM clock spread exceeds frozen 5% threshold', 'clock_spread': clock_spread}
    resolved = observed > threshold and observed > 1 + max(.01, 2 * noise)
    return {'status': 'UNAVAILABLE', 'reason': 'formal benefit unavailable on this shared machine with preserved registered viewer; exploratory observation only', 'observation': 'RESOLVED' if resolved else 'UNRESOLVED', 'exploratory_ratio': observed, 'exploratory_resolved': resolved, 'required_ratio': max(threshold, 1 + max(.01, 2 * noise)), 'reference_gap': gap, 'max_window_range': noise, 'clock_spread': clock_spread, 'arm_medians_ms': medians}
