"""Paired equivalence test (TOST) for training-level checks; standard library only.

d_i = metric(candidate, seed_i) - metric(reference, seed_i). delta must be frozen before looking at
results, anchored on the reference's own seed-to-seed spread. Verdicts:
  equivalent    90% CI of mean(d) inside (-delta, +delta)
  different     the paired t-test rejects mean(d) = 0 and the CI leaves the band
  inconclusive  neither
  suspicious    significantly better than +delta in the "good" direction: re-check for an N5 change
"""
import math
import statistics


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tost(diffs, delta, alpha=0.05, higher_is_better=None):
    n = len(diffs)
    if n < 2:
        raise ValueError('need at least two paired differences')
    if delta <= 0:
        raise ValueError('delta must be positive and frozen in advance')
    mean, sd = statistics.fmean(diffs), statistics.stdev(diffs)
    se, df = sd / math.sqrt(n), n - 1
    if se == 0:
        lo = hi = mean
        p_lower = p_upper = 0.0 if -delta < mean < delta else 1.0
        p_zero = 0.0 if mean != 0 else 1.0
    else:
        q = t_ppf(1 - alpha, df)
        lo, hi = mean - q * se, mean + q * se
        p_lower = 1 - t_cdf((mean + delta) / se, df)
        p_upper = t_cdf((mean - delta) / se, df)
        p_zero = 2 * (1 - t_cdf(abs(mean) / se, df))
    if max(p_lower, p_upper) < alpha:
        verdict = 'equivalent'
    elif p_zero < 2 * alpha and (lo >= delta or hi <= -delta or lo > 0 or hi < 0):
        verdict = 'different'
    else:
        verdict = 'inconclusive'
    if higher_is_better is not None:
        good_side = lo > delta if higher_is_better else hi < -delta
        if good_side:
            verdict = 'suspicious'
    return {'n': n, 'mean': mean, 'sd': sd, 'ci90': [lo, hi], 'delta': delta, 'p_tost': max(p_lower, p_upper),
            'p_paired_t': p_zero, 'verdict': verdict}
