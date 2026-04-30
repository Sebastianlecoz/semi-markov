# -*- coding: utf-8 -*-
"""
src/football_model/distribution/gof_tests.py

Goodness-of-fit tests for distributions and splines:
  - ks_p_simple: SciPy KS test
  - ks_p_asymptotique: asymptotic KS p-value
  - ks_bootstrap_parametric: bootstrap KS for parametric models
  - compute_ks_stat: KS statistic for spline survival functions
  - compute_ks_test: asymptotic KS test for splines
  - ks_bootstrap_spline: bootstrap KS for spline models
"""

import numpy as np
from scipy.stats import kstest, kstwo


# -------- KS simple --------
def ks_p_simple(sample, cdf):
    """
    Simple two-sided Kolmogorov-Smirnov test using SciPy's kstest.
    This exact test compares the empirical distribution of `sample` against
    a theoretical CDF `cdf`. Returns the statistic D and p-value.
    Methodology based on standard KS test (Massey, 1951).
    """
    res = kstest(sample, cdf)
    return float(res.statistic), float(res.pvalue)


# -------- KS bootstrap parametric --------
def ks_bootstrap_parametric(sample, cdf_func, sampler, n_boot=1000, random_state=None):
    """
    Parametric bootstrap for KS test (Davison & Hinkley, 1997).
    1. Compute observed KS statistic D_obs using ks_p_simple.
    2. Generate B bootstrap samples from the fitted model via `sampler`.
    3. Compute KS stat for each simulated sample against original cdf_func.
    4. p-value = proportion of bootstrap D_bs >= D_obs, with 1/(B+1) bias correction.
    Returns (D_obs, p_boot).
    """
    rng = np.random.RandomState(random_state)
    data = np.sort(np.asarray(sample))
    D_obs, _ = ks_p_simple(data, cdf_func)
    boots = []
    for _ in range(n_boot):
        try:
            bs = sampler(len(data), rng)
            bs_sorted = np.sort(bs)
            D_bs, _ = ks_p_simple(bs_sorted, cdf_func)
            boots.append(D_bs)
        except Exception:
            pass
    if boots:
        arr = np.array(boots)
        p_boot = max(np.mean(arr >= D_obs), 1.0/(len(arr)+1))
    else:
        p_boot = 0.0
    return D_obs, p_boot


# -------- KS for splines --------
def compute_ks_stat(data, sf):
    """
    Compute the KS statistic between empirical CDF of observed durations
    and model CDF inferred from lifelines.SplineFitter's survival function.
    D = max |F_emp(x) - F_model(x)|, where F_model = 1 - S(x).
    """
    d = np.sort(np.asarray(data))
    ecdf = np.arange(1, len(d)+1) / len(d)
    model_cdf = 1 - sf.survival_function_at_times(d).values.flatten()
    return np.max(np.abs(ecdf - model_cdf))


def compute_ks_test(data, sf):
    """
    Two-sided asymptotic KS test for spline models.
    Uses scipy.stats.kstwo survival function to approximate p-value.
    Appropriate for moderate to large n (Massey, 1951; Kolmogorov, 1933).
    Returns (D, p).
    """
    D = compute_ks_stat(data, sf)
    p = kstwo.sf(D, len(data))
    return D, float(p)


def ks_bootstrap_spline(data, sf, n_boot=100, boot_size=None, alpha=0.05, random_state=None):
    """
    Bootstrap goodness-of-fit for spline models (parametric bootstrap):
    1. Inversion sampling from spline survival via sample_from_spline.
    2. Re-fit spline on each bootstrap sample (safe_spline_fit fallback logic).
    3. Compute KS statistic for each re-fit model.
    4. p-value = proportion of bootstrap stats >= observed, with bias correction.
    Based on Claeskens & Hjort (2008) and Davison & Hinkley (1997).
    Returns dict with 'ks_stat', 'ks_p_boot', 'model_pass', and diagnostic counts.
    """
    from .monotone_mspline import sample_from_spline, determine_boot_size
    from .safe_spline import safe_spline_fit

    rng = np.random.RandomState(random_state)
    d = np.sort(np.asarray(data))
    n = len(d)
    if boot_size is None:
        boot_size = determine_boot_size(n)
    D_obs = compute_ks_stat(d, sf)
    vals = []
    for _ in range(n_boot):
        bs = sample_from_spline(sf, boot_size, rng.randint(0, 2**32-1))
        res = safe_spline_fit(bs, events=None)
        if res.get('sf') is not None:
            vals.append(compute_ks_stat(bs, res['sf']))
    if vals:
        arr = np.array(vals)
        p_boot = max(np.mean(arr >= D_obs), 1.0/(len(arr)+1))
    else:
        p_boot = 0.0
    return {
        'ks_stat': D_obs,
        'ks_p_boot': p_boot,
        'model_pass': p_boot > alpha,
        'boot_size': boot_size,
        'successful_boots': len(vals),
        'total_boots': n_boot
    }
