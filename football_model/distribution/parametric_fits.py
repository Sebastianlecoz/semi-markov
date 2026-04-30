# -*- coding: utf-8 -*-
"""
src/football_model/distribution/parametric_fits.py

Parametric distribution fittings and model selection:
  - compute_aic_bic: calculate AIC/BIC for fits
  - fit_model: fit a single distribution, return parameters, KS stats
  - fit_all_distributions: fit all defined models
  - extract_transitions, build_duration_dict, filter_positive_durations: data utilities
"""

import numpy as np
import scipy.stats as stats

# -----------------------------------------------------------------------------
# 1) Calcul des AIC / BIC pour les différentes lois
# -----------------------------------------------------------------------------
def compute_aic_bic(model, params, data):
    """
    Calcule l'AIC et le BIC pour un modèle donné en utilisant la log-vraisemblance basée sur la densité.
    """
    n = len(data)
    eps = 1e-12

    # Lookup model info and return None if undefined
    info = models_info.get(model)
    if info is None:
        return {'AIC': None, 'BIC': None}

    dist = info['dist']
    # Prepare parameters: separate shape from loc/scale
    names = info['params']
    shape_args = [params[name] for name in names if name not in ('loc', 'scale')]
    loc = params.get('loc', 0)
    scale = params.get('scale', 1)

    # Compute pdf and log-likelihood
    pdf_vals = dist.pdf(data, *shape_args, loc=loc, scale=scale)
    pdf_vals = np.maximum(pdf_vals, eps)
    logL = np.sum(np.log(pdf_vals))

    # Number of parameters (shape + scale + optional loc)
    num_shape = getattr(dist, 'numargs', len(shape_args))
    k = num_shape + 1  # scale always estimated
    if not info.get('fix_loc', False):
        k += 1

    # Calculate AIC and BIC directly
    aic_val = 2 * k - 2 * logL
    bic_val = k * np.log(n) - 2 * logL
    return {'AIC': aic_val, 'BIC': bic_val}


# -----------------------------------------------------------------------------
# 2) Configuration de chaque distribution / loi
# -----------------------------------------------------------------------------
models_info = {
    'exponential':   {'dist': stats.expon,          'params': ['loc', 'scale'],                'fix_loc': True,  'ks_name': 'expon'},
    'weibull':       {'dist': stats.weibull_min,    'params': ['c', 'loc', 'scale'],           'fix_loc': True,  'ks_name': 'weibull_min'},
    'lognormal':     {'dist': stats.lognorm,        'params': ['s', 'loc', 'scale'],           'fix_loc': True,  'ks_name': 'lognorm'},
    'loglogistic':   {'dist': stats.fisk,           'params': ['c', 'loc', 'scale'],           'fix_loc': True,  'ks_name': 'fisk'},
    'gen_weibull':   {'dist': stats.exponweib,      'params': ['a', 'c', 'loc', 'scale'],      'fix_loc': True,  'ks_name': 'exponweib'},
    #'genf':          {'dist': stats.mielke,         'params': ['k', 's', 'loc', 'scale'],      'fix_loc': True,  'ks_name': 'mielke'},
    'johnsonsu':     {'dist': stats.johnsonsu,      'params': ['a', 'b', 'loc', 'scale'],      'fix_loc': False, 'ks_name': 'johnsonsu'},
    'invgauss':      {'dist': stats.invgauss,       'params': ['mu', 'loc', 'scale'],          'fix_loc': False, 'ks_name': 'invgauss'},
    'mielke':        {'dist': stats.mielke,         'params': ['k', 's', 'loc', 'scale'],      'fix_loc': True,  'ks_name': 'mielke'},
    'geninvgauss':   {'dist': stats.geninvgauss,    'params': ['p', 'b', 'loc', 'scale'],      'fix_loc': True,  'ks_name': 'geninvgauss'},
    'genlogistic':   {'dist': stats.genlogistic,    'params': ['c', 'loc', 'scale'],           'fix_loc': True,  'ks_name': 'genlogistic'},
    'gamma':         {'dist': stats.gamma,          'params': ['a', 'loc', 'scale'],           'fix_loc': True,  'ks_name': 'gamma'},
    'gengamma':      {'dist': stats.gengamma,       'params': ['a', 'c', 'loc', 'scale'],      'fix_loc': True,  'ks_name': 'gengamma'},
}


# -----------------------------------------------------------------------------
# 3) Fonction d'ajustement pour un modèle unique
# -----------------------------------------------------------------------------
def fit_model(model_name, data):
    """
    Ajuste un modèle donné et renvoie un dictionnaire contenant
    les paramètres estimés, le résultat du test KS, l'AIC et le BIC.
    """
    info = models_info[model_name]
    dist = info['dist']
    fix_loc = info.get('fix_loc', True)
    ks_name = info.get('ks_name', model_name)
    try:
        if fix_loc:
            params_tuple = dist.fit(data, floc=0)
        else:
            params_tuple = dist.fit(data)
        
        param_names = info['params']
        params = {name: float(value) for name, value in zip(param_names, params_tuple)}

        # Test KS bilatéral
        ks_res = stats.kstest(data, ks_name, args=params_tuple)
        params['KS_stat'] = float(ks_res.statistic)
        params['KS_p'] = float(ks_res.pvalue)

        # AIC / BIC
        bic_dict = compute_aic_bic(model_name, params, data)
        params.update(bic_dict)
        if params.get('BIC') is None or not np.isfinite(params.get('BIC')):
            return None

        return params

    except Exception as e:
        print(f"[Warning] Erreur dans l'ajustement du modèle '{model_name}': {e}")
        return None


# -----------------------------------------------------------------------------
# 4) Ajustement de toutes les distributions
# -----------------------------------------------------------------------------
def fit_all_distributions(data):
    """
    Ajuste l'ensemble des modèles définis et retourne un dict {model_name: params}.
    """
    fit_results = {}
    for model_name in models_info.keys():
        result = fit_model(model_name, data)
        if result is not None:
            fit_results[model_name] = result
    return fit_results


# -----------------------------------------------------------------------------
# 5) Fonctions utilitaires pour construire les transitions et filtrer
# -----------------------------------------------------------------------------
def extract_transitions(all_sequences):
    """
    Extrait toutes les transitions (i → j) et durées associées.
    """
    transitions = []
    for seq in all_sequences:
        for k in range(len(seq) - 1):
            i_state = seq[k]['state']
            i_dur   = seq[k]['duration']
            j_state = seq[k + 1]['state']
            transitions.append((i_state, j_state, i_dur))
    return transitions


def build_duration_dict(transitions):
    """
    Construit un dict où chaque clé (i, j) est la liste des durées t.
    """
    from collections import defaultdict
    d_ij = defaultdict(list)
    for i, j, t in transitions:
        d_ij[(i, j)].append(t)
    return dict(d_ij)


def filter_positive_durations(d_ij):
    """
    Garde uniquement les durées strictement positives.
    """
    return {key: [x for x in durations if x > 0] for key, durations in d_ij.items()}
