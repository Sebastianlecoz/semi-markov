#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
football_model/distribution/sampler.py

Implémente des fonctions d'échantillonnage pour les différentes distributions.
Utilisé par le bootstrap paramétrique pour générer des échantillons simulés.
"""

import numpy as np
from scipy import stats


def get_sampler_for_model(model_name, params):
    """
    Retourne une fonction capable d'échantillonner à partir du modèle spécifié.
    
    Parameters:
    -----------
    model_name : str
        Nom du modèle paramétrique
    params : dict
        Dictionnaire des paramètres du modèle
    
    Returns:
    --------
    sampler : callable
        Fonction d'échantillonnage (size, random_state) -> échantillons
    """
    if model_name == "exponential":
        # Paramètres de scipy.stats.expon: loc, scale
        def sampler(size, random_state=None):
            return stats.expon.rvs(
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "weibull":
        # Paramètres de scipy.stats.weibull_min: c, loc, scale
        def sampler(size, random_state=None):
            return stats.weibull_min.rvs(
                c=params.get("c", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "lognormal":
        # Paramètres de scipy.stats.lognorm: s, loc, scale
        def sampler(size, random_state=None):
            return stats.lognorm.rvs(
                s=params.get("s", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "loglogistic":
        # Paramètres de scipy.stats.fisk: c, loc, scale
        def sampler(size, random_state=None):
            return stats.fisk.rvs(
                c=params.get("c", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "gen_weibull":
        # Paramètres de scipy.stats.exponweib: a, c, loc, scale
        def sampler(size, random_state=None):
            return stats.exponweib.rvs(
                a=params.get("a", 1), 
                c=params.get("c", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "genf":
        # Paramètres de scipy.stats.mielke: k, s, loc, scale
        def sampler(size, random_state=None):
            return stats.mielke.rvs(
                k=params.get("k", 1), 
                s=params.get("s", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "johnsonsu":
        # Paramètres de scipy.stats.johnsonsu: a, b, loc, scale
        def sampler(size, random_state=None):
            return stats.johnsonsu.rvs(
                a=params.get("a", 0), 
                b=params.get("b", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "invgauss":
        # Paramètres de scipy.stats.invgauss: mu, loc, scale
        def sampler(size, random_state=None):
            return stats.invgauss.rvs(
                mu=params.get("mu", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "mielke":
        # Paramètres de scipy.stats.mielke: k, s, loc, scale
        def sampler(size, random_state=None):
            return stats.mielke.rvs(
                k=params.get("k", 1), 
                s=params.get("s", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "geninvgauss":
        # Paramètres de scipy.stats.geninvgauss: p, b, loc, scale
        def sampler(size, random_state=None):
            return stats.geninvgauss.rvs(
                p=params.get("p", 1), 
                b=params.get("b", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "genlogistic":
        # Paramètres de scipy.stats.genlogistic: c, loc, scale
        def sampler(size, random_state=None):
            return stats.genlogistic.rvs(
                c=params.get("c", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "gamma":
        # Paramètres de scipy.stats.gamma: a, loc, scale
        def sampler(size, random_state=None):
            return stats.gamma.rvs(
                a=params.get("a", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
        
    elif model_name == "gengamma":
        # Paramètres de scipy.stats.gengamma: a, c, loc, scale
        def sampler(size, random_state=None):
            return stats.gengamma.rvs(
                a=params.get("a", 1), 
                c=params.get("c", 1), 
                loc=params.get("loc", 0), 
                scale=params.get("scale", 1), 
                size=size, 
                random_state=random_state
            )
        return sampler
    
    # Si le modèle n'est pas reconnu, retourner None
    return None
