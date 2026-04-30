# ==============================================
# src/football_model/aft/mapping.py
# ==============================================
"""Transition → model mapping registry.

Two modes of operation
----------------------
1. **Dynamic (recommended)** — call ``spec_from_scipy(dist_name)`` with the
   best-fit scipy distribution name returned by ``evaluate_sejourn_distributions()``.
   The AFT family is chosen automatically from the fitted data.

2. **Static (fallback)** — ``TRANSITION_MAP`` hardcodes the AFT family per
   transition from offline QQ-score selection.  Used when
   ``evaluate_sejourn_distributions()`` has not been called.

All twelve candidate families are covered
-----------------------------------------
    Log-normal         lognorm       → LogNormalAFTFitter               (lifelines native)
    Johnson SU         johnsonsu     → JohnsonSUAFTFitter               (custom)
    Log-logistic       loglogistic   → LogLogisticAFTFitter             (lifelines native)
    Mielke             mielke        → MielkeAFTFitter                  (custom)
    Inverse Gaussian   invgauss      → InvGaussAFTFitter                (custom)
    Gen. Inv. Gaussian geninvgauss   → GenInvGaussAFTFitter             (custom)
    Weibull            weibull_min   → WeibullAFTFitter                 (lifelines native)
    Exponential        expon         → WeibullAFTFitter                 (shape=1 special case)
    Gen. Weibull       exponweib     → GenWeibullAFTFitter              (custom)
    Gen. Logistic      genlogistic   → GenLogisticAFTFitter             (custom)
    Gen. Gamma         gengamma      → GeneralizedGammaRegressionFitter (lifelines native)
    Gamma              gamma         → GeneralizedGammaRegressionFitter (Gamma is a special case)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from lifelines import (
    GeneralizedGammaRegressionFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from .fitters import (
    GenInvGaussAFTFitter,
    GenLogisticAFTFitter,
    GenWeibullAFTFitter,
    InvGaussAFTFitter,
    JohnsonSUAFTFitter,
    MielkeAFTFitter,
)

__all__ = [
    "TransitionSpec",
    "TRANSITION_MAP",
    "SCIPY_TO_AFT",
    "spec_from_scipy",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransitionSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TransitionSpec:
    """Container for an AFT transition model specification."""
    model:      Type
    penalizer:  float                           = 0.1
    regressors: Optional[Dict[str, List[str]]] = None

    def kwargs(self) -> dict:
        return {"penalizer": self.penalizer}


# ---------------------------------------------------------------------------
# Covariate column lists
# ---------------------------------------------------------------------------

_dist_angl_disp = ["dist_scaled", "ang_scaled", "disp_scaled"]
_int            = ["Intercept"]


# ---------------------------------------------------------------------------
# Regressor templates — one per model family
# ---------------------------------------------------------------------------
# Each template routes covariates to the *primary* location/scale parameter.
# Shape / tail parameters are intercept-only: they are hard to identify with
# covariates and tend to converge poorly.

_REGS_MIELKE = {
    "lambda_": _dist_angl_disp + _int,   # scale
    "k_":      _int,                     # shape 1
    "s_":      _int,                     # shape 2
}

_REGS_INVGAUSS = {
    "mu_":  _dist_angl_disp + _int,      # mean
    "lam_": _int,                        # shape
}

_REGS_GENINVGAUSS = {
    "lambda_": _dist_angl_disp + _int,   # location
    "p_":      _int,                     # tail
    "b_":      _int,                     # concentration
}

_REGS_JOHNSONSU = {
    "gamma_":  _dist_angl_disp + _int,   # location
    "delta_":  _int,                     # shape
    "lambda_": _int,                     # scale
    "xi_":     _int,                     # shift
}

_REGS_GENGAMMA = {
    "mu_":     _dist_angl_disp + _int,   # log-scale
    "sigma_":  _int,                     # scale
    "lambda_": _int,                     # shape
}

_REGS_GENWEIBULL = {
    "lambda_": _dist_angl_disp + _int,   # scale
    "k_":      _int,                     # inner Weibull shape
    "alpha_":  _int,                     # exponent (tail thickness)
}

_REGS_LOGNORM = {
    "mu_":    _dist_angl_disp + _int,    # log-mean
    "sigma_": _int,                      # log-std
}

_REGS_GENLOGISTIC = {
    "lambda_": _dist_angl_disp + _int,   # scale
    "rho_":    _int,                     # shape
}


# ---------------------------------------------------------------------------
# SCIPY_TO_AFT — dynamic lookup (all 12 candidate families)
# ---------------------------------------------------------------------------

SCIPY_TO_AFT: dict[str, TransitionSpec] = {

    # ── Log-normal ──────────────────────────────────────────────────────────
    "lognorm":      TransitionSpec(model=LogNormalAFTFitter,
                                   regressors=_REGS_LOGNORM),

    # ── Johnson SU ──────────────────────────────────────────────────────────
    "johnsonsu":    TransitionSpec(model=JohnsonSUAFTFitter,
                                   regressors=_REGS_JOHNSONSU),

    # ── Log-logistic ────────────────────────────────────────────────────────
    "loglogistic":  TransitionSpec(model=LogLogisticAFTFitter,
                                   regressors=None),
    "fisk":         TransitionSpec(model=LogLogisticAFTFitter,   # scipy alias
                                   regressors=None),

    # ── Mielke Beta-Kappa ───────────────────────────────────────────────────
    "mielke":       TransitionSpec(model=MielkeAFTFitter,
                                   regressors=_REGS_MIELKE),
    "burr12":       TransitionSpec(model=MielkeAFTFitter,        # Burr XII ≡ Mielke
                                   regressors=_REGS_MIELKE),

    # ── Inverse Gaussian ────────────────────────────────────────────────────
    "invgauss":     TransitionSpec(model=InvGaussAFTFitter,
                                   regressors=_REGS_INVGAUSS),
    "wald":         TransitionSpec(model=InvGaussAFTFitter,      # scipy alias
                                   regressors=_REGS_INVGAUSS),

    # ── Generalized Inverse Gaussian ─────────────────────────────────────────
    "geninvgauss":  TransitionSpec(model=GenInvGaussAFTFitter,
                                   regressors=_REGS_GENINVGAUSS),
    "norminvgauss": TransitionSpec(model=GenInvGaussAFTFitter,   # closest match
                                   regressors=_REGS_GENINVGAUSS),

    # ── Weibull ─────────────────────────────────────────────────────────────
    "weibull_min":  TransitionSpec(model=WeibullAFTFitter,
                                   regressors=None),
    "weibull":      TransitionSpec(model=WeibullAFTFitter,
                                   regressors=None),

    # ── Exponential (Weibull shape = 1) ─────────────────────────────────────
    "expon":        TransitionSpec(model=WeibullAFTFitter,
                                   regressors=None),

    # ── Generalized Weibull (Exponentiated Weibull) ─────────────────────────
    "exponweib":    TransitionSpec(model=GenWeibullAFTFitter,
                                   regressors=_REGS_GENWEIBULL),
    "genweibull":   TransitionSpec(model=GenWeibullAFTFitter,    # non-scipy alias
                                   regressors=_REGS_GENWEIBULL),

    # ── Generalized Logistic ─────────────────────────────────────────────────
    "genlogistic":  TransitionSpec(model=GenLogisticAFTFitter,
                                   regressors=_REGS_GENLOGISTIC),

    # ── Generalized Gamma ────────────────────────────────────────────────────
    "gengamma":     TransitionSpec(model=GeneralizedGammaRegressionFitter,
                                   regressors=_REGS_GENGAMMA),

    # ── Gamma (GenGamma is a strict superset — Gamma is the λ→0 limit) ──────
    "gamma":        TransitionSpec(model=GeneralizedGammaRegressionFitter,
                                   regressors=_REGS_GENGAMMA),
}

# Default when the scipy name is completely unknown
_FALLBACK_SPEC = TransitionSpec(model=WeibullAFTFitter, regressors=None)


def spec_from_scipy(dist_name: str) -> TransitionSpec:
    """
    Return the ``TransitionSpec`` that best matches a scipy distribution name.

    Parameters
    ----------
    dist_name : str
        The ``best_model`` string from ``process_all_transitions``, e.g.
        ``"mielke"``, ``"geninvgauss"``, ``"johnsonsu"``, ``"exponweib"``.

    Returns
    -------
    TransitionSpec
        Ready-to-use spec.  Falls back to ``WeibullAFTFitter`` with a warning
        if the name is not in ``SCIPY_TO_AFT``.
    """
    key  = dist_name.lower().strip()
    spec = SCIPY_TO_AFT.get(key)
    if spec is None:
        log.warning(
            "scipy distribution '%s' has no registered AFT equivalent — "
            "falling back to WeibullAFTFitter.  "
            "Add it to SCIPY_TO_AFT in mapping.py if you need a better match.",
            dist_name,
        )
        return _FALLBACK_SPEC
    return spec


# ---------------------------------------------------------------------------
# TRANSITION_MAP — static legacy map
# (Pass -> Pass and Shot -> Carry now use InvGaussAFTFitter instead of Weibull)
# ---------------------------------------------------------------------------

TRANSITION_MAP: dict[str, TransitionSpec] = {

    # ── Carry ────────────────────────────────────────────────────────────────
    "Carry -> Pass":      TransitionSpec(model=GenInvGaussAFTFitter,
                                         regressors=_REGS_GENINVGAUSS),
    "Carry -> Pressure":  TransitionSpec(model=MielkeAFTFitter,
                                         regressors=_REGS_MIELKE),
    "Carry -> Loss":      TransitionSpec(model=GenInvGaussAFTFitter,
                                         regressors=_REGS_GENINVGAUSS),
    "Carry -> Stoppage":  TransitionSpec(model=GeneralizedGammaRegressionFitter,
                                         regressors=_REGS_GENGAMMA),
    "Carry -> Shot":      TransitionSpec(model=JohnsonSUAFTFitter,
                                         regressors=_REGS_JOHNSONSU),
    "Carry -> Carry":     TransitionSpec(model=GenInvGaussAFTFitter,
                                         regressors=_REGS_GENINVGAUSS),

    # ── Pass ─────────────────────────────────────────────────────────────────
    "Pass -> Carry":      TransitionSpec(model=MielkeAFTFitter,
                                         regressors=_REGS_MIELKE),
    "Pass -> Loss":       TransitionSpec(model=MielkeAFTFitter,
                                         regressors=_REGS_MIELKE),
    "Pass -> Pressure":   TransitionSpec(model=LogLogisticAFTFitter,
                                         regressors=None),
    "Pass -> Shot":       TransitionSpec(model=MielkeAFTFitter,
                                         regressors=_REGS_MIELKE),
    "Pass -> Stoppage":   TransitionSpec(model=GeneralizedGammaRegressionFitter,
                                         regressors=_REGS_GENGAMMA),
    "Pass -> Pass":       TransitionSpec(model=InvGaussAFTFitter,     # was WeibullAFT
                                         regressors=_REGS_INVGAUSS),

    # ── Pressure ─────────────────────────────────────────────────────────────
    "Pressure -> Carry":    TransitionSpec(model=MielkeAFTFitter,
                                           regressors=_REGS_MIELKE),
    "Pressure -> Pass":     TransitionSpec(model=MielkeAFTFitter,
                                           regressors=_REGS_MIELKE),
    "Pressure -> Loss":     TransitionSpec(model=MielkeAFTFitter,
                                           regressors=_REGS_MIELKE),
    "Pressure -> Shot":     TransitionSpec(model=MielkeAFTFitter,
                                           regressors=_REGS_MIELKE),
    "Pressure -> Stoppage": TransitionSpec(model=MielkeAFTFitter,
                                           regressors=_REGS_MIELKE),
    "Pressure -> Pressure": TransitionSpec(model=JohnsonSUAFTFitter,
                                           regressors=_REGS_JOHNSONSU),

    # ── Shot ─────────────────────────────────────────────────────────────────
    "Shot -> Goal":     TransitionSpec(model=MielkeAFTFitter,
                                       regressors=_REGS_MIELKE),
    "Shot -> Loss":     TransitionSpec(model=MielkeAFTFitter,
                                       regressors=_REGS_MIELKE),
    "Shot -> Pass":     TransitionSpec(model=GenInvGaussAFTFitter,
                                       regressors=_REGS_GENINVGAUSS),
    "Shot -> Carry":    TransitionSpec(model=InvGaussAFTFitter,       # was WeibullAFT
                                       regressors=_REGS_INVGAUSS),
    "Shot -> Pressure": TransitionSpec(model=GenInvGaussAFTFitter,
                                       regressors=_REGS_GENINVGAUSS),
    "Shot -> Stoppage": TransitionSpec(model=GenInvGaussAFTFitter,
                                       regressors=_REGS_GENINVGAUSS),

    # ── Stoppage ─────────────────────────────────────────────────────────────
    "Stoppage -> Stoppage": TransitionSpec(model=JohnsonSUAFTFitter,
                                           regressors=_REGS_JOHNSONSU),
}