# ==============================================
# src/football_model/aft/fitters.py
# ==============================================
"""Custom parametric AFT fitters.

Covers all twelve candidate families evaluated during sojourn-time selection:

    Log-normal         → lifelines.LogNormalAFTFitter       (native)
    Johnson SU         → JohnsonSUAFTFitter                 (custom)
    Log-logistic       → lifelines.LogLogisticAFTFitter     (native)
    Mielke             → MielkeAFTFitter                    (custom)
    Inverse Gaussian   → InvGaussAFTFitter                  (custom)   ← NEW
    Gen. Inv. Gaussian → GenInvGaussAFTFitter               (custom)
    Weibull            → lifelines.WeibullAFTFitter         (native)
    Exponential        → lifelines.WeibullAFTFitter         (shape=1 special case)
    Gen. Weibull       → GenWeibullAFTFitter                (custom)   ← NEW
    Gen. Logistic      → GenLogisticAFTFitter               (custom)
    Gen. Gamma         → lifelines.GeneralizedGammaRegressionFitter (native)
    Gamma              → lifelines.GeneralizedGammaRegressionFitter (superset)
"""
from __future__ import annotations

import autograd.numpy as np
from autograd.scipy.stats import norm
from lifelines.fitters import ParametricRegressionFitter

from .math_utils import rbetainc

__all__ = [
    "GenInvGaussAFTFitter",
    "InvGaussAFTFitter",
    "JohnsonSUAFTFitter",
    "GenLogisticAFTFitter",
    "GenWeibullAFTFitter",
    "GenFAFTFitter",
    "MielkeAFTFitter",
]

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _BaseFitter(ParametricRegressionFitter):
    """Common helper methods for all custom AFT fitters."""

    _scipy_fit_method = "L-BFGS-B"

    @staticmethod
    def _h_from_survival(surv: np.ndarray) -> np.ndarray:
        """Cumulative hazard from a survival probability array."""
        return -np.log(np.clip(surv, _EPS, 1.0))


# ---------------------------------------------------------------------------
# Mielke Beta-Kappa (Burr XII)
# ---------------------------------------------------------------------------

class MielkeAFTFitter(_BaseFitter):
    """
    Mielke Beta-Kappa (Burr XII / Singh-Maddala) AFT model.

    Parameterisation
    ----------------
    S(t | λ, k, s) = (1 + (t / λ)^k)^{−s/k}

    Cumulative hazard
    -----------------
    H(t | λ, k, s) = (s/k) · log(1 + (t/λ)^k)

    where λ = exp(Xβ) is the AFT scale/location parameter (primary).
    Shape parameters k > 0 and s > 0 are intercept-only by default.

    Notes
    -----
    * When k = s the distribution collapses to Log-Logistic.
    * Equivalent to scipy.stats.mielke / scipy.stats.burr12 up to
      reparameterisation.

    Regressor template
    ------------------
    _REGS_MIELKE = {
        "lambda_": ["dist_scaled", "ang_scaled", "disp_scaled", "Intercept"],
        "k_":      ["Intercept"],
        "s_":      ["Intercept"],
    }
    """

    _fitted_parameter_names = ["lambda_", "k_", "s_"]
    _primary_parameter_name = "lambda_"
    _scipy_fit_options      = {"method": "SLSQP"}

    def _cumulative_hazard(
        self,
        params: dict,
        T:      np.ndarray,
        Xs:     dict,
    ) -> np.ndarray:
        lam = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))  # scale λ > 0
        k   = np.exp(np.dot(Xs["k_"],      params["k_"]))       # shape k > 0
        s   = np.exp(np.dot(Xs["s_"],      params["s_"]))       # shape s > 0

        z   = np.clip(T / lam, 1e-16, None)   # t/λ, guarded against 0
        z_k = np.power(z, k)                   # (t/λ)^k

        # H = (s/k) * log(1 + z^k)
        return (s / k) * np.log1p(z_k)


# ---------------------------------------------------------------------------
# Generalized Inverse Gaussian
# ---------------------------------------------------------------------------

class GenInvGaussAFTFitter(_BaseFitter):
    """
    Generalized Inverse Gaussian AFT model.

    Uses the log-normal survival approximation via the GIG log-CDF.

    Parameterisation
    ----------------
    λ = exp(Xβ)  — location (primary)
    p            — tail shape  (intercept-only)
    b = exp(·)   — concentration (intercept-only)

    Regressor template
    ------------------
    _REGS_GENINVGAUSS = {
        "lambda_": ["dist_scaled", "ang_scaled", "disp_scaled", "Intercept"],
        "p_":      ["Intercept"],
        "b_":      ["Intercept"],
    }
    """

    _fitted_parameter_names = ["lambda_", "p_", "b_"]
    _primary_parameter_name = "lambda_"

    def _cumulative_hazard(self, params: dict, T: np.ndarray, Xs: dict) -> np.ndarray:
        lam = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))
        p   = np.dot(Xs["p_"],             params["p_"])
        b   = np.exp(np.dot(Xs["b_"],      params["b_"]))

        z = (np.log(T / lam) - p * np.log(b)) / np.sqrt(2.0 / b)
        S = 1.0 - norm.cdf(z)
        return self._h_from_survival(S)


# ---------------------------------------------------------------------------
# Inverse Gaussian  (Wald)                                              ← NEW
# ---------------------------------------------------------------------------

class InvGaussAFTFitter(_BaseFitter):
    """
    Inverse Gaussian (Wald) AFT model.

    Parameterisation
    ----------------
    T ~ IG(μ, λ)  with μ = exp(Xβ) and λ = exp(intercept).

    CDF (Schrödinger 1915 form)
    ----------------------------
    F(t) = Φ( √(λ/t) · (t/μ − 1) )
         + exp(2λ/μ) · Φ( −√(λ/t) · (t/μ + 1) )

    Survival / Cumulative hazard
    ----------------------------
    S(t) = Φ( −√(λ/t) · (t/μ − 1) )
         − exp(2λ/μ) · Φ( −√(λ/t) · (t/μ + 1) )

    H(t) = −log S(t)

    Numerical notes
    ---------------
    The term exp(2λ/μ) can be very large.  We compute the second Φ term
    in log-space and subtract in a numerically safe way.

    Regressor template
    ------------------
    _REGS_INVGAUSS = {
        "mu_":  ["dist_scaled", "ang_scaled", "disp_scaled", "Intercept"],
        "lam_": ["Intercept"],
    }
    """

    _fitted_parameter_names = ["mu_", "lam_"]
    _primary_parameter_name = "mu_"

    def _cumulative_hazard(self, params: dict, T: np.ndarray, Xs: dict) -> np.ndarray:
        mu  = np.exp(np.dot(Xs["mu_"],  params["mu_"]))   # mean  > 0
        lam = np.exp(np.dot(Xs["lam_"], params["lam_"]))  # shape > 0

        sqrt_lam_over_t = np.sqrt(lam / np.clip(T, _EPS, None))

        a = sqrt_lam_over_t * (T / mu - 1.0)   # argument of first  Φ term
        b = sqrt_lam_over_t * (T / mu + 1.0)   # argument of second Φ term

        # S(t) = Φ(-a) − exp(2λ/μ) · Φ(-b)
        # Compute second term in log-space for stability:
        #   log[ exp(2λ/μ) · Φ(-b) ] = 2λ/μ + log Φ(-b)
        log_term2 = 2.0 * lam / mu + norm.logcdf(-b)
        term1     = norm.cdf(-a)
        term2     = np.exp(log_term2)

        S = np.clip(term1 - term2, _EPS, 1.0)
        return self._h_from_survival(S)


# ---------------------------------------------------------------------------
# Johnson SU
# ---------------------------------------------------------------------------

class JohnsonSUAFTFitter(_BaseFitter):
    """
    Johnson SU AFT model.

    Parameterisation
    ----------------
    T ~ JSU(γ, δ, λ, ξ)  with λ = exp(Xβ) (scale, primary), ξ intercept-only.

    CDF
    ---
    F(t) = Φ( γ + δ · arcsinh((t − ξ) / λ) )

    Regressor template
    ------------------
    _REGS_JOHNSONSU = {
        "gamma_":  ["dist_scaled", "ang_scaled", "disp_scaled", "Intercept"],
        "delta_":  ["Intercept"],
        "lambda_": ["Intercept"],
        "xi_":     ["Intercept"],
    }
    """

    _fitted_parameter_names = ["gamma_", "delta_", "lambda_", "xi_"]
    _primary_parameter_name = "gamma_"
    _scipy_fit_method       = "BFGS"

    def _cumulative_hazard(self, params: dict, T: np.ndarray, Xs: dict) -> np.ndarray:
        gamma = np.dot(Xs["gamma_"],   params["gamma_"])
        delta = np.exp(np.dot(Xs["delta_"],  params["delta_"]))
        lam   = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))
        xi    = np.dot(Xs["xi_"],      params["xi_"])

        z = gamma + delta * np.arcsinh((T - xi) / lam)
        S = 1.0 - norm.cdf(z)
        return self._h_from_survival(S)


# ---------------------------------------------------------------------------
# Generalized Logistic
# ---------------------------------------------------------------------------

class GenLogisticAFTFitter(_BaseFitter):
    """
    Generalized Logistic (type I / Fisk-Burr) AFT model.

    H(t) = log(1 + (t/λ)^ρ)

    Regressor template
    ------------------
    _REGS_GENLOGISTIC = {
        "lambda_": ["dist_scaled", "ang_scaled", "disp_scaled", "Intercept"],
        "rho_":    ["Intercept"],
    }
    """

    _fitted_parameter_names = ["lambda_", "rho_"]
    _primary_parameter_name = "lambda_"
    _scipy_fit_method       = "BFGS"

    def _cumulative_hazard(self, params: dict, T: np.ndarray, Xs: dict) -> np.ndarray:
        lam = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))
        rho = np.exp(np.dot(Xs["rho_"],    params["rho_"]))
        return np.log1p((T / lam) ** rho)


# ---------------------------------------------------------------------------
# Generalized Weibull  (Exponentiated Weibull / Mudholkar-Srivastava)   ← NEW
# ---------------------------------------------------------------------------

class GenWeibullAFTFitter(_BaseFitter):
    """
    Generalized Weibull (Exponentiated Weibull) AFT model.

    Also known as Mudholkar-Srivastava or Exponentiated Weibull.
    scipy equivalent: ``scipy.stats.exponweib``.

    Parameterisation
    ----------------
    S(t | λ, k, α) = 1 − (1 − exp(−(t/λ)^k))^α

    where λ = exp(Xβ) is the AFT scale/location parameter (primary),
    k > 0 is the inner Weibull shape, α > 0 is the exponent (tail thickness).

    Cumulative hazard
    -----------------
    H(t) = −log( 1 − (1 − exp(−(t/λ)^k))^α )

    Notes
    -----
    * When α = 1  it collapses to the standard Weibull.
    * When k = 1  it is the Exponentiated Exponential.
    * Computed in log-space where possible for numerical stability.

    Regressor template
    ------------------
    _REGS_GENWEIBULL = {
        "lambda_": ["dist_scaled", "ang_scaled", "disp_scaled", "Intercept"],
        "k_":      ["Intercept"],
        "alpha_":  ["Intercept"],
    }
    """

    _fitted_parameter_names = ["lambda_", "k_", "alpha_"]
    _primary_parameter_name = "lambda_"

    def _cumulative_hazard(self, params: dict, T: np.ndarray, Xs: dict) -> np.ndarray:
        lam   = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))  # scale  > 0
        k     = np.exp(np.dot(Xs["k_"],      params["k_"]))       # shape1 > 0
        alpha = np.exp(np.dot(Xs["alpha_"],  params["alpha_"]))   # shape2 > 0

        # u = (t/λ)^k  — clipped away from zero
        u = np.power(np.clip(T / lam, _EPS, None), k)

        # v = exp(−u) ∈ (0, 1)
        v = np.exp(-u)

        # w = 1 − exp(−u) ∈ (0, 1)  — clipped for log stability
        w = np.clip(1.0 - v, _EPS, 1.0 - _EPS)

        # S = 1 − w^α  — clipped away from zero
        S = np.clip(1.0 - np.power(w, alpha), _EPS, 1.0)

        return -np.log(S)


# ---------------------------------------------------------------------------
# Generalised F  (rarely used, kept for completeness)
# ---------------------------------------------------------------------------

class GenFAFTFitter(_BaseFitter):
    """Generalised F AFT model (four-parameter)."""

    _fitted_parameter_names = ["lambda_", "p_", "q_", "gamma_"]
    _primary_parameter_name = "lambda_"

    def _cumulative_hazard(self, params: dict, T: np.ndarray, Xs: dict) -> np.ndarray:
        mu    = np.dot(Xs["gamma_"],   params["gamma_"])
        sigma = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))
        Q     = np.dot(Xs["q_"],       params["q_"])
        P     = np.exp(np.dot(Xs["p_"], params["p_"]))

        tmp   = Q ** 2 + 2.0 * P
        delta = np.sqrt(tmp)
        w     = (np.log(T) - mu) * delta / sigma
        s1    = 2.0 / (tmp + Q * delta)
        s2    = 2.0 / (tmp - Q * delta)
        x     = s2 / (s2 + s1 * np.exp(w))
        S     = rbetainc(s2, s1, x)
        return self._h_from_survival(S)