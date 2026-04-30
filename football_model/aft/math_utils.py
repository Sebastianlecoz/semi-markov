# ==============================================
# src/football_model/aft/math_utils.py
# ==============================================
"""Custom math helpers (Autograd‑safe)."""
from __future__ import annotations

import autograd.numpy as np
import scipy.special as sp
from autograd.extend import  primitive#,defvjp
#from autograd.scipy.special import beta as asp_beta

__all__ = ["rbetainc"]

# Original Fortran/CExt function
_orig_betainc = sp.betainc


def _unwrap(x):
    for attr in ("val", "_value"):
        if hasattr(x, attr):
            return getattr(x, attr)
    return x


@primitive
def rbetainc(a, b, x):  # noqa: D401 (short imperative)
    """Regularised incomplete beta *Iₓ(a, b)* compatible with Autograd."""
    return _orig_betainc(_unwrap(a), _unwrap(b), _unwrap(x))


# ––– Finite‑difference VJP for *a* and *b* ––– #

# def _vjp_fd(idx: int):
#     def _vjp(ans, a, b, x):  # noqa: ANN001 (Autograd signature)
#         eps = 1e-5
#         args = [a, b, x]
#         args[idx] = args[idx] + eps
#         return lambda g: g * (rbetainc(*args) - ans) / eps
#
#     return _vjp


# ––– Analytical VJP for *x* ––– #

# def _vjp_x(ans, a, b, x):  # noqa: ANN001
#     return lambda g: g * x ** (a - 1) * (1 - x) ** (b - 1) / asp_beta(a, b)


#defvjp(rbetainc, _vjp_fd(0), _vjp_fd(1), _vjp_x)