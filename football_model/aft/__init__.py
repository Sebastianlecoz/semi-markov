# ==============================================
# src/football_model/aft/__init__.py
# ==============================================
"""High‑level public interface for the AFT sub‑package."""
from .fitters import (  # noqa: F401
    GenInvGaussAFTFitter,
    JohnsonSUAFTFitter,
    GenLogisticAFTFitter,
    GenFAFTFitter,
)
from .train import fit_all  # noqa: F401

__all__ = [
    "GenInvGaussAFTFitter",
    "JohnsonSUAFTFitter",
    "GenLogisticAFTFitter",
    "GenFAFTFitter",
    "fit_all",
]