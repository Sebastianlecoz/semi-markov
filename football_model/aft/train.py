from __future__ import annotations

import joblib
import logging
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Union
import warnings
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .datasets import load_and_prepare_from_features, scale_features
from .mapping import TRANSITION_MAP, TransitionSpec

__all__ = ["fit_all"]
_LOG = logging.getLogger(__name__)

_SIMPLE = {LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter}

# Ordered candidate covariates for each model family.
# disp_scaled is last — stepwise removal hits it before dist/ang if needed.
_SIMPLE_COVARS = ["dist_scaled", "ang_scaled", "disp_scaled"]
_SPLINE_COVARS = ["dist_norm", "ang_norm", "disp_norm"]

# Name of the saved scaler file (covers dist, ang, disp)
SCALER_FILENAME = "feature_scaler.pkl"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Duration", "event", "dist_scaled", "ang_scaled"]).copy()
    if "Intercept" not in df.columns:
        df["Intercept"] = 1.0
    return df


def _compute_timeline(durations: pd.Series, n_points: int = 200) -> np.ndarray:
    return np.linspace(1e-8, durations.max(), n_points)


def _available_covars(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return candidates that exist as columns and have ≥10 non-NaN values."""
    return [c for c in candidates if c in df.columns and df[c].notna().sum() >= 10]


# --------------------------------------------------------------------------- #
# Fit a single transition
# --------------------------------------------------------------------------- #

def _fit_single(
        tran: str,
        spec: TransitionSpec,
        df: pd.DataFrame,
        timeline: Optional[np.ndarray] = None,
) -> Optional[object]:
    base_tran = tran.replace(" (Spline)", "") if tran.endswith(" (Spline)") else tran
    subset = df[df["Transition"] == base_tran]

    min_obs = 1 if tran.endswith(" (Spline)") else 10
    if len(subset) < min_obs:
        _LOG.warning(f"[{tran}] skipped – insufficient data ({len(subset)})")
        return None

    fit_kwargs: Dict[str, Union[np.ndarray, bool]] = {}
    if timeline is not None:
        fit_kwargs["timeline"] = timeline

    # -------------------------------------------------------------------------
    # _SIMPLE models — stepwise removal of dist_scaled, ang_scaled, disp_scaled
    # -------------------------------------------------------------------------
    if spec.model in _SIMPLE:
        family = spec.model
        name = family.__name__

        current_covars = _available_covars(subset, _SIMPLE_COVARS)
        _LOG.info(f"[{tran}] {name}: starting covars = {current_covars}")

        covar_removed = True
        fitted_model = None

        while covar_removed:
            covar_removed = False
            cols = ["Duration", "event"] + current_covars
            df_fit = subset[cols].dropna()

            if len(df_fit) < 10:
                _LOG.warning(
                    f"[{tran}] {name}: only {len(df_fit)} rows after dropna "
                    f"for covars {current_covars}"
                )
                if "disp_scaled" in current_covars:
                    current_covars.remove("disp_scaled")
                    covar_removed = True
                    continue
                break

            try:
                m = family(**spec.kwargs())
                m.fit(df_fit[cols], duration_col="Duration", event_col="event",
                      **fit_kwargs)
                to_remove = []
                for param, cov in m.summary.index:
                    if cov == "Intercept" or cov not in current_covars:
                        continue
                    pval = (m.summary.loc[(param, cov), "p"]
                            if "p" in m.summary.columns
                            else m.summary.loc[(param, cov), "p-value"])
                    if pval > 0.005:
                        to_remove.append(cov)
                if to_remove:
                    for c in set(to_remove):
                        if c in current_covars:
                            current_covars.remove(c)
                    covar_removed = True
                    _LOG.info(f"[{tran}] {name}: removed {to_remove}")
                else:
                    fitted_model = m
                    break
            except Exception as e:
                _LOG.debug(f"[{tran}] {name} fit error: {e}")
                if "disp_scaled" in current_covars:
                    current_covars.remove("disp_scaled")
                    covar_removed = True
                    continue
                break

        # fallback: intercept-only
        if fitted_model is None:
            try:
                m = family(**spec.kwargs())
                m.fit(subset[["Duration", "event"]].dropna(),
                      duration_col="Duration", event_col="event", **fit_kwargs)
                fitted_model = m
                _LOG.info(f"[{tran}] {name}: intercept-only fallback")
            except Exception as e:
                _LOG.error(f"[{tran}] {name}: intercept-only fallback failed: {e}")

        if fitted_model is not None:
            _LOG.info(f"[{tran}] ✅ {name} fitted")
        return fitted_model


    df_cov = _prepare_covariates(subset)
    regressors = spec.regressors.copy() if spec.regressors is not None else {}

    # Drop disp_scaled from regressors if unavailable or too sparse
    if "disp_scaled" not in df_cov.columns or df_cov["disp_scaled"].isna().all():
        _LOG.info(f"[{tran}] disp_scaled unavailable – removing from regressors")
        regressors = {p: [c for c in covs if c != "disp_scaled"]
                      for p, covs in regressors.items()}
    else:
        df_cov_with_disp = df_cov.dropna(subset=["disp_scaled"])
        if len(df_cov_with_disp) < 10:
            _LOG.warning(f"[{tran}] too few rows with disp_scaled – excluding")
            regressors = {p: [c for c in covs if c != "disp_scaled"]
                          for p, covs in regressors.items()}
        else:
            df_cov = df_cov_with_disp

    if "Intercept" not in df_cov.columns:
        df_cov["Intercept"] = 1.0

    covar_removed = True
    while covar_removed and regressors:
        covar_removed = False
        aft_model = spec.model(**spec.kwargs())
        try:
            aft_model.fit(df_cov, duration_col="Duration", event_col="event",
                          regressors=regressors, **fit_kwargs)
        except Exception as exc:
            _LOG.warning(f"[{tran}] fit failed: {exc}")
            # retry without disp_scaled first
            has_disp = any("disp_scaled" in covs for covs in regressors.values())
            if has_disp:
                regressors = {p: [c for c in covs if c != "disp_scaled"]
                              for p, covs in regressors.items()}
                _LOG.info(f"[{tran}] retrying without disp_scaled")
                covar_removed = True
                continue
            # standard fallback chain
            try:
                aft_model = spec.model(penalizer=0.5)
                aft_model.fit(df_cov, duration_col="Duration", event_col="event",
                              regressors=regressors, **fit_kwargs)
            except Exception as exc2:
                _LOG.warning(f"[{tran}] penalizer=0.5 failed: {exc2}")
                try:
                    df_scaled = df_cov.copy()
                    df_scaled["Duration"] /= 100.0
                    aft_model = spec.model(**spec.kwargs())
                    aft_model.fit(df_scaled, duration_col="Duration", event_col="event",
                                  regressors=regressors, **fit_kwargs)
                except Exception as exc3:
                    _LOG.warning(f"[{tran}] scaled durations failed: {exc3}")
                    try:
                        simple_reg = {p: ["Intercept"] for p in regressors}
                        aft_model = spec.model(**spec.kwargs())
                        aft_model.fit(df_cov, duration_col="Duration", event_col="event",
                                      regressors=simple_reg, **fit_kwargs)
                    except Exception as exc4:
                        _LOG.error(f"[{tran}] all attempts failed: {exc4}")
                        try:
                            aft_model = spec.model(**spec.kwargs())
                            aft_model.fit(df_cov[["Duration", "event"]],
                                          duration_col="Duration", event_col="event",
                                          **fit_kwargs)
                        except Exception as exc5:
                            _LOG.error(f"[{tran}] cannot create base model: {exc5}")
                            return None

        to_remove: list[tuple[str, str]] = []
        for param, cov in aft_model.summary.index:
            if cov == "Intercept":
                continue
            pval = (aft_model.summary.loc[(param, cov), "p"]
                    if "p" in aft_model.summary.columns
                    else aft_model.summary.loc[(param, cov), "p-value"])
            if pval > 0.005 and param in regressors and cov in regressors.get(param, []):
                to_remove.append((param, cov))

        if to_remove:
            for param, cov in to_remove:
                regressors[param] = [c for c in regressors[param] if c != cov]
                if not regressors[param]:
                    del regressors[param]
            covar_removed = True
            _LOG.info(f"[{tran}] removed (p>0.005): {to_remove}")
        else:
            break

    if "aft_model" not in locals() or aft_model is None:
        _LOG.warning(f"[{tran}] creating intercept-only model")
        try:
            simple_reg = {p: ["Intercept"] for p in (spec.regressors or {})}
            aft_model = spec.model(**spec.kwargs())
            aft_model.fit(df_cov, duration_col="Duration", event_col="event",
                          regressors=simple_reg, **fit_kwargs)
        except Exception as exc:
            _LOG.error(f"[{tran}] final failure: {exc}")
            return None

    return aft_model



# --------------------------------------------------------------------------- #
# Fit all transitions
# --------------------------------------------------------------------------- #

def fit_all(
        *,
        pkl_path: Path | str,
        quiet: bool | int = False,
        timeline_points: int = 200,
) -> Dict[str, object]:
    level = logging.ERROR if quiet is True else (quiet or logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")

    df = load_and_prepare_from_features(pkl_path=pkl_path)

    # ── Single consistent scaler for dist + ang + disp ────────────────────
    # Only fit on rows where disp is available (non-NaN).
    # Rows missing disp still get dist_scaled/ang_scaled via the same scaler
    # (same mean/std as rows with disp, so features are comparable).
    scaler_cols = ["dist", "ang", "disp"]
    valid_mask = df[scaler_cols].notna().all(axis=1)
    feature_scaler = StandardScaler().fit(df.loc[valid_mask, scaler_cols])

    # Apply to all rows — rows with NaN disp will produce NaN for disp_scaled
    df[["dist_scaled", "ang_scaled", "disp_scaled"]] = feature_scaler.transform(
        df[scaler_cols].fillna(df[scaler_cols].median())  # tmp fill for transform
    )
    # Re-NaN disp_scaled for rows that had no displacement data
    df.loc[~valid_mask, "disp_scaled"] = np.nan

    # ── Save the scaler — THIS is what AFTLibrary will load at predict time ─
    scaler_dir = Path("models/aft")
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / SCALER_FILENAME
    joblib.dump(feature_scaler, scaler_path)
    _LOG.info(f"Saved feature scaler → {scaler_path}")
    _LOG.info(
        f"Scaler means:  dist={feature_scaler.mean_[0]:.3f}  "
        f"ang={feature_scaler.mean_[1]:.3f}  disp={feature_scaler.mean_[2]:.3f}"
    )
    _LOG.info(
        f"Scaler stds:   dist={feature_scaler.scale_[0]:.3f}  "
        f"ang={feature_scaler.scale_[1]:.3f}  disp={feature_scaler.scale_[2]:.3f}"
    )

    # ── MinMax normalise for spline models ────────────────────────────────
    mm = MinMaxScaler(feature_range=(0, 1))
    norm_in = ["dist_scaled", "ang_scaled", "disp_scaled"]
    norm_out = ["dist_norm", "ang_norm", "disp_norm"]
    valid_norm = df[norm_in].notna().all(axis=1)
    mm.fit(df.loc[valid_norm, norm_in])
    df.loc[valid_norm, norm_out] = mm.transform(df.loc[valid_norm, norm_in])

    _LOG.info(
        f"disp_scaled: {valid_mask.sum()} valid rows, "
        f"{(~valid_mask).sum()} NaN rows"
    )

    timeline = _compute_timeline(df["Duration"], n_points=timeline_points)

    fitted: Dict[str, object] = {}
    for tran, spec in TRANSITION_MAP.items():
        _LOG.info("Fitting %-24s …", tran)
        try:
            m = _fit_single(tran, spec, df, timeline=timeline)
            if m is not None:
                if _LOG.isEnabledFor(logging.DEBUG):
                    buf = StringIO()
                    m.print_summary(file=buf)
                    _LOG.debug("\n%s", buf.getvalue())
                fitted[tran] = m
                _LOG.info(f"[{tran}] ✅ done")
        except Exception as exc:
            _LOG.exception(f"[{tran}] ❌ error: {exc}")

    _LOG.info("Finished – %d/%d models fitted", len(fitted), len(TRANSITION_MAP))
    return fitted




def fit_all_in_memory(
    features_df,
    *,
    transition_map: Optional[Dict[str, TransitionSpec]] = None,   # ← NEW
    quiet: bool = False,
    timeline_points: int = 200,
) -> Tuple[Dict[str, Any], Any]:
    """
    In-memory version of fit_all.

    Accepts features_df directly (from Foot_semi_xt.features_df) and returns
    the fitted scaler as a Python object instead of saving it to disk.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of build_transition_features_from_sequences().
    quiet : bool
        Suppress INFO logs when True.
    timeline_points : int
        Resolution of the survival timeline used during fitting.

    Returns
    -------
    fitted : dict[str, model]
        Transition name -> fitted lifelines model.
    scaler : StandardScaler
        Fitted scaler for (dist, ang, disp) — pass directly to AFTLibrary.
    """



    if transition_map is None:
        transition_map = TRANSITION_MAP



    level = logging.ERROR if quiet is True else (quiet or logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")

    df = features_df.copy()

    # end-position columns — re-derive if not already present
    if "x_next" not in df.columns:
        df["x_next"] = df["x"].shift(-1)
        df["y_next"] = df["y"].shift(-1)
        mask = df["seq_id"] != df["seq_id"].shift(-1)
        df.loc[mask, ["x_next", "y_next"]] = None

    # displacement covariate
    if "disp" not in df.columns:
        df["disp"] = np.sqrt(
            (df["x_next"] - df["x"]) ** 2 + (df["y_next"] - df["y"]) ** 2
        )

    df["Transition"] = df["from_state"] + " -> " + df["to_state"]
    df["event"] = 1

    # single consistent scaler for dist + ang + disp
    scaler_cols = ["dist", "ang", "disp"]
    valid_mask = df[scaler_cols].notna().all(axis=1)
    feature_scaler = StandardScaler().fit(df.loc[valid_mask, scaler_cols])

    df[["dist_scaled", "ang_scaled", "disp_scaled"]] = feature_scaler.transform(
        df[scaler_cols].fillna(df[scaler_cols].median())
    )
    df.loc[~valid_mask, "disp_scaled"] = np.nan

    # MinMax normalise for spline models
    mm = MinMaxScaler(feature_range=(0, 1))
    norm_in  = ["dist_scaled", "ang_scaled", "disp_scaled"]
    norm_out = ["dist_norm",   "ang_norm",   "disp_norm"]
    valid_norm = df[norm_in].notna().all(axis=1)
    mm.fit(df.loc[valid_norm, norm_in])
    df.loc[valid_norm, norm_out] = mm.transform(df.loc[valid_norm, norm_in])

    _LOG.info(
        f"disp_scaled: {valid_mask.sum()} valid rows, "
        f"{(~valid_mask).sum()} NaN rows"
    )

    timeline = _compute_timeline(df["Duration"], n_points=timeline_points)

    fitted: Dict[str, object] = {}
    for tran, spec in transition_map.items():
        _LOG.info("Fitting %-24s …", tran)
        try:
            m = _fit_single(tran, spec, df, timeline=timeline)
            if m is not None:
                fitted[tran] = m
                _LOG.info(f"[{tran}] ✅ done")
        except Exception as exc:
            _LOG.exception(f"[{tran}] ❌ error: {exc}")

    _LOG.info("Finished – %d/%d models fitted", len(fitted), len(transition_map))
    return fitted, feature_scaler

