# ==============================================
# src/football_model/aft/datasets.py
# ==============================================
"""Load + feature-engineer the AFT training set.

Scaling is intentionally NOT done here — train.py owns all scaling so that
a single scaler covering dist, ang, and disp can be saved and reused at
prediction time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

from football_model.pipeline.features import build_transition_features_from_sequences

__all__ = ["load_and_prepare_from_features", "scale_features"]


def scale_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Append *<col>_scaled* columns using StandardScaler."""
    scaler = scaler or StandardScaler().fit(df[cols])
    df_out = df.copy()
    df_out[[f"{c}_scaled" for c in cols]] = scaler.transform(df[cols])
    return df_out, scaler


def _compute_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Euclidean distance (metres) between the start position (x, y) and the
    end position (x_next, y_next) of each action.  Rows where x_next/y_next
    are missing get NaN — downstream dropna() calls handle these gracefully.
    """
    df = df.copy()
    df["disp"] = np.sqrt(
        (df["x_next"] - df["x"]) ** 2 + (df["y_next"] - df["y"]) ** 2
    )
    return df


def load_and_prepare_from_features(pkl_path: Path | str) -> pd.DataFrame:
    """
    Load sequences, build transition features, and add the displacement
    covariate.  No scaling is applied here — train.py owns all scaling and
    saves the scaler so it can be reused at prediction time.
    """
    sequences = joblib.load(pkl_path)
    df = build_transition_features_from_sequences(sequences)

    # ── end-position (x_next, y_next) ────────────────────────────────────
    df["x_next"] = df["x"].shift(-1)
    df["y_next"] = df["y"].shift(-1)
    # nullify end-position across sequence boundaries
    mask = df["seq_id"] != df["seq_id"].shift(-1)
    df.loc[mask, ["x_next", "y_next"]] = None

    # recover end-position for the last event of each sequence
    last_idx = df.groupby("seq_id")["step_idx"].idxmax()
    for seq_id, idx in zip(df.loc[last_idx, "seq_id"], last_idx):
        seq = sequences[seq_id]
        if not seq:
            continue
        last_event = seq[-1]
        loc = last_event.get("location")
        if loc is not None and isinstance(loc, (list, tuple)) and len(loc) >= 2:
            df.at[idx, "x_next"] = loc[0]
            df.at[idx, "y_next"] = loc[1]

    # ── displacement feature ──────────────────────────────────────────────
    df = _compute_displacement(df)

    df["Transition"] = df["from_state"] + " -> " + df["to_state"]
    df["event"] = 1

    required = ["dist", "ang"]
    if not all(c in df.columns for c in required):
        raise ValueError(
            f"Required columns missing: {[c for c in required if c not in df.columns]}. "
            f"Available: {list(df.columns)}"
        )

    n_valid_disp = df["disp"].notna().sum()
    logging.info(
        f"load_and_prepare_from_features: {len(df)} rows, "
        f"{n_valid_disp} with valid disp ({len(df) - n_valid_disp} NaN)"
    )
    return df