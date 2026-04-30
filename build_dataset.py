"""
build_dataset.py
----------------
Standalone function to build the sequences dataset from StatsBomb data.
No files are written to disk — all data is returned as Python objects
to be passed directly into Foot_semi_xt.

Typical usage
-------------
    from build_dataset import build_dataset, list_competitions

    # Browse what's available
    list_competitions()

    # Build with default competitions (Ligue 1 + EURO)
    sequences, df, features_df = build_dataset()

    # Build with custom competitions
    sequences, df, features_df = build_dataset(
        competitions={55: [282, 43], 7: [235, 108, 27]}
    )
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
from statsbombpy import sb

from football_model.pipeline.events    import load_events, select_columns, sort_events, filter_events, rename_states, reorder_same_timestamp, clean_duration
from football_model.pipeline.synthetic import insert_goal_events, insert_loss_events
from football_model.pipeline.merge     import merge_events
from football_model.pipeline.sequences import build_sequences
from football_model.pipeline.features  import build_transition_features_from_sequences

warnings.filterwarnings(
    "ignore",
    message="credentials were not supplied",
    category=UserWarning,
)

# Default competitions used in the original paper
DEFAULT_COMPETITIONS: dict[int, list[int]] = {
    55: [282, 43],      # UEFA EURO
    7:  [235, 108, 27], # La Liga
}


def list_competitions() -> pd.DataFrame:
    """
    Returns a DataFrame of all available StatsBomb free competitions.
    Use this to find valid competition_id / season_id pairs.

    Example
    -------
        df = list_competitions()
        print(df[["competition_id", "competition_name", "season_id", "season_name"]])
    """
    return sb.competitions()


def build_dataset(
    competitions: Optional[dict[int, list[int]]] = None,
    verbose: bool = True,
) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """
    Run the full data pipeline and return all data as Python objects.
    Nothing is written to disk.

    Parameters
    ----------
    competitions : dict, optional
        Mapping of competition_id -> list of season_ids.
        Defaults to DEFAULT_COMPETITIONS (Ligue 1 + EURO).
        Use `list_competitions()` to browse what's available.

    verbose : bool
        Print progress steps (default True).

    Returns
    -------
    sequences : list[list[dict]]
        List of possession sequences, each a list of event dicts
        with keys: state, duration, location.

    df : pd.DataFrame
        Cleaned and merged events DataFrame.

    features_df : pd.DataFrame
        Transition features DataFrame (one row per state-to-state transition).

    Example
    -------
        sequences, df, features_df = build_dataset(
            competitions={55: [282, 43]}
        )
        model = Foot_semi_xt(sequences, df, features_df)
    """
    if competitions is None:
        competitions = DEFAULT_COMPETITIONS

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # --- Step 1: Download ---------------------------------------------------
    _log("Step 1/4 — Downloading events from StatsBomb...")
    raw_df = load_events(competitions)
    _log(f"  {len(raw_df):,} raw events loaded")

    # --- Step 2: Clean & merge ----------------------------------------------
    _log("Step 2/4 — Cleaning and merging events...")
    df = select_columns(raw_df)
    df = sort_events(df)
    df = filter_events(df)
    df = rename_states(df)
    df = reorder_same_timestamp(df)
    df = clean_duration(df)
    df = insert_goal_events(df)
    df = insert_loss_events(df)
    df = merge_events(df)
    _log(f"  DataFrame shape after cleaning: {df.shape}")

    # --- Step 3: Build sequences --------------------------------------------
    _log("Step 3/4 — Building possession sequences...")
    sequences = build_sequences(df)
    _log(f"  {len(sequences):,} sequences extracted")

    # --- Step 4: Transition features ----------------------------------------
    _log("Step 4/4 — Extracting transition features...")
    features_df = build_transition_features_from_sequences(
        sequences, disable_progress=not verbose
    )
    _log(f"  {len(features_df):,} transition features extracted")

    _log("Done.")
    return sequences, df, features_df
