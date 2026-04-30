# src/football_model/pipeline/events.py

import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from statsbombpy import sb

# On supprime bruyamment le warning "NoAuthWarning: credentials were not supplied"
warnings.filterwarnings(
    "ignore",
    message="credentials were not supplied",
    category=UserWarning,
)


def load_events(competitions: dict) -> pd.DataFrame:
    """
    Fetches all StatsBomb events for a given competitions dict.

    Parameters
    ----------
    competitions : dict
        Mapping of competition_id -> list of season_ids.
        Example: {55: [282, 43], 7: [235, 108, 27]}
        Use `statsbombpy.sb.competitions()` to browse available IDs.

    Returns
    -------
    pd.DataFrame
        Concatenated events DataFrame for all matches found.
    """
    match_ids = []
    print(competitions)
    for competition_id, seasons in competitions.items():
        print(competition_id)
        for season_id in seasons:

            matches_df = sb.matches(
                competition_id=competition_id,
                season_id=season_id,
            )
            match_ids.extend(matches_df["match_id"].tolist())


    df_list = []


    for mid in tqdm(match_ids, desc="Downloading events", unit="match"):
        df_match = sb.events(match_id=mid).assign(match_id=mid)
        df_list.append(df_match)

    return pd.concat(df_list, ignore_index=True)



def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Conserve les colonnes indispensables."""
    cols = [
        'match_id','team_id','team','possession_team_id','position','possession',
        'period','minute','second','timestamp','duration',
        'type','under_pressure','shot_outcome','location',
        'pass_end_location','goalkeeper_end_location','carry_end_location','shot_end_location'
    ]
    return df[[c for c in cols if c in df.columns]].copy()

def sort_events(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(['match_id','period','timestamp']).reset_index(drop=True)

def filter_events(df: pd.DataFrame) -> pd.DataFrame:
    filtered_types = {
        'Tactical Shift','Substitution','Player On','Clearance','Own Goal Against',
        'Own Goal For','Starting XI','Half End','Referee Ball-Drop','Half Start',
        'Ball Receipt*','Goal Keeper'
    }
    df = df[~df['type'].isin(filtered_types)]
    return df[df['period'] <= 4].copy()

def rename_states(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'Duel':'oneonone','Dribble':'oneonone','Dribbled Past':'oneonone',
        '50/50':'oneonone','Foul Won':'oneonone','Interception':'oneonone',
        'Ball Recovery':'oneonone','Dispossessed':'oneonone','Miscontrol':'oneonone',
        'Shield':'oneonone','Block':'oneonone',
        'Error':'Stoppage','Injury Stoppage':'Stoppage',
        'Player Off':'Stoppage','Offside':'Stoppage',
        'Bad Behaviour':'Stoppage','Foul Committed':'Stoppage'
    }
    df['states'] = df['type'].replace(mapping)
    return df

def reorder_same_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ['match_id','period','timestamp','possession_team_id']
    ).reset_index(drop=True)

def clean_duration(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = (
        df['duration']
        .astype(str).str.replace(',', '.')
        .astype(float)
        .fillna(0)
    )
    return df
