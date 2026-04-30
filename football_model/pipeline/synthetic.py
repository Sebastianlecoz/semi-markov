# src/football_model/pipeline/synthetic.py

import pandas as pd

def insert_goal_events(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute un événement goal juste après chaque shot réussi."""
    df = df.copy()
    mask = (df['type']=='Shot') & (df['shot_outcome']=='Goal')
    new_rows = []
    for idx in df[mask].index:
        base = df.loc[idx].copy()
        base['type']='Goal'
        base['states']='Goal'
        base['duration']=0
        ts = pd.to_timedelta(df.loc[idx,'timestamp']) + pd.Timedelta(milliseconds=1)
        base['timestamp']=str(ts).replace('0 days ','')
        new_rows.append(base)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values(['match_id','period','timestamp']).reset_index(drop=True)
    return df


def insert_loss_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insère un événement 'Loss' quand la possession change.
    On réindexe en positionnel pour pouvoir utiliser df.iloc.
    """
    # 1) reset_index pour avoir un RangeIndex utilisable en positionnel
    df = df.reset_index(drop=True).copy()

    # 2) repérer où la possession change (et qu'on n'est pas déjà sur 'Loss')
    prev_team  = df['possession_team_id'].shift(1)
    prev_state = df['states'].shift(1)
    change_mask = (
        (df['possession_team_id'] != prev_team)  # changement d’équipe
        & (prev_state != 'Loss')                # précédent != Loss
        & (df.index > 0)                        # on est bien à i>0
    )

    # 3) pour chaque i où change_mask est True, on prend la ligne i-1
    new_rows = []
    for i in change_mask[change_mask].index:
        base = df.iloc[i - 1].copy()
        base['type']     = 'Loss'
        base['states']   = 'Loss'
        base['duration'] = 0
        new_rows.append(base)

    # 4) on concatène, on re-tri et on re-reset_index
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = (
            df
            .sort_values(by=['match_id', 'period', 'timestamp'])
            .reset_index(drop=True)
        )

    return df