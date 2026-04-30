# src/football_model/pipeline/merge.py

from tqdm import tqdm
import numpy as np
import pandas as pd

def remove_oneonone_merge_duration(df, absorbing_states=("Goal","Loss","Stoppage")):
    """
    Version optimisée et professionnelle :
    - Ne modifie pas le DataFrame dans la boucle principale
    - Utilise des index et des opérations vectorisées pour transférer les durées et supprimer les lignes
    - Garantit le même résultat que la version originale
    """
    df = df.reset_index(drop=True).copy()
    mask_oneonone = df["states"] == "oneonone"
    to_drop = np.zeros(len(df), dtype=bool)
    duration_add = np.zeros(len(df))

    # Pour chaque ligne oneonone, déterminer l'action à faire
    for i in np.where(mask_oneonone)[0]:
        dur, mid, per = df.at[i, "duration"], df.at[i, "match_id"], df.at[i, "period"]
        # Précédent non absorbant ?
        if i > 0 and df.at[i-1, "match_id"] == mid and df.at[i-1, "period"] == per and df.at[i-1, "states"] not in absorbing_states:
            duration_add[i-1] += dur
            to_drop[i] = True
            continue
        # Suivant non absorbant ?
        if i < len(df)-1 and df.at[i+1, "match_id"] == mid and df.at[i+1, "period"] == per and df.at[i+1, "states"] not in absorbing_states:
            duration_add[i+1] += dur
            to_drop[i] = True
            continue
        # Sinon supprimer
        to_drop[i] = True

    # Appliquer les ajouts de durée
    df["duration"] = df["duration"] + duration_add
    # Supprimer les lignes marquées
    df = df[~to_drop].reset_index(drop=True)
    return df

def merge_consecutive_events_preserve_duration(df, event_states=("Pass","Carry","Pressure","Shot")):
    """
    Version optimisée : fusionne les événements consécutifs de même type en une seule passe vectorisée.
    """
    df = df.reset_index(drop=True).copy()
    # Identifie les groupes consécutifs à fusionner
    mask = (
        (df["states"].isin(event_states)) &
        (df["match_id"].eq(df["match_id"].shift())) &
        (df["period"].eq(df["period"].shift())) &
        (df["states"].eq(df["states"].shift()))
    )
    # Attribue un groupe unique à chaque séquence à fusionner
    group = (~mask).cumsum()
    # Fusionne les durées et garde la première ligne de chaque groupe
    df["duration"] = df.groupby(group)["duration"].transform("sum")
    df = df.groupby(group, as_index=False).first()
    return df

def remove_consecutive_absorbing_states_preserve_duration(df, absorbing_states=("Goal","Loss","Stoppage")):
    """
    Version optimisée : fusionne les états absorbants consécutifs en une seule passe vectorisée.
    """
    df = df.reset_index(drop=True).copy()
    mask = (
        (df["states"].isin(absorbing_states)) &
        (df["match_id"].eq(df["match_id"].shift())) &
        (df["period"].eq(df["period"].shift())) &
        (df["states"].eq(df["states"].shift()))
    )
    group = (~mask).cumsum()
    df["duration"] = df.groupby(group)["duration"].transform("sum")
    df = df.groupby(group, as_index=False).first()
    return df

def merge_events(df):
    df = remove_oneonone_merge_duration(df)
    df = merge_consecutive_events_preserve_duration(df)
    df = remove_consecutive_absorbing_states_preserve_duration(df)
    return df
