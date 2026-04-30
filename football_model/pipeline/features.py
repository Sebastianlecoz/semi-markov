# -*- coding: utf-8 -*-
"""
Extraction des *features* pour le modèle de transitions,

Sortie : un DataFrame où chaque ligne représente **UNE** transition
(cur → nxt) avec les covariables suivantes :

| Colonne    | Unité / type | Description                                                                                                                               |
|------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| seq_id     | int          | Identifiant de la séquence (ex : possession)                                                                                              |
| step_idx   | int          | Rang de la transition dans la séquence (0 = première)                                                                                     |
| dist       | mètres       | Distance 2-D entre la position courante *(x₁,y₁)* et le centre du but adversaire *(120,40)*.                                              |
| ang        | radians      | Angle absolu ∈ [0, π] entre le segment (position → but) et l’axe horizontal. Plus l’angle → 0, plus l’action est « centrée face au but ». |
| Duration   | secondes     | Durée exacte de l’action courante (*cur*). Peut être 0 pour certains évènements instantanés.                                              |
| vx, vy     | m·s⁻¹        | Composantes de la vitesse moyenne de l’action (dx/dt, dy/dt).                                                                             |
| from_state | str          | État d’origine de la transition (Pass, Carry, …).                                                                                         |
| to_state   | str          | État de destination (état du prochain évènement).                                                                                         |

"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Coordonnées StatsBomb du milieu du but adverse
CENTER_X, CENTER_Y = 120.0, 40.0

# États qu’on ignore pour déterminer la transition suivante
_BAD_NEXT = {"Goal", "Loss", "Stoppage"}


# --------------------------------------------------------------------------- #
# Helpers robustes pour lire les positions                                    #
# --------------------------------------------------------------------------- #
def _safe_loc(loc) -> Tuple[float, float]:
    """
    Convertit une liste ou tuple de xy → (float, float).
    Si invalide, retourne (0, 0) pour éviter les NaN plus loin.
    """
    if isinstance(loc, (tuple, list)) and len(loc) == 2:
        try:
            return float(loc[0]), float(loc[1])
        except Exception:
            pass
    return 0.0, 0.0


def _safe_end_loc(cur: Dict) -> Tuple[float, float] | None:
    """
    Pour un évènement `cur`, tente de récupérer la position de fin
    (pass_end_location, carry_end_location, …). Si aucune dispo,
    renvoie `None` : on utilisera alors la position du prochain évènement.
    """
    for key in (
        "end_location",
        "pass_end_location",
        "carry_end_location",
        "goalkeeper_end_location",
    ):
        if key in cur and cur[key] is not None:
            return _safe_loc(cur[key])
    return None


# --------------------------------------------------------------------------- #
def build_transition_features_from_sequences(
    sequences: Sequence[List[Dict]],
    disable_progress: bool = False,
    include_absorbing: bool = True,  # Ajout option pour inclure les états absorbants
) -> pd.DataFrame:
    """
    Convertit une liste de séquences d’évènements en DataFrame de
    **transitions** — une ligne par paire (évènement_i, évènement_{i+1}).
    Si include_absorbing=True, inclut la dernière transition vers l'état absorbant avec sa position.
    """
    rows: list[dict] = []

    for seq_id, seq in enumerate(
        tqdm(sequences, desc="Seq", disable=disable_progress)
    ):
        n = len(seq)
        for k, (cur, nxt) in enumerate(zip(seq, seq[1:])):
            if cur.get("state") in _BAD_NEXT:
                continue

            # -------- position courante ---------------------------------
            x1, y1 = _safe_loc(cur.get("location"))

            # -------- position finale de l’action courante --------------
            end = _safe_end_loc(cur) or _safe_loc(nxt.get("location"))
            x2, y2 = end

            # -------- durée brute ---------------------------------------
            try:
                dur = float(str(cur.get("duration", 0)).replace(",", "."))
            except Exception:
                dur = 0.0

            # -------- distance / angle (covariables principales) --------
            dist = np.hypot(CENTER_X - x1, CENTER_Y - y1)          # m
            ang = np.arctan2(CENTER_Y - y1, CENTER_X - x1)    # rad (NON absolu)

            # -------- composantes de vitesse (dx/dt, dy/dt) -------------
            dx, dy = x2 - x1, y2 - y1
            vx, vy = dx / (dur + 1e-6), dy / (dur + 1e-6)

            rows.append(
                {
                    # Identifiants
                    "seq_id": seq_id,
                    "step_idx": k,
                    # Covariables principales (brutes)
                    "dist": dist,
                    "ang": ang,
                    # Positions réelles
                    "x": x1,
                    "y": y1,
                    # Variables contextuelles
                    "Duration": dur,
                    "vx": vx,
                    "vy": vy,
                    # États
                    "from_state": cur.get("state"),
                    "to_state": nxt.get("state"),
                }
            )
    return pd.DataFrame(rows).dropna()
