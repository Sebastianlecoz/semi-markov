
from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

from combined_semi_markov_kernel import CombinedSemiMarkovKernel, ABSORBING_STATES
from sim_modules.xt import AFTLibrary
from sim_modules.model_utils import load_aft_models

import sklearn.compose._column_transformer
class _RemainderColsList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

warnings.filterwarnings("ignore")

GOAL_STATE    = "Goal"
ALL_ABSORBING = ABSORBING_STATES


# ---------------------------------------------------------------------------
# Infinite-Horizon Absorbing Markov Chain xT
# ---------------------------------------------------------------------------

class InfiniteHorizonMarkovXT:
    def __init__(self, nx: int = 16, ny: int = 12) -> None:
        self.nx = nx
        self.ny = ny
        self.combined_states:  List[Tuple]                     = []
        self.state_to_idx:     Dict[Tuple, int]                = {}
        self.transition_probs: Dict[Tuple, Dict[Tuple, float]] = {}
        self.play_states:      List[str]                       = []
        self._xt:     Dict[Tuple, float] = {}
        self._fitted = False

    def fit(self, train_events: List[dict]) -> "InfiniteHorizonMarkovXT":
        play_states_seen: set = set()
        counts: Dict[Tuple, Dict[Tuple, int]] = {}

        for ev in train_events:
            ps_from = ev.get("type",      "Unknown")
            ps_to   = ev.get("next_type", "End")
            z_from  = self._xy_to_zone(ev.get("x",      60.0), ev.get("y",      40.0))
            z_to    = self._xy_to_zone(ev.get("next_x", 60.0), ev.get("next_y", 40.0))
            cs_from = (ps_from, z_from)
            cs_to   = (ps_to,   z_to)
            play_states_seen.update([ps_from, ps_to])
            counts.setdefault(cs_from, {})
            counts[cs_from][cs_to] = counts[cs_from].get(cs_to, 0) + 1

        probs: Dict[Tuple, Dict[Tuple, float]] = {}
        for cs_from, nexts in counts.items():
            total = sum(nexts.values())
            probs[cs_from] = {k: v / total for k, v in nexts.items()}

        self.transition_probs  = probs
        self.play_states       = sorted(play_states_seen)
        non_absorbing          = sorted({cs for cs in counts if cs[0] not in ALL_ABSORBING})
        self.combined_states   = non_absorbing
        self.state_to_idx      = {cs: i for i, cs in enumerate(non_absorbing)}
        self._solve()
        return self

    def _solve(self) -> None:
        n = len(self.combined_states)
        print(f"  [MarkovXT] Solving ({n} transient states)...")

        Q = lil_matrix((n, n), dtype=np.float64)
        r = np.zeros(n, dtype=np.float64)

        for i, cs_from in enumerate(self.combined_states):
            for cs_to, p_ij in self.transition_probs.get(cs_from, {}).items():
                ps_to = cs_to[0]
                if ps_to == GOAL_STATE:
                    r[i] += p_ij
                elif ps_to not in ALL_ABSORBING:
                    j = self.state_to_idx.get(cs_to)
                    if j is not None:
                        Q[i, j] += p_ij

        Q_csr     = Q.tocsr()
        I_minus_Q = (lil_matrix(np.eye(n, dtype=np.float64)) - Q).tocsr()

        try:
            x = spsolve(I_minus_Q, r)
        except Exception as e:
            print(f"  [MarkovXT] spsolve failed ({e}), falling back to power iteration...")
            x = np.zeros(n)
            for _ in range(1000):
                x_new = r + Q_csr @ x
                if np.max(np.abs(x_new - x)) < 1e-8:
                    break
                x = x_new
            x = x_new

        x            = np.clip(x, 0.0, 1.0)
        self._xt     = {cs: float(x[i]) for i, cs in enumerate(self.combined_states)}
        self._fitted = True
        print(f"  [MarkovXT] Solved. xT range: [{x.min():.4f}, {x.max():.4f}]")

    def predict(self, play_state: str, x: float, y: float) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        cs = (play_state, self._xy_to_zone(x, y))
        return self._xt.get(cs) or self._nearest_xt(play_state, self._xy_to_zone(x, y))

    def _nearest_xt(self, play_state: str, zone: int) -> float:
        best_val, best_dist = 0.0, float("inf")
        xi_q, yi_q = zone // self.ny, zone % self.ny
        for cs, val in self._xt.items():
            if cs[0] != play_state:
                continue
            d = abs(cs[1] // self.ny - xi_q) + abs(cs[1] % self.ny - yi_q)
            if d < best_dist:
                best_dist, best_val = d, val
        return best_val

    def _xy_to_zone(self, x: float, y: float) -> int:
        xi = int(np.clip(x / 120.0 * self.nx, 0, self.nx - 1))
        yi = int(np.clip(y / 80.0  * self.ny, 0, self.ny - 1))
        return xi * self.ny + yi

    def summary(self) -> dict:
        vals = list(self._xt.values())
        return {
            "n_combined_states": len(self.combined_states),
            "xT_mean": float(np.mean(vals)) if vals else 0.0,
            "xT_max":  float(np.max(vals))  if vals else 0.0,
        }


# ---------------------------------------------------------------------------
# Event-level evaluation — goal within horizon window
# ---------------------------------------------------------------------------
def run_comparison(
    test_seqs:    List[List[dict]],
    markov_model: InfiniteHorizonMarkovXT,
) -> Tuple[list, list, list, dict]:
    """
    Evaluate both models at the event level.

    Ground truth: y = 1 if a Goal occurs within `horizon` seconds
                  of the current event (identical to compare_models.py).

    Note: the infinite-horizon Markov prediction ignores the horizon
    parameter — it returns the timeless P(eventually score), which is
    then evaluated against the windowed ground truth.
    """

    y_true, p_markov = [], []
    total_raw = sum(len(s) for s in test_seqs)
    pbar      = tqdm(total=total_raw, desc="Evaluating", colour="cyan")

    for seq in test_seqs:
        for j, event in enumerate(seq):
            pbar.update(1)
            loc   = event.get("location")
            state = event.get("state")
            if not loc or state in ALL_ABSORBING:
                continue
            goal_in_win = 0


            for k in range(j, len(seq)):

                if GOAL_STATE in seq:
                    goal_in_win = 1
                    break

            x, y = loc[0], loc[1]
            y_true.append(goal_in_win)
            p_markov.append(markov_model.predict(state, x, y))

    pbar.close()

    stats = {
        "total_seqs": len(test_seqs),
        "total_eval": len(y_true),
        "pos_rate":   float(np.mean(y_true)) if y_true else 0.0,
    }
    return y_true, p_markov, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(y_true, preds_dict: Dict[str, list], title: str = "REPORT") -> None:
    SEP = "=" * 58
    print(f"\n{SEP}\n {title}\n{SEP}")
    print(f"  Events evaluated: {len(y_true):,}")
    print(f"  Positive rate:    {np.mean(y_true):.4f}")
    print("-" * 58)
    for name, preds in preds_dict.items():
        yp = np.array(preds)
        print(f"\n  {name}:")
        print(f"    AUC:     {roc_auc_score(y_true, yp):.4f}")
        print(f"    Brier:   {brier_score_loss(y_true, yp):.4f}")
        print(f"    LogLoss: {log_loss(y_true, np.clip(yp, 1e-7, 1-1e-7)):.4f}")
        print(f"    Pred mean/std:  {yp.mean():.4f} / {yp.std():.4f}")
        print(f"    Pred range:     [{yp.min():.4f}, {yp.max():.4f}]")
    print(f"\n{SEP}\n")


def save_report(y_true, preds_dict, path, title="REPORT") -> None:
    import io, sys
    buf = io.StringIO()
    sys.stdout, old = buf, sys.stdout
    print_report(y_true, preds_dict, title)
    sys.stdout = old
    with open(path, "w") as f:
        f.write(buf.getvalue())
    print(f"Report saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed set to: {seed}")


def main():
    data_path = "../../../data/processed/all_sequences_ligue_1_EURO.pkl"
    seed      = 42
    set_seed(seed)

    # ----------------------------------------------------------------
    # Load & split
    # ----------------------------------------------------------------
    print("Loading sequences...")
    all_sequences = joblib.load(Path(data_path))
    random.shuffle(all_sequences)

    split_idx  = int(len(all_sequences) * 0.8)
    train_seqs = all_sequences[:split_idx]
    test_pool  = all_sequences[split_idx:]

    goals     = [s for s in test_pool if any(e.get("state") == GOAL_STATE for e in s)]
    non_goals = [s for s in test_pool if not any(e.get("state") == GOAL_STATE for e in s)]
    test_seqs = goals + non_goals

    print(f"  Train: {len(train_seqs):,}  |  Test: {len(test_seqs):,} "
          f"({len(goals):,} with goal, {len(non_goals):,} without)")

    # ----------------------------------------------------------------
    # Build train_events  (identical to compare_models.py)
    # ----------------------------------------------------------------
    print("\nBuilding train events...")
    terminal_failures = {"Loss", "Stoppage", "Out"}
    train_events      = []


    train_events_inf = []  # For the Infinite-Horizon Markov Model


    for seq in tqdm(train_seqs, desc="Processing train sequences"):
        for i in range(len(seq)):
            curr_event = seq[i]
            curr_loc = curr_event.get("location")
            curr_state = curr_event.get("state")

            # Skip if no location or if it's already an absorbing state
            if not curr_loc or curr_state in (terminal_failures | {GOAL_STATE}):
                continue

            # Shared transition info
            next_event = seq[i + 1] if i + 1 < len(seq) else None
            next_loc = (next_event.get("location") if next_event else None) or curr_loc
            next_state = next_event.get("state") if next_event else "End"

            # 1. INFINITE HORIZON EVENT
            # We use the literal next_state (Goal, Pass, etc.) regardless of time.
            train_events_inf.append({
                "x": curr_loc[0],
                "y": curr_loc[1],
                "next_x": next_loc[0],
                "next_y": next_loc[1],
                "type": curr_state,
                "next_type": next_state,
                "is_goal": 1 if any(s.get("state") == GOAL_STATE for s in seq[i:]) else 0
            })




    print(f"  {len(train_events_inf):,} training events processed.")

    # ----------------------------------------------------------------
    # Fit models using their respective data sets
    # ----------------------------------------------------------------
    print("\nFitting Infinite-Horizon Markov xT (using full sequence data)...")
    markov_xt = InfiniteHorizonMarkovXT(nx=16, ny=12)
    # This model now "sees" goals even if they happen at 50 seconds
    markov_xt.fit(train_events_inf)
    print(f"  {markov_xt.summary()}")



    y_true, p_markov, stats = run_comparison(test_seqs, markov_xt)

    title= f"Infinite Markov"
    preds_dict = {
        "Infinite-Horizon Markov xT":           p_markov,
    }

    print_report(y_true, preds_dict, title=title)
    save_report(
        y_true, preds_dict,
        f"results_inf_markov_seed{seed}.txt",
        title=title,
    )



if __name__ == "__main__":
    main()