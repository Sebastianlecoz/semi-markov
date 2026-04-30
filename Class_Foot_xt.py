"""
foot_semi_xt.py
---------------
Main class for the Semi-Markov Expected Threat model.

Typical usage
-------------
    from build_dataset import build_dataset
    from foot_semi_xt import Foot_semi_xt

    sequences, df, features_df = build_dataset(competitions={55: [282, 43]})

    model = Foot_semi_xt(sequences, df, features_df)

    # --- Distribution fitting ---
    model.evaluate_sejourn_distributions()
    model.QQ_plot("Pass", "Carry")
    model.Survival_plot("Pass", "Carry")
    model.Sejourn_distribution_table("Pass", "Carry")
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from football_model.distribution.parametric_fits import (
    extract_transitions,
    build_duration_dict,
    filter_positive_durations,
)
from football_model.distribution.gof_tests import ks_p_simple, ks_bootstrap_parametric
from football_model.distribution.sampler import get_sampler_for_model


# Reuse helpers from fits_dist_qq_plot without touching that file
from fits_dist_qq_plot import (
    get_all_parametric_models_detailed,
    get_best_parametric_model,
    get_top_distributions,
    get_cdf_function,
    get_ppf_function,
    analyze_transition_workflow,
    process_all_transitions,
    plot_qq,
    plot_model_fit,
)

from sim_modules.AFTLIB import AFTLibrary
from football_model.aft.train import fit_all_in_memory
from football_model.aft.mapping import (
    TransitionSpec,
    TRANSITION_MAP,
    spec_from_scipy,
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

_ABSORBING_STATES = {"Goal", "Loss", "Stoppage", "Out", "Foul"}
_ABSORBING_NON_GOAL = {"Loss", "Stoppage", "Out", "Foul"}
_GOAL_STATE = "Goal"
_FALLBACK_MEAN_SOJOURN = 1.8  # seconds — used when no AFT model is found


def _xy_to_zone(x: float, y: float, nx: int, ny: int) -> int:
    """Map pitch coordinates (metres) to a flat zone index."""
    xi = int(np.clip(x / 120.0 * nx, 0, nx - 1))
    yi = int(np.clip(y / 80.0 * ny, 0, ny - 1))
    return xi * ny + yi


def _zone_to_xy(zone: int, nx: int, ny: int) -> Tuple[float, float]:
    """Return the centre coordinates of a zone."""
    xi = zone // ny
    yi = zone % ny
    return (xi + 0.5) / nx * 120.0, (yi + 0.5) / ny * 80.0


def _zone_to_dist_ang(zone: int, nx: int, ny: int) -> Tuple[float, float]:
    """Distance and angle from a zone centre to the goal (120, 40)."""
    x, y = _zone_to_xy(zone, nx, ny)
    dx, dy = 120.0 - x, 40.0 - y
    return float(np.sqrt(dx * dx + dy * dy)), float(np.arctan2(dy, dx))


def _zone_displacement(z_from: int, z_to: int, nx: int, ny: int) -> float:
    """Euclidean distance (metres) between centres of two zones."""
    x1, y1 = _zone_to_xy(z_from, nx, ny)
    x2, y2 = _zone_to_xy(z_to, nx, ny)
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


# ============================================================================
# PRIVATE HELPER CLASS 1 — Baseline timed-Markov xT grid
# ============================================================================

class _BaselineTimedGridXT:
    """
    Timed Markov xT grid built purely from training data.

    Algorithm
    ---------
    1. Build a (state, cell) -> {(next_state, next_cell): prob} transition map
       from the training events.
    2. Run timed value iteration:

           V(s, z, t) = GoalProb(s, z)
                        + Σ_j  p_ij * V(j, ns, t - mean_sojourn_ij)

       where the look-back index respects the mean sojourn time of each
       (state_i, state_j) pair (computed from the data, not hard-coded).

    No external files are read or written.
    """

    def __init__(self, bins_x: int = 16, bins_y: int = 12) -> None:
        self.bins_x = bins_x
        self.bins_y = bins_y

        self._timed_surface: Optional[np.ndarray] = None  # (n_states, n_cells, n_bins+1)
        self._states: Optional[List[str]] = None
        self._delta_t: Optional[float] = None
        self._horizon: Optional[float] = None



    # ── grid helpers ─────────────────────────────────────────────────────────

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(np.clip(x / (120.0 / self.bins_x), 0, self.bins_x - 1))
        iy = int(np.clip(y / (80.0 / self.bins_y), 0, self.bins_y - 1))
        return iy, ix

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
            self,
            train_events: List[dict],
            mean_times: Dict[str, float],
            delta_t: float = 1.0,
            horizon: float = 10.0,
    ) -> "_BaselineTimedGridXT":
        """
        Parameters
        ----------
        train_events : list of dicts with keys:
            x, y, next_x, next_y, type, next_type, is_goal
            (produced by Foot_semi_xt._build_train_events)
        mean_times   : dict  "StateA -> StateB" -> mean sojourn (seconds)
            (produced by Foot_semi_xt._compute_mean_times)
        delta_t      : time-step in seconds
        horizon      : maximum look-ahead horizon in seconds
        """
        self._delta_t = delta_t
        self._horizon = horizon
        n_cells = self.bins_x * self.bins_y

        # ── 1. Discover transient play-states ────────────────────────────────
        transient = {ev["type"] for ev in train_events if ev["type"] not in _ABSORBING_STATES}
        self._states = sorted(transient)
        n_states = len(self._states)
        s_idx = {s: i for i, s in enumerate(self._states)}

        # ── 2. Accumulate raw counts ─────────────────────────────────────────
        counts: np.ndarray = np.zeros((n_states, n_cells))
        goal_prob: np.ndarray = np.zeros((n_states, n_cells))
        loss_prob: np.ndarray = np.zeros((n_states, n_cells))
        trans_map: Dict[Tuple, Dict[Tuple, int]] = {}

        for ev in train_events:
            curr = ev.get("type", "")
            if curr not in s_idx:
                continue

            i = s_idx[curr]
            iy, ix = self._get_cell(ev["x"], ev["y"])
            s = iy * self.bins_x + ix
            counts[i, s] += 1

            nxt = ev.get("next_type", "End")

            if ev.get("is_goal", False) or nxt == _GOAL_STATE:
                goal_prob[i, s] += 1
            elif nxt in _ABSORBING_NON_GOAL:
                # Absorb into loss — do NOT forward this mass
                loss_prob[i, s] += 1
            else:
                niy, nix = self._get_cell(ev["next_x"], ev["next_y"])
                ns = niy * self.bins_x + nix
                j = s_idx.get(nxt, i)
                key = (i, s)
                trans_map.setdefault(key, {})
                trans_map[key][(j, ns)] = trans_map[key].get((j, ns), 0) + 1

        # ── 3. Normalise ─────────────────────────────────────────────────────
        for i in range(n_states):
            for s in range(n_cells):
                c = counts[i, s]
                if c > 0:
                    goal_prob[i, s] /= c
                    loss_prob[i, s] /= c
                    if (i, s) in trans_map:
                        for k in trans_map[(i, s)]:
                            trans_map[(i, s)][k] /= c


        # ── 4. Timed value iteration ─────────────────────────────────────────
        # Surface shape: (n_states, n_cells, n_bins+1)
        # Index 0 = t=0  (all zeros by construction — no goal yet)
        # Index t = value given t*delta_t seconds remaining
        n_bins = int(horizon / delta_t)
        surface = np.zeros((n_states, n_cells, n_bins + 1), dtype=np.float64)

        # Precompute lambda per (si, sj) pair
        lam = {}
        for i, si in enumerate(self._states):
            for j, sj in enumerate(self._states):
                tau = mean_times.get(f"{si} -> {sj}", _FALLBACK_MEAN_SOJOURN)
                lam[(i, j)] = delta_t / max(tau, 1e-6)

        # Flatten trans_map into a list of edges — avoids dict lookups in loop
        edges = []
        for (i, s), successors in trans_map.items():
            for (j, ns), p_ij in successors.items():
                edges.append((i, s, j, ns, p_ij))

        # Running convolution: conv[(i,j)] shape (n_cells,)
        # conv[(i,j)][ns] = Σ_l λ(1-λ)^l · surface[j, ns, t-1-l]
        # Updated each step via: conv = λ·surface[j,:,t-1] + (1-λ)·conv
        running_conv = {
            (i, j): np.zeros(n_cells, dtype=np.float64)
            for i in range(n_states)
            for j in range(n_states)
        }

        for t_bin in range(1, n_bins + 1):
            # Update all running convolutions — 16 numpy ops over n_cells
            for i in range(n_states):
                for j in range(n_states):
                    l = lam[(i, j)]
                    running_conv[(i, j)] = (
                            l * surface[j, :, t_bin - 1]
                            + (1.0 - l) * running_conv[(i, j)]
                    )

            # Accumulate edge contributions into surface
            surface[:, :, t_bin] = goal_prob
            for i, s, j, ns, p_ij in edges:
                surface[i, s, t_bin] += p_ij * running_conv[(i, j)][ns]

        self._timed_surface = surface
        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, state: str, x: float, y: float, horizon: float) -> float:
        """Return xT for (state, x, y) given a remaining time budget of `horizon` s."""
        if self._timed_surface is None:
            raise RuntimeError("Call fit() first.")
        iy, ix = self._get_cell(x, y)
        s = iy * self.bins_x + ix
        idx = self._states.index(state) if state in self._states else 0
        t_bin = int(np.clip(
            horizon / self._delta_t, 0, self._timed_surface.shape[2] - 1
        ))
        return float(self._timed_surface[idx, s, t_bin])


# ============================================================================
# PRIVATE HELPER CLASS 2 — In-memory semi-Markov xT kernel
# ============================================================================

class _SemiMarkovKernel:
    """
    Semi-Markov xT kernel over the joint (play_state, grid_zone) state space.

    Algorithm (identical to CombinedSemiMarkovKernel)
    -------------------------------------------------
    For each time-step k the xT vector is updated via:

        v_k = h_k ⊙ (M_base @ v_{k-1} + goal_contrib) + (1 - h_k) ⊙ v_{k-1}

    where h_k(i) is the CONDITIONAL hazard at step k for combined state i,
    obtained from the fitted AFT survival models (or an exponential fallback).

    Key difference vs the original file
    ------------------------------------
    The surface is stored entirely in RAM — no cache files are written or read.
    The AFTLibrary instance is passed in directly (already attached to
    Foot_semi_xt._aft_lib).
    """

    def __init__(self, nx: int = 16, ny: int = 12, delta_t: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.n_zones = nx * ny
        self.delta_t = delta_t



        self.play_states: List[str] = []
        self.combined_states: List[Tuple] = []
        self.state_to_idx: Dict[Tuple, int] = {}
        self.transition_probs: Dict[Tuple, Dict[Tuple, float]] = {}
        self.transition_counts: Dict[Tuple, Dict[Tuple, float]] = {}

        self._xt_surface: Optional[np.ndarray] = None
        self._surface_horizon: Optional[float] = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, train_events: List[dict]) -> "_SemiMarkovKernel":
        """
        Learn the (play_state, zone) transition probabilities from train_events.
        """
        play_states_seen: set = set()
        counts: Dict[Tuple, Dict[Tuple, int]] = {}

        for ev in train_events:
            ps_from = ev.get("type", "Unknown")
            ps_to = ev.get("next_type", "End")
            z_from = _xy_to_zone(ev.get("x", 60.0), ev.get("y", 40.0), self.nx, self.ny)
            z_to = _xy_to_zone(ev.get("next_x", 60.0), ev.get("next_y", 40.0), self.nx, self.ny)

            cs_from = (ps_from, z_from)
            cs_to = (ps_to, z_to)
            play_states_seen.update([ps_from, ps_to])
            counts.setdefault(cs_from, {})
            counts[cs_from][cs_to] = counts[cs_from].get(cs_to, 0) + 1

        probs: Dict[Tuple, Dict[Tuple, float]] = {}
        for cs_from, nexts in counts.items():
            total = sum(nexts.values())
            probs[cs_from] = {k: v / total for k, v in nexts.items()}

        self.transition_probs = probs
        self.play_states = sorted(play_states_seen)
        self.transition_counts = counts

        # Only non-absorbing states appear as rows in the surface
        non_absorbing = sorted({cs for cs in counts if cs[0] not in _ABSORBING_STATES})
        self.combined_states = non_absorbing
        self.state_to_idx = {cs: i for i, cs in enumerate(non_absorbing)}
        return self

    # ── surface computation ───────────────────────────────────────────────────

    def compute_xt_surface(self, aft_lib, horizon: float = 10.0) -> np.ndarray:
        """
        Build the xT surface and keep it in RAM.
        No files are written.
        """
        print(
            f"  [SemiMarkovKernel] Building surface "
            f"({len(self.combined_states)} combined states, "
            f"horizon={horizon}s, dt={self.delta_t}s)..."
        )
        surface = self._build_surface_sparse(aft_lib, horizon)
        self._xt_surface = surface
        self._surface_horizon = horizon
        return surface

    def _get_hazard_rates(
            self,
            aft_lib,
            from_state: str,
            to_state: str,
            dist: float,
            ang: float,
            n_steps: int,
            disp: Optional[float] = None,
    ) -> np.ndarray:
        import numpy as np

        if hasattr(aft_lib, "get_mass_distribution"):
            masses = aft_lib.get_mass_distribution(
                from_state, to_state, dist, ang, n_steps, self.delta_t, disp=disp
            )
            if masses is not None and len(masses) == n_steps:
                masses = np.asarray(masses, dtype=float)

                # --- FIX 5a: zero-survival floor ----------------------------
                # Once the cumulative mass has consumed virtually all probability
                # mass, survival is negligible and we should not divide by it.
                # Replace any mass beyond that point with 0 so we never divide
                # by a near-zero survival.
                cum_f      = np.cumsum(masses)
                cutoff_idx = np.searchsorted(cum_f, 1.0 - 1e-4)   # first index where S < 1e-4
                if cutoff_idx < n_steps:
                    masses[cutoff_idx:] = 0.0
                    # Renormalise so total mass ≤ 1
                    total = masses.sum()
                    if total > 1.0:
                        masses /= total

                # --- FIX 5b: conservative survival floor --------------------
                cum_f     = np.cumsum(masses)
                surv_prev = np.empty(n_steps)
                surv_prev[0]  = 1.0
                surv_prev[1:] = np.clip(1.0 - cum_f[:-1], 1e-4, 1.0)   # 1e-6 → 1e-4

                hazard = masses / surv_prev
                return np.clip(hazard, 0.0, 1.0)


        print("_______________________________________________________________")
        print("_______________________________________________________________")
        print("Exponential fallback used for semimarkov ")
        print(from_state)
        print('to')
        print(to_state)
        print("_______________________________________________________________")
        print("_______________________________________________________________")
        # Exponential fallback
        lam = self.delta_t / _FALLBACK_MEAN_SOJOURN
        return np.full(n_steps, float(1.0 - np.exp(-lam)))

    def _build_surface_sparse_h_z(self, aft_lib, horizon: float) -> np.ndarray:
        from scipy.sparse import lil_matrix

        n_states = len(self.combined_states)
        n_steps = int(np.ceil(horizon / self.delta_t))

        # ── STEP 1: Static sparse transition matrix (transient → transient) ──
        M_lil = lil_matrix((n_states, n_states), dtype=np.float64)
        for i, cs_from in enumerate(self.combined_states):
            for cs_to, p_ij in self.transition_probs.get(cs_from, {}).items():
                if cs_to[0] not in _ABSORBING_STATES:
                    j = self.state_to_idx.get(cs_to)
                    if j is not None:
                        M_lil[i, j] += p_ij
        M_base = M_lil.tocsr()

        # ── STEP 2: Goal contribution vector ─────────────────────────────────
        goal_contrib = np.zeros(n_states, dtype=np.float64)
        for i, cs_from in enumerate(self.combined_states):
            for cs_to, p_ij in self.transition_probs.get(cs_from, {}).items():
                if cs_to[0] == _GOAL_STATE:
                    goal_contrib[i] += p_ij

        # ── STEP 3: Hazard-rate matrix  (n_steps × n_states) ─────────────────
        # h_matrix[k, i] = weighted average hazard over all outgoing transitions
        # from combined state i at time-step k.
        # Cache key includes z_to so that each (from, to) zone pair gets its
        # own displacement value fed to the AFT model.
        hazard_matrix = np.zeros((n_steps, n_states), dtype=np.float64)
        hazard_cache: Dict[Tuple, np.ndarray] = {}

        for i, cs_from in enumerate(self.combined_states):
            ps_from = cs_from[0]
            z_from = cs_from[1]
            dist, ang = _zone_to_dist_ang(z_from, self.nx, self.ny)

            for cs_to, p_ij in self.transition_probs.get(cs_from, {}).items():
                ps_to = cs_to[0]
                z_to = cs_to[1]
                disp = _zone_displacement(z_from, z_to, self.nx, self.ny)
                cache_key = (ps_from, ps_to, z_from, z_to)

                if cache_key not in hazard_cache:
                    hazard_cache[cache_key] = self._get_hazard_rates(
                        aft_lib, ps_from, ps_to, dist, ang, n_steps, disp=disp
                    )
                hazard_matrix[:, i] += p_ij * hazard_cache[cache_key]

        hazard_matrix = np.clip(hazard_matrix, 0.0, 1.0)

        # ── STEP 4: Forward iteration ─────────────────────────────────────────
        # surface[:, k] = xT values after k time-steps
        surface = np.zeros((n_states, n_steps + 1), dtype=np.float64)
        v = np.zeros(n_states, dtype=np.float64)

        for k in range(1, n_steps + 1):
            h = hazard_matrix[k - 1]
            e = M_base @ v + goal_contrib  # expected value if a transition fires
            v = h * e + (1.0 - h) * v  # mix: fire vs. stay
            surface[:, k] = v

        return surface

    def _build_surface_sparse(self, aft_lib, horizon: float):
        from scipy.sparse import csr_matrix
        import numpy as np

        n_states = len(self.combined_states)
        n_steps = int(np.ceil(horizon / self.delta_t))

        # ── STEP 3: precompute mass_cache and surv_cache (UNCHANGED) ─────────────
        mass_cache = {}
        surv_cache = {}

        for i, cs_from in enumerate(self.combined_states):
            ps_from, z_from = cs_from
            dist, ang = _zone_to_dist_ang(z_from, self.nx, self.ny)
            for cs_to, p_ij in self.transition_probs.get(cs_from, {}).items():
                ps_to, z_to = cs_to
                disp = _zone_displacement(z_from, z_to, self.nx, self.ny)
                key = (ps_from, ps_to, z_from, z_to)
                if key not in mass_cache:
                    hazards = self._get_hazard_rates(
                        aft_lib, ps_from, ps_to, dist, ang, n_steps, disp=disp
                    )
                    surv = np.ones(n_steps + 1)
                    surv[1:] = np.cumprod(np.clip(1.0 - hazards, 0.0, 1.0))
                    surv_cache[key] = surv
                    mass_cache[key] = -np.diff(surv)  # positive: S(k) - S(k+1)

        # ── SPEEDUP 1: precompute s_all[k, i] for ALL k at once ──────────────────
        # s_all[k, i] = Σ_j  p_ij * S_ij(k·dt)   (marginal survival)
        # Shape: (n_steps+1, n_states)
        # Replaces the per-k inner loops that computed s_prev / s_curr.
        s_all = np.zeros((n_steps + 1, n_states), dtype=np.float64)
        for i, cs_from in enumerate(self.combined_states):
            for cs_to, p_ij in self.transition_probs.get(cs_from, {}).items():
                ps_to, z_to = cs_to
                key = (cs_from[0], ps_to, cs_from[1], z_to)
                if key in surv_cache:
                    s_all[:, i] += p_ij * surv_cache[key]  # broadcast over k axis

        # ── SPEEDUP 2: precompute edge list + weight matrix W[k, edge] ───────────
        # Instead of rebuilding M_k (a lil_matrix) from scratch at every step k,
        # we build the sparsity structure ONCE, then fill a 2-D weight array.
        #
        # edge_rows[e], edge_cols[e]  = (i, j) for transient→transient edges
        # goal_edge_i[e], goal_p[e]   = (i, p_ij) for transient→goal edges
        # W_trans[k, e]  = w_ki for transient→transient edge e at step k
        # W_goal[k, e]   = w_ki for transient→goal edge e at step k
        #
        # w_ki = p_ij * mass[k] / s_prev[k-1, i]

        edge_rows, edge_cols = [], []
        edge_p, edge_key = [], []
        goal_rows, goal_p, goal_key = [], [], []

        for i, cs_from in enumerate(self.combined_states):
            for cs_to, p_ij in self.transition_probs.get(cs_from, {}).items():
                ps_to, z_to = cs_to
                key = (cs_from[0], ps_to, cs_from[1], z_to)
                if ps_to == _GOAL_STATE:
                    goal_rows.append(i);
                    goal_p.append(p_ij);
                    goal_key.append(key)
                elif ps_to not in _ABSORBING_STATES:
                    j = self.state_to_idx.get(cs_to)
                    if j is not None:
                        edge_rows.append(i);
                        edge_cols.append(j)
                        edge_p.append(p_ij);
                        edge_key.append(key)

        edge_rows = np.array(edge_rows, dtype=np.int32)
        edge_cols = np.array(edge_cols, dtype=np.int32)

        # Precompute denominator: s_all[0:n_steps, i] for each edge  (shape n_steps × n_edges)
        s_denom_trans = np.maximum(s_all[:-1, edge_rows], 1e-8)  # (n_steps, n_edges)
        s_denom_goal = np.maximum(s_all[:-1, np.array(goal_rows, dtype=np.int32)], 1e-8) if goal_rows else None

        # mass arrays stacked: (n_steps, n_edges)
        mass_trans = np.column_stack(
            [mass_cache[k] for k in edge_key]
        ) if edge_key else np.zeros((n_steps, 0))

        mass_goal = np.column_stack(
            [mass_cache[k] for k in goal_key]
        ) if goal_key else np.zeros((n_steps, 0))

        edge_p_arr = np.array(edge_p, dtype=np.float64)
        goal_p_arr = np.array(goal_p, dtype=np.float64)

        # W_trans[k, e] = p_ij * mass[k] / s_prev[k, i]
        W_trans = (edge_p_arr * mass_trans) / s_denom_trans  # (n_steps, n_edges)
        W_goal = (goal_p_arr * mass_goal) / s_denom_goal if goal_rows else np.zeros((n_steps, 0))

        # Precompute stay factor: s_all[k] / s_all[k-1]  (shape n_steps × n_states)
        stay_all = s_all[1:] / np.maximum(s_all[:-1], 1e-8)  # (n_steps, n_states)

        # Precompute goal contribution vector for each k  (n_steps × n_states)
        g_all = np.zeros((n_steps, n_states), dtype=np.float64)
        if goal_rows:
            goal_rows_arr = np.array(goal_rows, dtype=np.int32)
            np.add.at(g_all.T, goal_rows_arr, W_goal.T)  # equivalent to loop but vectorised

        # ── STEP 4: forward iteration — now only numpy ops inside the loop ────────
        surface = np.zeros((n_states, n_steps + 1), dtype=np.float64)
        v = np.zeros(n_states, dtype=np.float64)

        for k in range(1, n_steps + 1):
            # Build M_k from precomputed weights — O(n_edges) scipy call, no Python loop
            M_k = csr_matrix(
                (W_trans[k - 1], (edge_rows, edge_cols)),
                shape=(n_states, n_states)
            )
            v = M_k @ v + g_all[k - 1] + stay_all[k - 1] * v
            surface[:, k] = v

        return surface

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, play_state: str, x: float, y: float, horizon: float) -> float:
        """Return xT for (play_state, x, y) given a horizon of `horizon` seconds."""
        if self._xt_surface is None:
            raise RuntimeError("Call compute_xt_surface() first.")

        zone = _xy_to_zone(x, y, self.nx, self.ny)
        cs = (play_state, zone)
        idx = self.state_to_idx.get(cs)
        if idx is None:
            idx = self._nearest_state_idx(play_state, zone)
        if idx is None:
            return 0.0

        n_steps = self._xt_surface.shape[1] - 1
        k = min(n_steps, max(1, int(np.ceil(horizon / self.delta_t))))
        return float(self._xt_surface[idx, k])

    def _nearest_state_idx(self, play_state: str, zone: int) -> Optional[int]:
        """
        Fallback: find the closest combined state with the same play_state
        using Manhattan distance on the grid.
        """
        best_idx, best_dist = None, float("inf")
        xi_q, yi_q = zone // self.ny, zone % self.ny
        for cs, idx in self.state_to_idx.items():
            if cs[0] != play_state:
                continue
            d = abs(cs[1] // self.ny - xi_q) + abs(cs[1] % self.ny - yi_q)
            if d < best_dist:
                best_dist, best_idx = d, idx
        return best_idx

    # ── diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "n_combined_states": len(self.combined_states),
            "n_play_states": len(self.play_states),
            "n_zones": self.n_zones,
            "grid": f"{self.nx}x{self.ny}",
            "delta_t": self.delta_t,
            "surface_ready": self._xt_surface is not None,
            "surface_horizon": self._surface_horizon,
        }


class Foot_semi_xt:
    """
    Semi-Markov Expected Threat model.

    Parameters
    ----------
    sequences : list[list[dict]]
        Possession sequences from build_dataset().
    df : pd.DataFrame
        Cleaned events DataFrame from build_dataset().
    features_df : pd.DataFrame
        Transition features DataFrame from build_dataset().
    """

    def __init__(
        self,
        sequences: List[List[dict]],
        df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> None:
        self.sequences   = sequences
        self.df          = df
        self.features_df = features_df

        # --- Distribution fitting results (populated by evaluate_sejourn_distributions) ---
        self._duration_dict:         Optional[Dict[Tuple, List[float]]] = None
        self._distribution_results:  Optional[Dict[str, Any]]           = None

        # --- Dynamic transition map (populated by _build_dynamic_transition_map) ---
        # Maps "From -> To" string keys to TransitionSpec objects.
        # Built from _distribution_results when evaluate_sejourn_distributions() has
        # been called; otherwise falls back to the static TRANSITION_MAP.
        self._dynamic_transition_map: Optional[Dict[str, TransitionSpec]] = None

        # --- AFT results (populated by train_aft) ---
        self._aft_lib:    Optional[AFTLibrary]        = None
        self._aft_models: Optional[Dict[str, Any]]    = None
        self._aft_scaler                              = None

        # --- Train/test split (populated by train_XT_* methods) ---
        self._train_seqs            = None
        self._test_seqs             = None
        self._train_events          = None
        self._seed                  = None

        # --- Model objects (populated by train_XT_* methods) ---
        self._baseline_model        = None
        self._semi_model            = None
        self._original_model        = None

        # --- Model parameters (populated by train_XT_* methods) ---
        self._horizon               = None
        self._delta_t               = None

        # --- Evaluation results (populated by evaluate_* methods) ---
        self._eval_results          = {}

        # Pre-build the duration dict once
        self._build_duration_dict()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_duration_dict(self) -> None:
        """Extract and store per-transition duration lists from sequences."""
        transitions = extract_transitions(self.sequences)
        d           = build_duration_dict(transitions)
        self._duration_dict = filter_positive_durations(d)

    def _check_distributions_fitted(self) -> None:
        if self._distribution_results is None:
            raise RuntimeError(
                "Distribution results not found. "
                "Call evaluate_sejourn_distributions() first."
            )

    def _get_transition_key(
        self, from_state: str, to_state: str
    ) -> Optional[Tuple[str, str]]:
        key = (from_state, to_state)
        if key in self._duration_dict:
            return key
        str_key = str(key)
        if str_key in self._duration_dict:
            return str_key
        return None

    def _get_result_for_transition(
        self, from_state: str, to_state: str
    ) -> Dict[str, Any]:
        self._check_distributions_fitted()
        key     = str((from_state, to_state))
        result  = self._distribution_results.get(key)
        if result is None:
            available = list(self._distribution_results.keys())
            raise KeyError(
                f"Transition ('{from_state}', '{to_state}') not found in results.\n"
                f"Available transitions: {available}"
            )
        return result

    def _get_data_for_transition(
        self, from_state: str, to_state: str
    ) -> np.ndarray:
        key = self._get_transition_key(from_state, to_state)
        if key is None:
            raise KeyError(
                f"No duration data for transition ('{from_state}', '{to_state}'). "
                f"Available: {list(self._duration_dict.keys())}"
            )
        return np.asarray(self._duration_dict[key], dtype=float)

    # -----------------------------------------------------------------------
    # Dynamic transition map
    # -----------------------------------------------------------------------

    def _build_dynamic_transition_map(self) -> Dict[str, TransitionSpec]:
        """
        Build a transition → TransitionSpec mapping from the fitted distribution
        results stored in ``self._distribution_results``.

        For each transition the ``best_model`` field (a scipy distribution name)
        is looked up in ``SCIPY_TO_AFT`` via ``spec_from_scipy()``.  If a
        transition has no fitted result (e.g. too few observations) it falls
        back to the static ``TRANSITION_MAP`` entry, or to WeibullAFT if that
        is also absent.

        Returns
        -------
        dict[str, TransitionSpec]
            Keys are ``"From -> To"`` strings matching ``features_df`` convention.
        """
        self._check_distributions_fitted()

        dynamic_map: Dict[str, TransitionSpec] = {}
        missing_best_model = []

        for str_key, result in self._distribution_results.items():
            # str_key is str(('From', 'To'))  e.g. "('Pass', 'Carry')"
            # Convert to AFT-style key  e.g. "Pass -> Carry"
            try:
                # ast.literal_eval is safest; tuple.__str__ is reversible
                import ast
                pair = ast.literal_eval(str_key)      # → ('Pass', 'Carry')
                aft_key = f"{pair[0]} -> {pair[1]}"
            except Exception:
                log.warning("Could not parse transition key '%s' — skipping.", str_key)
                continue

            best_model = result.get("best_model")
            status     = result.get("status", "")

            if best_model and status.startswith("validated"):
                dynamic_map[aft_key] = spec_from_scipy(best_model)
                log.debug("Dynamic map: %-30s  →  %s  (scipy: %s)",
                          aft_key,
                          dynamic_map[aft_key].model.__name__,
                          best_model)
            else:
                # Distribution was not validated — fall back to static map
                missing_best_model.append(aft_key)
                static_spec = TRANSITION_MAP.get(aft_key)
                if static_spec is not None:
                    dynamic_map[aft_key] = static_spec
                    log.warning(
                        "Transition '%s' has no validated scipy model (status='%s'). "
                        "Using static TRANSITION_MAP fallback: %s.",
                        aft_key, status, static_spec.model.__name__,
                    )
                else:
                    from football_model.aft.mapping import _FALLBACK_SPEC
                    dynamic_map[aft_key] = _FALLBACK_SPEC
                    log.warning(
                        "Transition '%s' has no validated scipy model and no static "
                        "fallback — defaulting to WeibullAFTFitter.",
                        aft_key,
                    )

        if missing_best_model:
            log.info(
                "%d transition(s) fell back to static/default spec: %s",
                len(missing_best_model), missing_best_model,
            )

        self._dynamic_transition_map = dynamic_map
        return dynamic_map

    # -----------------------------------------------------------------------
    # Public API — distribution fitting
    # -----------------------------------------------------------------------

    def evaluate_sejourn_distributions(
        self,
        min_obs: int = 10,
        n_boot:  int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Fit parametric sojourn time distributions for all state transitions.
        Results are stored internally — no files are written.

        After calling this method, ``train_aft()`` will automatically select the
        appropriate AFT family for each transition based on these results, rather
        than using the hardcoded ``TRANSITION_MAP``.

        Parameters
        ----------
        min_obs : int
            Minimum number of observations required to attempt fitting (default 10).
        n_boot : int
            Number of bootstrap replicates for the KS validation (default 100).
        verbose : bool
            Print progress (default True).
        """
        if not verbose:
            logging.disable(logging.CRITICAL)

        log.info(
            "Fitting distributions for %d transitions (min_obs=%d, n_boot=%d)...",
            len(self._duration_dict), min_obs, n_boot,
        )

        self._distribution_results = process_all_transitions(
            self._duration_dict,
            min_obs=min_obs,
            n_boot=n_boot,
            create_plots=False,
            output_dir=None,
        )

        # Invalidate any previously built dynamic map so it is rebuilt fresh
        # the next time train_aft() is called.
        self._dynamic_transition_map = None

        if not verbose:
            logging.disable(logging.NOTSET)

        n_ok = sum(
            1 for r in self._distribution_results.values()
            if r.get("status", "").startswith("validated")
        )
        if verbose:
            print(
                f"Done. {len(self._distribution_results)} transitions fitted, "
                f"{n_ok} validated."
            )

    def _build_train_events(
            self,
            sequences: List[List[dict]],
            horizon: float,
    ) -> List[dict]:
        """
        Flatten possession sequences into a list of event dicts ready for
        grid fitting.

        For each transient event the method computes a binary ``is_goal``
        label using a forward-looking window of exactly ``horizon`` seconds:
          - 1 if a Goal event is reached before accumulating ``horizon``
            seconds of play or hitting a terminal state.
          - 0 otherwise.

        This is the same labelling logic used in compare_models.py, now
        generalised to any horizon value and fully self-contained.

        Parameters
        ----------
        sequences : list of possession sequences (from build_dataset or split)
        horizon   : look-ahead window in seconds

        Returns
        -------
        list of dicts, each with keys:
            x, y, next_x, next_y, type, next_type, is_goal
        """
        TERMINAL = {"Loss", "Stoppage", "Out"}
        events: List[dict] = []

        for seq in sequences:
            for i, ev in enumerate(seq):
                loc = ev.get("location")
                state = ev.get("state")

                # Skip events with no location or that are already terminal
                if not loc or state in (TERMINAL | {_GOAL_STATE}):
                    continue

                # ── Horizon-based goal label ─────────────────────────────────
                is_goal = 0
                accumulated = 0.0

                for k in range(i, len(seq)):
                    if seq[k].get("state") == _GOAL_STATE:
                        is_goal = 1
                        break
                    dur = seq[k].get("duration", 0.0)
                    if accumulated + dur > horizon:
                        break
                    # Stop if the NEXT event is a terminal non-goal state
                    if k + 1 < len(seq) and seq[k + 1].get("state") in TERMINAL:
                        break
                    accumulated += dur

                # ── Next event info ──────────────────────────────────────────
                nxt = seq[i + 1] if i + 1 < len(seq) else None
                nxt_loc = nxt.get("location") if nxt else None
                if nxt_loc is None:
                    nxt_loc = loc  # safe fallback: stay in same cell

                events.append({
                    "x": loc[0],
                    "y": loc[1],
                    "next_x": nxt_loc[0],
                    "next_y": nxt_loc[1],
                    "type": state,
                    "next_type": nxt.get("state") if nxt else "End",
                    "is_goal": is_goal,
                })

        return events

    def _compute_mean_times(self) -> Dict[str, float]:
        """
        Compute mean sojourn time for every state-to-state transition from
        ``self._duration_dict`` (already populated in ``__init__``).

        This replaces the hard-coded ``mean_times`` dict in compare_models.py
        with values derived directly from the current dataset.

        Returns
        -------
        dict
            Keys: ``"StateA -> StateB"``  (e.g. ``"Pass -> Carry"``)
            Values: mean sojourn duration in seconds.
        """
        import ast

        mean_times: Dict[str, float] = {}

        for key, durations in self._duration_dict.items():
            if not durations:
                continue
            # Keys may be stored as tuple or as str(tuple)
            if isinstance(key, str):
                try:
                    pair = ast.literal_eval(key)
                except Exception:
                    continue
            else:
                pair = key
            aft_key = f"{pair[0]} -> {pair[1]}"
            mean_times[aft_key] = float(np.mean(durations))

        return mean_times

    # -----------------------------------------------------------------------
    # Public API — Baseline xT (timed Markov grid)
    # -----------------------------------------------------------------------

    def train_XT_baseline(
            self,
            train_sequences: List[List[dict]],
            test_sequences: List[List[dict]],
            delta_t: float = 1.0,
            horizon: float = 10.0,
            seed: int = 42,
            bins_x: int = 16,
            bins_y: int = 12,
    ) -> None:
        """
        Train the timed Markov baseline xT model.

        The transition probabilities and goal probabilities are estimated
        directly from ``train_sequences``.  Mean sojourn times per
        (state_i → state_j) pair are computed from ``self._duration_dict``
        (i.e. from the same dataset used to initialise the class), so no
        hard-coded look-up table is needed.

        Parameters
        ----------
        train_sequences : possession sequences used for training
        test_sequences  : possession sequences held out for evaluation;
                          stored in the class for later use by evaluate_*
        delta_t         : time-step in seconds (resolution of the surface)
        horizon         : look-ahead window in seconds — controls BOTH the
                          goal-labelling of training events AND the depth of
                          the timed value iteration
        seed            : random seed for reproducibility
        bins_x, bins_y  : pitch grid resolution (default 16×12)
        """
        np.random.seed(seed)

        # ── Store for later evaluate_* calls ────────────────────────────────
        self._train_seqs = train_sequences
        self._test_seqs = test_sequences
        self._horizon = horizon
        self._delta_t = delta_t
        self._seed = seed

        # ── Build flat event list ────────────────────────────────────────────
        print("Building training events for baseline model...")
        train_events = self._build_train_events(train_sequences, horizon)
        print(f"  {len(train_events):,} events constructed "
              f"(horizon={horizon}s, {len(train_sequences):,} sequences).")

        # ── Mean sojourn times from data ─────────────────────────────────────
        mean_times = self._compute_mean_times()
        print(f"  {len(mean_times)} mean sojourn times derived from dataset.")

        # ── Fit ──────────────────────────────────────────────────────────────
        print(f"Fitting baseline timed grid (delta_t={delta_t}s, horizon={horizon}s)...")
        model = _BaselineTimedGridXT(bins_x=bins_x, bins_y=bins_y)
        model.fit(train_events, mean_times, delta_t=delta_t, horizon=horizon)

        self._baseline_model = model
        print("Done. Baseline model stored in self._baseline_model.")

    def predict_XT_baseline(
            self,
            state: str,
            x: float,
            y: float,
    ) -> float:
        """
        Predict xT from the timed Markov baseline model.

        Parameters
        ----------
        state : play state at the moment of possession
                (e.g. ``"Pass"``, ``"Carry"``, ``"Shot"``)
        x     : pitch x-coordinate in metres (0 = own goal-line, 120 = opp.)
        y     : pitch y-coordinate in metres (0–80)

        Returns
        -------
        float
            Expected threat value in [0, 1].

        Raises
        ------
        RuntimeError
            If ``train_XT_baseline()`` has not been called.
        """
        if self._baseline_model is None:
            raise RuntimeError(
                "Baseline model not trained. Call train_XT_baseline() first."
            )
        return self._baseline_model.predict(state, x, y, self._horizon)

    # -----------------------------------------------------------------------
    # Public API — Semi-Markov xT
    # -----------------------------------------------------------------------

    def train_XT_semi(
            self,
            train_sequences: List[List[dict]],
            test_sequences: List[List[dict]],
            delta_t: float = 1.0,
            horizon: float = 10.0,
            seed: int = 42,
            bins_x: int = 16,
            bins_y: int = 12,
    ) -> None:
        """
        Train the semi-Markov xT model.

        The hazard rates that modulate the value-iteration update at each
        time-step are derived from the AFT survival models already fitted by
        ``train_aft()``.  This method therefore requires ``train_aft()`` to
        have been called first.

        The xT surface is held entirely in RAM (no cache files).

        Parameters
        ----------
        train_sequences : possession sequences used for training
        test_sequences  : possession sequences held out for evaluation
        delta_t         : time-step in seconds
        horizon         : look-ahead window in seconds (goal labelling + surface depth)
        seed            : random seed
        bins_x, bins_y  : pitch grid resolution (default 16×12)

        Raises
        ------
        RuntimeError
            If ``train_aft()`` has not been called.
        """
        if self._aft_lib is None:
            raise RuntimeError(
                "AFT library not found. "
                "Call evaluate_sejourn_distributions() and train_aft() first."
            )

        np.random.seed(seed)

        # ── Store for later evaluate_* calls ────────────────────────────────
        self._train_seqs = train_sequences
        self._test_seqs = test_sequences
        self._horizon = horizon
        self._delta_t = delta_t
        self._seed = seed

        # ── Build flat event list ────────────────────────────────────────────
        print("Building training events for semi-Markov model...")
        train_events = self._build_train_events(train_sequences, horizon)
        print(f"  {len(train_events):,} events constructed "
              f"(horizon={horizon}s, {len(train_sequences):,} sequences).")

        # ── Fit the transition kernel ────────────────────────────────────────
        print(f"Fitting semi-Markov kernel (grid={bins_x}×{bins_y}, delta_t={delta_t}s)...")
        kernel = _SemiMarkovKernel(nx=bins_x, ny=bins_y, delta_t=delta_t)
        kernel.fit(train_events)
        print(f"  Kernel fitted: {len(kernel.combined_states):,} combined states.")

        # ── Compute xT surface (in RAM, using self._aft_lib) ─────────────────
        print(f"Computing xT surface (horizon={horizon}s)...")
        print("  This may take a few minutes depending on grid size and horizon.")
        kernel.compute_xt_surface(self._aft_lib, horizon=horizon)

        self._semi_model = kernel
        print("Done. Semi-Markov model stored in self._semi_model.")

    def predict_XT_semi(
            self,
            state: str,
            x: float,
            y: float,
    ) -> float:
        """
        Predict xT from the semi-Markov model.

        Parameters
        ----------
        state : play state at the moment of possession
                (e.g. ``"Pass"``, ``"Carry"``, ``"Shot"``)
        x     : pitch x-coordinate in metres (0–120)
        y     : pitch y-coordinate in metres (0–80)

        Returns
        -------
        float
            Expected threat value in [0, 1].

        Raises
        ------
        RuntimeError
            If ``train_XT_semi()`` has not been called.
        """
        if self._semi_model is None:
            raise RuntimeError(
                "Semi-Markov model not trained. Call train_XT_semi() first."
            )
        return self._semi_model.predict(state, x, y, self._horizon)

    # -----------------------------------------------------------------------
    # Public API — AFT training
    # -----------------------------------------------------------------------

    def train_aft(
        self,
        quiet: bool = False,
        timeline_points: int = 200,
    ) -> None:
        """
        Train AFT (Accelerated Failure Time) survival models for all
        state-to-state transitions.

        If ``evaluate_sejourn_distributions()`` has been called, the AFT family
        for each transition is chosen **dynamically** from those fitted results
        (best scipy distribution → closest lifelines AFT family via
        ``spec_from_scipy()``).  Otherwise the static ``TRANSITION_MAP`` from
        ``mapping.py`` is used as a fallback.

        Results are stored internally — nothing is written to disk.

        Parameters
        ----------
        quiet : bool
            Suppress INFO logs during fitting (default False).
        timeline_points : int
            Resolution of the survival timeline used during fitting (default 200).
        """
        # ── 1. Decide which transition map to use ───────────────────────────
        if self._distribution_results is not None:
            # evaluate_sejourn_distributions() was called → build dynamic map
            if self._dynamic_transition_map is None:
                # Not yet built (or invalidated by a re-run of evaluate_*)
                self._build_dynamic_transition_map()

            transition_map = self._dynamic_transition_map

            # Summary for the user
            family_counts: Dict[str, int] = {}
            for spec in transition_map.values():
                name = spec.model.__name__
                family_counts[name] = family_counts.get(name, 0) + 1
            print("Dynamic AFT family selection (from sojourn distribution fits):")
            for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
                print(f"  {family:<42} × {count}")
        else:
            log.warning(
                "evaluate_sejourn_distributions() has not been called. "
                "Falling back to the static TRANSITION_MAP in mapping.py. "
                "Run evaluate_sejourn_distributions() first for data-driven family selection."
            )
            transition_map = TRANSITION_MAP

        # ── 2. Fit ──────────────────────────────────────────────────────────
        print(f"\nTraining AFT models for {len(transition_map)} transitions "
              f"(this may take several minutes)...")

        fitted_models, scaler = fit_all_in_memory(
            self.features_df,
            transition_map=transition_map,   # ← pass dynamic map here
            quiet=quiet,
            timeline_points=timeline_points,
        )

        self._aft_models = fitted_models
        self._aft_scaler = scaler
        self._aft_lib    = AFTLibrary(fitted_models, scaler=scaler)

        print(f"Done. {len(fitted_models)} AFT models trained and stored.")

    # -----------------------------------------------------------------------
    # Public API — inspection helpers
    # -----------------------------------------------------------------------

    def aft_parameters(
        self,
        from_state: str,
        to_state: str,
    ) -> pd.DataFrame:
        """
        Print and return the parameter table for a fitted AFT model.

        Parameters
        ----------
        from_state : str
            Origin state (e.g. "Pass").
        to_state : str
            Destination state (e.g. "Carry").

        Returns
        -------
        pd.DataFrame
            Columns: param, covariate, coef, exp(coef), se(coef), p, AIC.
        """
        if self._aft_models is None:
            raise RuntimeError("AFT models not trained. Call train_aft() first.")

        key   = f"{from_state} -> {to_state}"
        model = self._aft_models.get(key)

        if model is None:
            available = list(self._aft_models.keys())
            raise KeyError(
                f"No AFT model found for '{from_state} -> {to_state}'.\n"
                f"Available transitions: {available}"
            )

        summary = model.summary.reset_index()
        aic     = getattr(model, "AIC_", None)

        col_map = {col: "p" for col in summary.columns if col == "p-value"}
        if col_map:
            summary = summary.rename(columns=col_map)

        keep  = [c for c in ["param", "covariate", "coef", "exp(coef)", "se(coef)", "p"]
                 if c in summary.columns]
        table = summary[keep].copy()
        table["AIC"] = aic

        # Show which family was actually used (dynamic vs static)
        family_source = "dynamic" if self._dynamic_transition_map else "static"
        if self._dynamic_transition_map and key in self._dynamic_transition_map:
            family_name = self._dynamic_transition_map[key].model.__name__
        else:
            family_name = type(model).__name__

        print(f"\n{'='*60}")
        print(f"AFT model — {from_state} → {to_state}  [{family_source} selection]")
        print(f"  Model family  : {family_name}")
        print(f"  AIC           : {f'{aic:.2f}' if aic is not None else 'N/A'}")
        print(f"{'='*60}")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

        return table

    def aft_family_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame showing which AFT family was selected for each
        transition and whether it came from the dynamic (data-driven) or
        static (hardcoded) map.

        Useful for auditing the dynamic family selection after calling
        ``evaluate_sejourn_distributions()`` and ``train_aft()``.

        Returns
        -------
        pd.DataFrame
            Columns: transition, scipy_best, aft_family, source.
        """
        self._check_distributions_fitted()

        rows = []
        dmap = self._dynamic_transition_map or {}

        for str_key, result in self._distribution_results.items():
            import ast
            try:
                pair    = ast.literal_eval(str_key)
                aft_key = f"{pair[0]} -> {pair[1]}"
            except Exception:
                continue

            scipy_best = result.get("best_model", "N/A")
            status     = result.get("status",     "N/A")

            if aft_key in dmap:
                aft_family = dmap[aft_key].model.__name__
                source     = "dynamic" if status.startswith("validated") else "static_fallback"
            elif aft_key in TRANSITION_MAP:
                aft_family = TRANSITION_MAP[aft_key].model.__name__
                source     = "static_only"
            else:
                aft_family = "WeibullAFTFitter"
                source     = "default_fallback"

            rows.append({
                "transition": aft_key,
                "scipy_best": scipy_best,
                "status":     status,
                "aft_family": aft_family,
                "source":     source,
            })

        df = pd.DataFrame(rows).sort_values("transition").reset_index(drop=True)
        print(df.to_string(index=False))
        return df

    # -----------------------------------------------------------------------
    # Visualisation helpers (unchanged)
    # -----------------------------------------------------------------------

    def QQ_plot(
        self,
        from_state: str,
        to_state:   str,
        top_n:      int = 4,
    ) -> None:
        result = self._get_result_for_transition(from_state, to_state)
        data   = self._get_data_for_transition(from_state, to_state)

        all_models_details = result.get("all_models_details", {})
        if not all_models_details:
            print(f"No model details available for ('{from_state}', '{to_state}').")
            return

        top_models = get_top_distributions(all_models_details, top_n=top_n)
        if not top_models:
            print("No successfully fitted models to plot.")
            return

        n           = len(data)
        MAX_POINTS  = 5_000
        data_sorted = np.sort(data)
        if n > MAX_POINTS:
            idx         = np.linspace(0, n - 1, MAX_POINTS, dtype=int)
            empirical_q = data_sorted[idx]
            plot_n      = MAX_POINTS
        else:
            empirical_q = data_sorted
            plot_n      = n
        probs = (np.arange(1, plot_n + 1) - 0.5) / plot_n

        ncols = min(2, len(top_models))
        nrows = int(np.ceil(len(top_models) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 5 * nrows),
                                 squeeze=False)
        palette    = sns.color_palette("tab10", len(top_models))
        global_lo  = np.percentile(empirical_q, 1)  * 0.95
        global_hi  = np.percentile(empirical_q, 99) * 1.05

        for ax_idx, ((model_name, details), color) in enumerate(zip(top_models, palette)):
            ax     = axes[ax_idx // ncols][ax_idx % ncols]
            params = details["parameters"]
            score  = details.get("qq_score", np.inf)
            bic    = details.get("bic", np.inf)

            try:
                ppf = get_ppf_function(model_name, params)
                tq  = ppf(probs)
            except Exception as e:
                ax.set_title(f"{model_name}\n(error: {e})", fontsize=11)
                continue

            mask = np.isfinite(tq)
            ax.scatter(tq[mask], empirical_q[mask],
                       color=color, s=18, alpha=0.65, zorder=3, label="Observed")
            ax.plot([global_lo, global_hi], [global_lo, global_hi],
                    "k--", linewidth=1.4, label="y = x")
            ax.set_xlim(global_lo, global_hi)
            ax.set_ylim(global_lo, global_hi)
            ax.set_xlabel("Theoretical quantiles", fontsize=11)
            ax.set_ylabel("Empirical quantiles",   fontsize=11)
            ax.set_title(f"{model_name}  (QQ={score:.4f} | BIC={bic:.0f})",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_aspect("equal", adjustable="box")
            if ax_idx == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor("goldenrod")
                    spine.set_linewidth(3)

        for idx in range(len(top_models), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            f"QQ plots — {from_state} → {to_state}  (n={n})\n"
            "Gold border = best model by QQ score",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        plt.show()

    def Survival_plot(
        self,
        from_state: str,
        to_state:   str,
        top_n:      int = 3,
    ) -> None:
        result = self._get_result_for_transition(from_state, to_state)
        data   = self._get_data_for_transition(from_state, to_state)

        all_models_details = result.get("all_models_details", {})
        top_models_list    = get_top_distributions(all_models_details, top_n=top_n)
        if not top_models_list:
            print("No successfully fitted models to plot.")
            return

        top_models = {
            name: details["parameters"]
            for name, details in top_models_list
        }

        from scipy.stats import cumfreq
        res      = cumfreq(data, numbins=len(data))
        x_vals   = res.lowerlimit + res.binsize * (np.arange(res.cumcount.size) + 0.5)
        surv_emp = 1 - res.cumcount / res.cumcount[-1]

        x       = np.linspace(max(0, data.min()), data.max(), 300)
        palette = sns.color_palette("tab10", len(top_models))

        plt.figure(figsize=(10, 6))
        plt.step(x_vals, surv_emp, where="mid",
                 label="Empirical", color="black", linewidth=2)

        for (model_name, params), color in zip(top_models.items(), palette):
            cdf        = get_cdf_function(model_name, params)
            surv_model = 1 - cdf(x)
            score      = all_models_details[model_name].get("qq_score", np.inf)
            plt.plot(x, surv_model, linestyle="--", linewidth=2,
                     label=f"{model_name} (QQ={score:.4f})",
                     alpha=0.85, color=color)

        plt.xlabel("Duration (s)", fontsize=13)
        plt.ylabel("Survival probability S(t)", fontsize=13)
        plt.title(f"Survival curve — {from_state} → {to_state}  (n={len(data)})",
                  fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.show()

    def Sejourn_distribution_table(
        self,
        from_state: str,
        to_state:   str,
    ) -> pd.DataFrame:
        result             = self._get_result_for_transition(from_state, to_state)
        all_models_details = result.get("all_models_details", {})

        rows = []
        for model_name, details in all_models_details.items():
            rows.append({
                "model":     model_name,
                "qq_score":  details.get("qq_score", np.inf),
                "bic":       details.get("bic",      np.inf),
                "aic":       details.get("aic",      np.inf),
                "ks_stat":   details.get("ks_stat"),
                "ks_p":      details.get("ks_p"),
                "status":    details.get("status",   "unknown"),
            })

        table = (
            pd.DataFrame(rows)
            .sort_values("qq_score")
            .reset_index(drop=True)
        )

        best_model = result.get("best_model", "N/A")
        n_obs      = result.get("n_obs",      0)
        val_status = result.get("status",     "N/A")
        val_method = result.get("validation_method", "N/A")

        # Also show which AFT family this would map to
        aft_spec   = spec_from_scipy(best_model) if best_model != "N/A" else None
        aft_family = aft_spec.model.__name__ if aft_spec else "N/A"

        print(f"\n{'='*60}")
        print(f"Transition: {from_state} → {to_state}")
        print(f"  Observations  : {n_obs}")
        print(f"  Best model    : {best_model}  (by QQ score)")
        print(f"  AFT family    : {aft_family}  ← will be used by train_aft()")
        print(f"  Validation    : {val_status}  [{val_method}]")
        print(f"{'='*60}")
        print(table.to_string(index=True, float_format=lambda x: f"{x:.5f}"))
        print()

        return table

    def evaluate_xt(
            self,
            verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate and compare the baseline and semi-Markov xT models on the
        held-out test sequences stored in ``self._test_seqs``.

        The horizon and delta_t are read directly from the trained models and
        must match — if they differ, a RuntimeError is raised before any
        evaluation is run.

        Metrics computed
        ----------------
        - ROC-AUC     : discrimination ability
        - Brier score : calibration / mean squared error
        - Log-loss    : cross-entropy
        - Mean xT     : average predicted value (sanity check)

        A per-state breakdown is also computed so you can see which action
        types each model handles better.

        Parameters
        ----------
        verbose : bool
            Print the report to stdout (default True).

        Returns
        -------
        pd.DataFrame
            Summary table with one row per model and columns:
            model, auc, brier, logloss, mean_xT, n_events, pos_rate.
            Also stored in ``self._eval_results["summary"]``.

        Raises
        ------
        RuntimeError
            If either model has not been trained yet, if test sequences are
            missing, or if the two models were trained with different horizons
            or delta_t values.
        """
        from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
        from tqdm import tqdm

        if self._baseline_model is None:
            raise RuntimeError(
                "Baseline model not trained. Call train_XT_baseline() first."
            )
        if self._semi_model is None:
            raise RuntimeError(
                "Semi-Markov model not trained. Call train_XT_semi() first."
            )
        if self._test_seqs is None:
            raise RuntimeError(
                "No test sequences found. "
                "Pass test_sequences to train_XT_baseline() / train_XT_semi()."
            )

        # ── Read horizon & delta_t from the models themselves ────────────────
        # _BaselineTimedGridXT stores ._horizon and ._delta_t (set in fit()).
        # _SemiMarkovKernel stores ._surface_horizon and .delta_t (set in
        # compute_xt_surface / __init__).
        h_base = self._baseline_model._horizon
        h_semi = self._semi_model._surface_horizon
        dt_base = self._baseline_model._delta_t
        dt_semi = self._semi_model.delta_t

        if h_base is None:
            raise RuntimeError(
                "Baseline model has no stored horizon. "
                "Was _BaselineTimedGridXT.fit() called correctly? "
                "Make sure self._horizon = horizon is set inside fit()."
            )
        if h_semi is None:
            raise RuntimeError(
                "Semi-Markov model has no stored surface horizon. "
                "Was compute_xt_surface() called? Call train_XT_semi() first."
            )
        if abs(h_base - h_semi) > 1e-6:
            raise RuntimeError(
                f"Horizon mismatch: baseline was trained with horizon={h_base}s "
                f"but semi-Markov was trained with horizon={h_semi}s. "
                f"Re-train both models with the same horizon before evaluating."
            )
        if abs(dt_base - dt_semi) > 1e-6:
            raise RuntimeError(
                f"delta_t mismatch: baseline used delta_t={dt_base}s "
                f"but semi-Markov used delta_t={dt_semi}s. "
                f"Re-train both models with the same delta_t before evaluating."
            )

        h = h_base  # confirmed equal
        dt = dt_base  # confirmed equal

        # ── Collect predictions ──────────────────────────────────────────────
        y_true, p_base, p_semi, states_log, seq_ids = [], [], [], [], []

        total_events = sum(len(s) for s in self._test_seqs)
        pbar = tqdm(total=total_events, desc="Evaluating models", unit="ev")

        TERMINAL = {"Loss", "Stoppage", "Out", "Foul"}

        for seq_idx, seq in enumerate(self._test_seqs):
            for j, ev in enumerate(seq):
                pbar.update(1)

                loc = ev.get("location")
                state = ev.get("state")

                if not loc or state in (TERMINAL | {_GOAL_STATE}):
                    continue

                # ── Ground truth: did a goal occur within horizon? ───────────
                is_goal = 0
                accumulated = 0.0
                for k in range(j, len(seq)):
                    if seq[k].get("state") == _GOAL_STATE:
                        is_goal = 1
                        break
                    dur = seq[k].get("duration", 0.0)
                    if accumulated + dur > h:
                        break
                    if k + 1 < len(seq) and seq[k + 1].get("state") in TERMINAL:
                        break
                    accumulated += dur

                x_coord, y_coord = loc[0], loc[1]

                pb = self._baseline_model.predict(state, x_coord, y_coord, h)
                ps = self._semi_model.predict(state, x_coord, y_coord, h)

                y_true.append(is_goal)
                p_base.append(pb)
                p_semi.append(ps)
                states_log.append(state)
                seq_ids.append(seq_idx)

        pbar.close()

        y_true = np.array(y_true, dtype=float)
        p_base = np.clip(np.array(p_base, dtype=float), 1e-7, 1 - 1e-7)
        p_semi = np.clip(np.array(p_semi, dtype=float), 1e-7, 1 - 1e-7)
        n_ev = len(y_true)
        pos_r = float(y_true.mean()) if n_ev else 0.0

        # ── Global metrics ───────────────────────────────────────────────────
        def _metrics(y, p, name):
            return {
                "model": name,
                "auc": roc_auc_score(y, p),
                "brier": brier_score_loss(y, p),
                "logloss": log_loss(y, p),
                "mean_xT": float(p.mean()),
                "std_xT": float(p.std()),
                "n_events": n_ev,
                "pos_rate": pos_r,
            }

        rows = [
            _metrics(y_true, p_base, "Baseline (timed Markov)"),
            _metrics(y_true, p_semi, "Semi-Markov xT"),
        ]
        summary_df = pd.DataFrame(rows)

        # ── Per-state breakdown ──────────────────────────────────────────────
        states_arr = np.array(states_log)
        state_rows = []
        for s in sorted(set(states_log)):
            mask = states_arr == s
            if mask.sum() < 10:
                continue
            yt = y_true[mask]
            pb = p_base[mask]
            ps = p_semi[mask]
            for preds, name in [(pb, "Baseline"), (ps, "Semi-Markov")]:
                try:
                    auc = roc_auc_score(yt, preds)
                except Exception:
                    auc = float("nan")
                state_rows.append({
                    "state": s,
                    "model": name,
                    "n": int(mask.sum()),
                    "auc": auc,
                    "brier": brier_score_loss(yt, preds),
                    "logloss": log_loss(yt, np.clip(preds, 1e-7, 1 - 1e-7)),
                    "mean_xT": float(preds.mean()),
                })
        state_df = pd.DataFrame(state_rows)

        # ── Store ────────────────────────────────────────────────────────────
        self._eval_results = {
            "summary": summary_df,
            "per_state": state_df,
            "y_true": y_true,
            "p_base": p_base,
            "p_semi": p_semi,
            "states": states_arr,
            "horizon": h,
            "delta_t": dt,
            "seq_ids": np.array(seq_ids, dtype=int),
        }

        # ── Print report ─────────────────────────────────────────────────────
        if verbose:
            SEP = "=" * 60
            print(f"\n{SEP}")
            print(f"  xT MODEL EVALUATION")
            print(SEP)
            print(f"  Horizon  (h)    : {h}s")
            print(f"  Time-step (dt)  : {dt}s")
            print(f"  Test sequences  : {len(self._test_seqs):,}")
            print(f"  Events scored   : {n_ev:,}")
            print(f"  Goal rate       : {pos_r:.4f}  ({int(y_true.sum())} goals)")
            print(f"{'-' * 60}")

            for _, row in summary_df.iterrows():
                print(f"\n  {row['model']}")
                print(f"    AUC      : {row['auc']:.4f}")
                print(f"    Brier    : {row['brier']:.4f}")
                print(f"    Log-loss : {row['logloss']:.4f}")
                print(f"    Mean xT  : {row['mean_xT']:.4f}  (±{row['std_xT']:.4f})")

            d_auc = rows[1]["auc"] - rows[0]["auc"]
            d_brier = rows[1]["brier"] - rows[0]["brier"]
            d_ll = rows[1]["logloss"] - rows[0]["logloss"]
            print(f"\n  Delta (Semi-Markov − Baseline):")
            print(f"    ΔAUC      : {d_auc:+.4f}  {'✓' if d_auc > 0 else '✗'}")
            print(f"    ΔBrier    : {d_brier:+.4f}  {'✓' if d_brier < 0 else '✗'}")
            print(f"    ΔLog-loss : {d_ll:+.4f}  {'✓' if d_ll < 0 else '✗'}")

            if not state_df.empty:
                print(f"\n{'-' * 60}")
                print("  Per-state AUC breakdown:")
                pivot = (
                    state_df.pivot_table(
                        index="state", columns="model", values="auc"
                    )
                    .round(4)
                )
                print(pivot.to_string())

            print(f"\n{'-' * 60}")
            print(f"  [Config used]  horizon={h}s  |  delta_t={dt}s")
            print(f"{SEP}\n")

        return summary_df

    def transition_times_table(self) -> "pd.DataFrame":
        """
        Compute mean, median, Q1, Q3, and IQR for every fitted transition.
        Addresses Reviewer 1 Minor remark 1: 'report medians/IQR'.

        Returns
        -------
        pd.DataFrame sorted by mean duration (descending), with columns:
            transition, N, mean, median, Q1, Q3, IQR
        """
        import ast
        import numpy as np
        import pandas as pd

        if self._duration_dict is None:
            raise RuntimeError("No duration data. Run evaluate_sejourn_distributions() first.")

        rows = []
        for key, durations in self._duration_dict.items():
            if not durations or len(durations) < 2:
                continue

            # Key may be a tuple or a string repr of a tuple
            if isinstance(key, str):
                try:
                    pair = ast.literal_eval(key)
                except Exception:
                    continue
            else:
                pair = key

            d = np.asarray(durations, dtype=float)
            d = d[d > 0]
            if len(d) < 2:
                continue

            q1, median, q3 = np.percentile(d, [25, 50, 75])
            rows.append({
                "transition": f"{pair[0]} → {pair[1]}",
                "N": len(d),
                "mean": round(float(d.mean()), 3),
                "median": round(float(median), 3),
                "Q1": round(float(q1), 3),
                "Q3": round(float(q3), 3),
                "IQR": round(float(q3 - q1), 3),
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )
        print(df.to_string(index=False))
        return df

    def bootstrap_ci(
            self,
            n_boot: int = 1000,
            alpha: float = 0.05,
            seed: int = 42,
    ) -> "pd.DataFrame":
        """
        Bootstrap confidence intervals for AUC, Brier score, and Log-loss,
        resampling at the possession-sequence level to respect autocorrelation.
        Addresses Reviewer 1 Minor remark 2: 'bootstrapped confidence intervals'.

        Requires evaluate_xt() to have been called first.

        Parameters
        ----------
        n_boot : number of bootstrap replicates (default 1000)
        alpha  : significance level; produces (1-alpha) CI (default 0.05 → 95% CI)
        seed   : random seed

        Returns
        -------
        pd.DataFrame with columns:
            model, metric, observed, ci_lower, ci_upper
        """
        from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
        import numpy as np
        import pandas as pd

        if self._eval_results is None:
            raise RuntimeError("No evaluation results. Call evaluate_xt() first.")

        res = self._eval_results
        y_true = res["y_true"]
        p_base = res["p_base"]
        p_semi = res["p_semi"]
        seq_ids = res.get("seq_ids")  # added by CHANGE 1

        if seq_ids is None:
            raise RuntimeError(
                "seq_ids not found in _eval_results.\n"
                "Apply CHANGE 1 to evaluate_xt() and re-run it."
            )

        rng = np.random.default_rng(seed)
        unique_seqs = np.unique(seq_ids)
        n_seqs = len(unique_seqs)

        boot_base = {"auc": [], "brier": [], "logloss": []}
        boot_semi = {"auc": [], "brier": [], "logloss": []}

        for _ in range(n_boot):
            # Resample sequences with replacement
            sampled = rng.choice(unique_seqs, size=n_seqs, replace=True)
            idx = np.concatenate([np.where(seq_ids == s)[0] for s in sampled])

            yt = y_true[idx]
            pb = np.clip(p_base[idx], 1e-7, 1 - 1e-7)
            ps = np.clip(p_semi[idx], 1e-7, 1 - 1e-7)

            # Skip degenerate bootstrap samples (only one class)
            if len(np.unique(yt)) < 2:
                continue

            boot_base["auc"].append(roc_auc_score(yt, pb))
            boot_base["brier"].append(brier_score_loss(yt, pb))
            boot_base["logloss"].append(log_loss(yt, pb))

            boot_semi["auc"].append(roc_auc_score(yt, ps))
            boot_semi["brier"].append(brier_score_loss(yt, ps))
            boot_semi["logloss"].append(log_loss(yt, ps))

        lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

        rows = []
        for model_name, boot_dict, p_obs in [
            ("Baseline (timed Markov)", boot_base, p_base),
            ("Semi-Markov xT", boot_semi, p_semi),
        ]:
            from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss as _ll
            obs = {
                "auc": roc_auc_score(y_true, p_obs),
                "brier": brier_score_loss(y_true, p_obs),
                "logloss": _ll(y_true, p_obs),
            }
            for metric, values in boot_dict.items():
                arr = np.array(values)
                rows.append({
                    "model": model_name,
                    "metric": metric,
                    "observed": round(obs[metric], 4),
                    "ci_lower": round(float(np.percentile(arr, lo)), 4),
                    "ci_upper": round(float(np.percentile(arr, hi)), 4),
                })

        df = pd.DataFrame(rows)

        # Pretty print
        print(f"\n{'=' * 65}")
        print(f"  Bootstrap CI  (n_boot={n_boot}, {int((1 - alpha) * 100)}% CI, "
              f"sequence-level resampling)")
        print(f"{'=' * 65}")
        for model_name in df["model"].unique():
            print(f"\n  {model_name}")
            sub = df[df["model"] == model_name]
            for _, r in sub.iterrows():
                print(f"    {r['metric']:<10}  {r['observed']:.4f}  "
                      f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
        print(f"{'=' * 65}\n")

        return df
    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @property
    def transitions(self) -> List[Tuple[str, str]]:
        """List of all transition pairs that have duration data."""
        if self._duration_dict is None:
            return []
        return list(self._duration_dict.keys())

    @property
    def fitted_transitions(self) -> List[str]:
        """List of transitions for which distribution fitting has been run."""
        if self._distribution_results is None:
            return []
        return list(self._distribution_results.keys())


def transition_times_table(self) -> "pd.DataFrame":
    """
    Compute mean, median, Q1, Q3, and IQR for every fitted transition.
    Addresses Reviewer 1 Minor remark 1: 'report medians/IQR'.

    Returns
    -------
    pd.DataFrame sorted by mean duration (descending), with columns:
        transition, N, mean, median, Q1, Q3, IQR
    """
    import ast
    import numpy as np
    import pandas as pd

    if self._duration_dict is None:
        raise RuntimeError("No duration data. Run evaluate_sejourn_distributions() first.")

    rows = []
    for key, durations in self._duration_dict.items():
        if not durations or len(durations) < 2:
            continue

        # Key may be a tuple or a string repr of a tuple
        if isinstance(key, str):
            try:
                pair = ast.literal_eval(key)
            except Exception:
                continue
        else:
            pair = key

        d = np.asarray(durations, dtype=float)
        d = d[d > 0]
        if len(d) < 2:
            continue

        q1, median, q3 = np.percentile(d, [25, 50, 75])
        rows.append({
            "transition": f"{pair[0]} → {pair[1]}",
            "N": len(d),
            "mean": round(float(d.mean()), 3),
            "median": round(float(median), 3),
            "Q1": round(float(q1), 3),
            "Q3": round(float(q3), 3),
            "IQR": round(float(q3 - q1), 3),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    print(df.to_string(index=False))
    return df