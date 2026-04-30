"""
qq_sensitivity.py
-----------------
Paste this function anywhere (e.g. at the bottom of fits_dist_qq_plot.py,
or in a new file you import from example.py).
"""

import numpy as np
import pandas as pd
from fits_dist_qq_plot import qq_score  # already defined in your codebase


def qq_sensitivity_analysis(distribution_results, verbose=True):
    """
    For each fitted transition, re-run qq_score across a grid of
    (bulk_pct, tail_factor) values and check if the selected family changes.

    Parameters
    ----------
    distribution_results : dict
        model._distribution_results  (from evaluate_sejourn_distributions)
    verbose : bool
        Print the summary table.

    Returns
    -------
    pd.DataFrame
        One row per transition with stability statistics.
    """
    BULK_PCTS = [0.90, 0.925, 0.95, 0.975]
    TAIL_FACTORS = [2.5, 3.0, 3.5]
    N_CONFIGS = len(BULK_PCTS) * len(TAIL_FACTORS)  # = 12

    rows = []

    for key, result in distribution_results.items():
        reference = result.get("best_model")
        all_models = result.get("all_models_details", {})
        n_obs = result.get("n_obs", 0)

        if reference is None or not all_models:
            continue

        # Keep only successfully fitted candidates with stored parameters
        candidates = {
            name: det for name, det in all_models.items()
            if det.get("status") == "fitted" and det.get("parameters") is not None
        }
        if len(candidates) < 2:
            continue

        # Reconstruct the data array from parameters is not needed —
        # qq_score only needs (data, model_name, params, bulk_pct, tail_factor).
        # BUT we need the raw data. It lives inside result["parameters"] → no.
        # We have to pass data separately.  See note below.
        rows.append({
            "transition": key,
            "N": n_obs,
            "reference_model": reference,
            "candidates": candidates,
        })

    # NOTE: qq_score requires the raw data array, which is NOT stored in
    # distribution_results. Pass model._duration_dict as well (see example below).
    print("Call qq_sensitivity_analysis_with_data() instead — see example.py")
    return pd.DataFrame(rows)


def qq_sensitivity_analysis_with_data(distribution_results, duration_dict, verbose=True):
    """
    Same as above but accepts duration_dict so qq_score can be recomputed.

    Parameters
    ----------
    distribution_results : dict   — model._distribution_results
    duration_dict        : dict   — model._duration_dict
    """
    import ast

    BULK_PCTS = [0.90, 0.925, 0.95, 0.975]
    TAIL_FACTORS = [2.5, 3.0, 3.5]
    N_CONFIGS = len(BULK_PCTS) * len(TAIL_FACTORS)  # 12 combinations

    rows = []

    for str_key, result in distribution_results.items():
        reference = result.get("best_model")
        all_models = result.get("all_models_details", {})

        if reference is None or not all_models:
            continue

        # Recover raw durations ─────────────────────────────────────────
        try:
            pair = ast.literal_eval(str_key)  # e.g. ('Pass', 'Carry')
        except Exception:
            continue

        data = duration_dict.get(pair) or duration_dict.get(str_key)
        if data is None or len(data) < 10:
            continue
        data = np.asarray(data, dtype=float)
        data = data[data > 0]

        # Only candidates that were successfully fitted ──────────────────
        candidates = {
            name: det for name, det in all_models.items()
            if det.get("status") == "fitted" and det.get("parameters") is not None
        }
        if len(candidates) < 2:
            continue

        # Run the grid ──────────────────────────────────────────────────
        n_stable = 0
        changes = []

        for bulk in BULK_PCTS:
            for tail in TAIL_FACTORS:
                scores = {
                    name: qq_score(data, name, det["parameters"],
                                   bulk_pct=bulk, tail_penalty_factor=tail)
                    for name, det in candidates.items()
                }
                finite = {m: s for m, s in scores.items() if np.isfinite(s)}
                if not finite:
                    continue
                winner = min(finite, key=finite.__getitem__)
                if winner == reference:
                    n_stable += 1
                else:
                    changes.append(f"bulk={bulk}/tail={tail}→{winner}")

        label = f"{pair[0]} -> {pair[1]}"
        rows.append({
            "transition": label,
            "N": len(data),
            "reference_model": reference,
            "n_stable": n_stable,
            "stability_pct": round(100 * n_stable / N_CONFIGS, 1),
            "changes": "; ".join(changes) if changes else "—",
        })

    df = pd.DataFrame(rows).sort_values("stability_pct")

    if verbose:
        print("\n" + "=" * 65)
        print("  QQ SENSITIVITY  (bulk ∈ {0.90,0.925,0.95,0.975}  ×  tail ∈ {2.5,3.0,3.5})")
        print("=" * 65)
        for _, r in df.iterrows():
            bar = "█" * int(r["stability_pct"] // 10)
            print(f"  {r['transition']:<26} N={r['N']:>7,}  "
                  f"{r['reference_model']:<18}  {r['stability_pct']:>5.1f}%  {bar}")
            if r["changes"] != "—":
                print(f"    ↳ {r['changes']}")
        print(f"\n  Overall mean stability: {df['stability_pct'].mean():.1f}%")
        print("=" * 65 + "\n")

    return df