"""
example.py
----------
Smoke-test for build_dataset() and the first six Foot_semi_xt methods.

Run from the project root:
    python example.py
"""

from build_dataset import build_dataset, list_competitions
from Class_Foot_xt import Foot_semi_xt
import ast
import joblib
import os
import random
import time
import pandas as pd

# ── 0. Browse available competitions ───────────────────────────────────────
print("Available StatsBomb competitions (first 5 rows):")
print(list_competitions()[["competition_id", "competition_name",
                            "season_id", "season_name"]].head())

# ── 1. Build dataset ────────────────────────────────────────────────────────
print("Building dataset...")
sequences, df, features_df = build_dataset(
    #competitions={55: [43]},   # EURO 2020 — small, fast
    competitions={55: [282, 43], 7: [235, 108, 27]},
    verbose=True,
)
print(f"\nsequences : {len(sequences)} possession sequences")
print(f"df        : {df.shape}")
print(f"features  : {features_df.shape}\n")

# ── 2-7. Initialise + fit — load from disk if already done ──────────────────
PRETRAINED_PATH = "foot_semi_xt_pretrained.joblib"

if os.path.exists(PRETRAINED_PATH):
    # ── LOAD pre-trained model (skips steps 2-7) ─────────────────────────────
    print(f"Loading pre-trained model from {PRETRAINED_PATH}...")
    model = joblib.load(PRETRAINED_PATH)
    print("Loaded. Skipping distribution fitting and AFT training.\n")

else:
    # ── TRAIN from scratch ───────────────────────────────────────────────────
    print("Initialising Foot_semi_xt...")
    model = Foot_semi_xt(sequences, df, features_df)
    print(f"Available transitions : {len(model.transitions)}")
    print(f"First 5               : {model.transitions[:5]}\n")

    # ── 3. Fit sojourn distributions ─────────────────────────────────────────
    print("Fitting sojourn distributions...")
    model.evaluate_sejourn_distributions(min_obs=10, n_boot=50, verbose=True)
    print(f"Fitted transitions: {len(model.fitted_transitions)}\n")

    first_key            = model.fitted_transitions[0]
    from_state, to_state = ast.literal_eval(first_key)
    print(f"Using transition '{from_state}' → '{to_state}' for plots\n")

    # ── Table 2 — mean + median + IQR ────────────────────────────────────────
    print("Computing transition time statistics (Table 2)...")
    times_table = model.transition_times_table()
    times_table.to_csv("table2_transition_times.csv", index=False)
    print("Saved → table2_transition_times.csv\n")

    # ── QQ sensitivity analysis ───────────────────────────────────────────────
    # from QQ_sens import qq_sensitivity_analysis_with_data
    # sensitivity_df = qq_sensitivity_analysis_with_data(
    #     distribution_results = model._distribution_results,
    #     duration_dict        = model._duration_dict,
    #     verbose = True,
    # )
    # sensitivity_df.to_csv("qq_sensitivity.csv", index=False)

    # ── 4. Distribution table ─────────────────────────────────────────────────
    table = model.Sejourn_distribution_table(from_state, to_state)

    # ── 5-6. Plots (uncomment to show) ───────────────────────────────────────
    # model.QQ_plot(from_state, to_state, top_n=4)
    # model.Survival_plot(from_state, to_state, top_n=3)

    # ── 7. Train AFT models ───────────────────────────────────────────────────
    print("Training AFT models...")
    model.train_aft(quiet=True)

    # ── Save so next run skips all of the above ───────────────────────────────
    print(f"Saving pre-trained model to {PRETRAINED_PATH}...")
    joblib.dump(model, PRETRAINED_PATH)
    print(f"Saved → {PRETRAINED_PATH}\n")

# ── 9. Train/test split ─────────────────────────────────────────────────────
print("\nSplitting sequences into train / test (80 / 20)...")
random.seed(42)
shuffled = sequences[:]
random.shuffle(shuffled)
split      = int(0.8 * len(shuffled))
train_seqs = shuffled[:split]
test_seqs  = shuffled[split:]
print(f"  Train: {len(train_seqs)} sequences  |  Test: {len(test_seqs)} sequences")

# ── 15. Full evaluation grid ─────────────────────────────────────────────────
# SPEEDUP: build surface once at H=300 per delta_t, then slice for H < 300.
# predict() reads surface[:, ceil(H/dt)] — same value regardless of build horizon.
# Result: 3 surface builds instead of 18 (one per delta_t, not per horizon).
print("----------------------------------------------------")
all_results = []

DELTA_TS = [0.1,0.5,1]
HORIZONS  = [10, 20, 30, 60,12,300]

for d_t in DELTA_TS:

    # ── Build once at H=300 ──────────────────────────────────────────────────


    # ── Evaluate at each horizon — just override the stored horizon value ─────
    for H in HORIZONS:
        t0 = time.time()
        model.train_XT_baseline(
            train_sequences=train_seqs,
            test_sequences=test_seqs,
            delta_t=d_t,
            horizon=H,
        )
        print(f"Train baseline  delta_t={d_t}  H=300  →  {time.time() - t0:.1f}s")

        t0 = time.time()
        model.train_XT_semi(
            train_sequences=train_seqs,
            test_sequences=test_seqs,
            delta_t=d_t,
            horizon=H,
        )
        print(f"Train semi      delta_t={d_t}  H=300  →  {time.time() - t0:.1f}s")


        # Tell the model which horizon to use for ground-truth labelling
        # and for surface slicing in predict(). Surface itself is unchanged.
        model._horizon                     = H
        model._baseline_model._horizon     = H
        model._semi_model._surface_horizon = H

        t0 = time.time()
        print(f"\nEvaluation  delta_t={d_t}  H={H}")
        suma = model.evaluate_xt()
        print(f"Evaluate  →  {time.time()-t0:.1f}s")

        suma["delta_t"] = d_t
        suma["horizon"] = H
        all_results.append(suma)
        print(suma)

        t0 = time.time()
        print("Computing bootstrap CIs...")
        ci_table = model.bootstrap_ci(n_boot=1000)
        fname = f"table4_bootstrap_ci_{d_t}_{H}.csv"
        ci_table.to_csv(fname, index=False)
        print(f"Saved → {fname}  ({time.time()-t0:.1f}s)")

    # Restore to 300 before next delta_t
    model._horizon                     = H
    model._baseline_model._horizon     = H
    model._semi_model._surface_horizon = H

# ── Save full results table ──────────────────────────────────────────────────
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv("table4_all_results.csv", index=False)
print("Saved → table4_all_results.csv")

print("\nSmoke-test complete.")