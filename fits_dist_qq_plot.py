import os
import sys
import json
import argparse
import logging
from typing import Dict, Tuple, Optional, Any, List, Callable

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import des modules locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from football_model.distribution.parametric_fits import (
    extract_transitions,
    build_duration_dict,
    filter_positive_durations,
    fit_all_distributions
)
from football_model.distribution.gof_tests import ks_p_simple, ks_bootstrap_parametric, compute_ks_test

from football_model.distribution.sampler import get_sampler_for_model


MODELS_INFO = {
    "exponential": {"dist": stats.expon, "params": ["loc", "scale"]},
    "weibull": {"dist": stats.weibull_min, "params": ["c", "loc", "scale"]},
    "lognormal": {"dist": stats.lognorm, "params": ["s", "loc", "scale"]},
    "loglogistic": {"dist": stats.fisk, "params": ["c", "loc", "scale"]},
    "gen_weibull": {"dist": stats.exponweib, "params": ["a", "c", "loc", "scale"]},
    "johnsonsu": {"dist": stats.johnsonsu, "params": ["a", "b", "loc", "scale"]},
    "invgauss": {"dist": stats.invgauss, "params": ["mu", "loc", "scale"]},
    "mielke": {"dist": stats.mielke, "params": ["k", "s", "loc", "scale"]},
    "geninvgauss": {"dist": stats.geninvgauss, "params": ["p", "b", "loc", "scale"]},
    "genlogistic": {"dist": stats.genlogistic, "params": ["c", "loc", "scale"]},
    "gamma": {"dist": stats.gamma, "params": ["a", "loc", "scale"]},
    "gengamma": {"dist": stats.gengamma, "params": ["a", "c", "loc", "scale"]},
}


# ---------------------------------------------------------------------------
# QQ score (replaces BIC as selection criterion)
# ---------------------------------------------------------------------------

def get_ppf_function(model_name: str, params: Dict[str, Any]) -> Callable:
    """Inverse CDF (percent-point function) for each model."""
    dispatch = {
        "exponential":  lambda p: stats.expon.ppf(p, loc=params["loc"], scale=params["scale"]),
        "weibull":      lambda p: stats.weibull_min.ppf(p, params["c"], loc=params["loc"], scale=params["scale"]),
        "lognormal":    lambda p: stats.lognorm.ppf(p, params["s"], loc=params["loc"], scale=params["scale"]),
        "loglogistic":  lambda p: stats.fisk.ppf(p, params["c"], loc=params["loc"], scale=params["scale"]),
        "gen_weibull":  lambda p: stats.exponweib.ppf(p, params["a"], params["c"], loc=params["loc"], scale=params["scale"]),
        "johnsonsu":    lambda p: stats.johnsonsu.ppf(p, params["a"], params["b"], loc=params["loc"], scale=params["scale"]),
        "invgauss":     lambda p: stats.invgauss.ppf(p, params["mu"], loc=params["loc"], scale=params["scale"]),
        "mielke":       lambda p: stats.mielke.ppf(p, params["k"], params["s"], loc=params["loc"], scale=params["scale"]),
        "geninvgauss":  lambda p: stats.geninvgauss.ppf(p, params["p"], params["b"], loc=params["loc"], scale=params["scale"]),
        "genlogistic":  lambda p: stats.genlogistic.ppf(p, params["c"], loc=params["loc"], scale=params["scale"]),
        "gamma":        lambda p: stats.gamma.ppf(p, params["a"], loc=params["loc"], scale=params["scale"]),
        "gengamma":     lambda p: stats.gengamma.ppf(p, params["a"], params["c"], loc=params["loc"], scale=params["scale"]),
    }
    if model_name not in dispatch:
        raise ValueError(f"Unknown model: {model_name}")
    return dispatch[model_name]


def qq_score(
    data: np.ndarray,
    model_name: str,
    params: Dict[str, Any],
    bulk_pct: float = 0.95,
    tail_penalty_factor: float = 3.0,
) -> float:
    """
    Lower is better.

    Mean squared relative deviation between empirical and theoretical
    quantiles, computed on the bulk (0 to bulk_pct-th percentile).

    A tail penalty is added when the model's 99th percentile exceeds
    tail_penalty_factor × the empirical 99th percentile. This discards
    models like mielke that fit the bulk acceptably but predict
    unrealistically long durations in the tail.

    Returns np.inf if the model cannot be evaluated.
    """
    try:
        n     = len(data)
        probs = (np.arange(1, n + 1) - 0.5) / n
        eq    = np.sort(data)

        ppf = get_ppf_function(model_name, params)
        tq  = ppf(probs)

        # ── tail sanity check ────────────────────────────────────────────
        emp_p99   = np.percentile(eq, 99)
        theo_p99  = ppf(np.array([0.99]))[0]
        if np.isfinite(theo_p99) and theo_p99 > tail_penalty_factor * emp_p99:
            # model predicts tails far beyond what the data shows — reject
            return np.inf

        # ── bulk score ───────────────────────────────────────────────────
        cutoff = np.percentile(eq, bulk_pct * 100)
        mask   = (eq <= cutoff) & np.isfinite(tq) & (tq > 0)
        if mask.sum() < 10:
            return np.inf

        rel_dev = (eq[mask] - tq[mask]) / tq[mask]
        return float(np.mean(rel_dev ** 2))

    except Exception:
        return np.inf


# ---------------------------------------------------------------------------
# Logging / args
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribution fitting — best model selected by QQ score"
    )
    parser.add_argument("--n_boot",     type=int, default=100)
    parser.add_argument("--min_obs",    type=int, default=10)
    parser.add_argument("--data_csv",   type=str, required=True)
    parser.add_argument("--data_pkl",   type=str, required=True)
    parser.add_argument("--output_json",type=str, default="results_fits_dist.json")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def get_all_parametric_models_detailed(data: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Fit ALL parametric models and return full details including QQ score."""
    fit_results = fit_all_distributions(pd.Series(data))
    detailed_results = {}

    for model_name, params in fit_results.items():
        score = qq_score(data, model_name, params)
        detailed_results[model_name] = {
            "status":     "fitted",
            "parameters": params,
            "bic":        params.get("BIC", float("inf")),
            "aic":        params.get("AIC", float("inf")),
            "ks_stat":    params.get("KS_stat", None),
            "ks_p":       params.get("KS_p", None),
            "qq_score":   score,
        }

    for model_name in MODELS_INFO:
        if model_name not in detailed_results:
            detailed_results[model_name] = {
                "status":     "failed",
                "parameters": None,
                "bic":        float("inf"),
                "aic":        float("inf"),
                "ks_stat":    None,
                "ks_p":       None,
                "qq_score":   float("inf"),
                "error":      "Fitting failed",
            }

    return detailed_results


def get_best_parametric_model(
    data: np.ndarray,
) -> Optional[Tuple[str, Dict[str, float], float, Dict[str, Dict[str, Any]]]]:
    """
    Select the best model by QQ score (lowest = best fit to data bulk).
    Returns (model_name, params, qq_score, all_models_details) or None.
    """
    all_models_details = get_all_parametric_models_detailed(data)

    successful = {
        name: det for name, det in all_models_details.items()
        if det["status"] == "fitted"
        and det.get("parameters") is not None
        and np.isfinite(det["qq_score"])
    }

    if not successful:
        return None

    # ── selection by QQ score ────────────────────────────────────────────
    best_name = min(successful, key=lambda x: successful[x]["qq_score"])
    best_det  = successful[best_name]

    return best_name, best_det["parameters"], best_det["qq_score"], all_models_details


def get_top_distributions(
    all_models_details: Dict[str, Dict[str, Any]],
    top_n: int = 4,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return top-N models sorted by QQ score (best first)."""
    successful = {
        name: det for name, det in all_models_details.items()
        if det["status"] == "fitted"
        and det.get("parameters") is not None
        and np.isfinite(det.get("qq_score", np.inf))
    }
    if not successful:
        return []

    sorted_models = sorted(successful.items(), key=lambda x: x[1]["qq_score"])
    return sorted_models[:top_n]


def display_qq_table(models_results: Dict[str, Dict[str, Any]], top_n: int = 5) -> None:
    """Display ranking table sorted by QQ score."""
    successful = {
        name: r for name, r in models_results.items()
        if r["status"] == "fitted" and np.isfinite(r.get("qq_score", np.inf))
    }
    if not successful:
        logging.warning("  No successfully fitted models.")
        return

    sorted_models = sorted(successful.items(), key=lambda x: x[1]["qq_score"])

    logging.info(f"  QQ RANKING (Top {min(top_n, len(sorted_models))}):")
    logging.info("  " + "=" * 60)
    logging.info(f"  {'Model':<15} {'QQ Score':<12} {'BIC':<12} {'KS p-val':<10}")
    logging.info("  " + "-" * 60)

    for i, (name, r) in enumerate(sorted_models[:top_n]):
        marker  = "🏆" if i == 0 else "  "
        ks_p    = r.get("ks_p")
        ks_str  = f"{ks_p:.4f}" if ks_p is not None else "N/A"
        logging.info(
            f"{marker} {name:<15} {r['qq_score']:<12.5f} "
            f"{r['bic']:<12.2f} {ks_str:<10}"
        )

    logging.info("  " + "=" * 60)


# ---------------------------------------------------------------------------
# CDF helpers (unchanged)
# ---------------------------------------------------------------------------

def get_cdf_function(model_name: str, params: Dict[str, Any]) -> Callable:
    if model_name == "exponential":
        return lambda x: stats.expon.cdf(x, loc=params["loc"], scale=params["scale"])
    elif model_name == "weibull":
        return lambda x: stats.weibull_min.cdf(x, params["c"], loc=params["loc"], scale=params["scale"])
    elif model_name == "lognormal":
        return lambda x: stats.lognorm.cdf(x, params["s"], loc=params["loc"], scale=params["scale"])
    elif model_name == "loglogistic":
        return lambda x: stats.fisk.cdf(x, params["c"], loc=params["loc"], scale=params["scale"])
    elif model_name == "gen_weibull":
        return lambda x: stats.exponweib.cdf(x, params["a"], params["c"], loc=params["loc"], scale=params["scale"])
    elif model_name == "johnsonsu":
        return lambda x: stats.johnsonsu.cdf(x, params["a"], params["b"], loc=params["loc"], scale=params["scale"])
    elif model_name == "invgauss":
        return lambda x: stats.invgauss.cdf(x, params["mu"], loc=params["loc"], scale=params["scale"])
    elif model_name == "mielke":
        return lambda x: stats.mielke.cdf(x, params["k"], params["s"], loc=params["loc"], scale=params["scale"])
    elif model_name == "geninvgauss":
        return lambda x: stats.geninvgauss.cdf(x, params["p"], params["b"], loc=params["loc"], scale=params["scale"])
    elif model_name == "genlogistic":
        return lambda x: stats.genlogistic.cdf(x, params["c"], loc=params["loc"], scale=params["scale"])
    elif model_name == "gamma":
        return lambda x: stats.gamma.cdf(x, params["a"], loc=params["loc"], scale=params["scale"])
    elif model_name == "gengamma":
        return lambda x: stats.gengamma.cdf(x, params["a"], params["c"], loc=params["loc"], scale=params["scale"])
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_qq(
    transition_key: Tuple[str, str],
    data: np.ndarray,
    all_models_details: Dict[str, Dict[str, Any]],
    top_n: int = 4,
    output_dir: str = "plots_distributions",
) -> None:
    """QQ plot for the top-N models (ranked by QQ score). Best model gets gold border."""
    Path(output_dir).mkdir(exist_ok=True)
    data = np.asarray(data, dtype=float)
    n    = len(data)

    top_models = get_top_distributions(all_models_details, top_n=top_n)
    if not top_models:
        return

    # subsample — 5000 evenly-spaced quantiles is plenty for a visual QQ plot
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
    palette = sns.color_palette("tab10", len(top_models))
    from_state, to_state = transition_key

    # global axis limits from empirical data — same scale on all subplots
    global_lo = np.percentile(empirical_q, 1)  * 0.95
    global_hi = np.percentile(empirical_q, 99) * 1.05

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
        eq   = empirical_q[mask]
        tq   = tq[mask]

        ax.scatter(tq, eq, color=color, s=18, alpha=0.65, zorder=3, label="Observed")
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

        # gold border for the best model (rank 0)
        if ax_idx == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor("goldenrod")
                spine.set_linewidth(3)

    for idx in range(len(top_models), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        f"QQ plots — {from_state} → {to_state}  (n={n})\n"
        f"Gold border = QQ-selected best model",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    filepath = Path(output_dir) / f"qq_{from_state}_{to_state}.png"
    plt.savefig(filepath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logging.info(f"  QQ plot saved: {filepath}")


def plot_model_fit(transition_key, data, fit_res, output_dir="plots"):
    """Survival curve — models ordered by QQ score (best first in legend)."""
    from scipy.stats import cumfreq
    Path(output_dir).mkdir(exist_ok=True)
    data = np.asarray(data)
    from_state, to_state = transition_key

    res      = cumfreq(data, numbins=len(data))
    x_vals   = res.lowerlimit + res.binsize * (np.arange(res.cumcount.size) + 0.5)
    surv_emp = 1 - res.cumcount / res.cumcount[-1]

    plt.figure(figsize=(12, 8))
    plt.step(x_vals, surv_emp, where="mid", label="Empirical", color="black", linewidth=2)

    x       = np.linspace(max(0, data.min()), data.max(), 300)
    palette = sns.color_palette("tab10", len(fit_res))

    for (model_name, params), color in zip(fit_res.items(), palette):
        cdf        = get_cdf_function(model_name, params)
        surv_model = 1 - cdf(x)
        plt.plot(x, surv_model, linestyle="--", linewidth=2.5,
                 label=model_name, alpha=0.8, color=color)

    plt.xlabel("Duration", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.title(f"Survival: {from_state} → {to_state}", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    filepath = Path(output_dir) / f"survival_{from_state}_{to_state}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logging.info(f"  Survival plot saved: {filepath}")


def plot_specific_transition_zoom(transition_key, data, fit_res, x_min=0, x_max=5, output_dir="plots"):
    from scipy.stats import cumfreq
    Path(output_dir).mkdir(exist_ok=True)
    data = np.asarray(data)
    from_state, to_state = transition_key

    res      = cumfreq(data, numbins=len(data))
    x_vals   = res.lowerlimit + res.binsize * (np.arange(res.cumcount.size) + 0.5)
    surv_emp = 1 - res.cumcount / res.cumcount[-1]

    plt.figure(figsize=(12, 8))
    plt.step(x_vals, surv_emp, where="mid", label="Empirical", color="black", linewidth=3)

    x       = np.linspace(x_min, x_max, 300)
    palette = sns.color_palette("Dark2", len(fit_res))

    for (model_name, params), color in zip(fit_res.items(), palette):
        cdf        = get_cdf_function(model_name, params)
        surv_model = 1 - cdf(x)
        plt.plot(x, surv_model, linestyle="dotted", linewidth=2.5,
                 label=model_name, alpha=0.8, color=color)

    plt.xlim([x_min, x_max])
    plt.ylim([0, 1.05])
    plt.xlabel("Duration (zoom)", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.title(f"Zoom Survival: {from_state} → {to_state}", fontsize=16)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filepath = Path(output_dir) / f"zoom_{from_state}_{to_state}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logging.info(f"  Zoom plot saved: {filepath}")


def plot_spline_with_knots(transition_key, data, spline_result, output_dir="plots"):
    from scipy.stats import cumfreq
    Path(output_dir).mkdir(exist_ok=True)
    sf = spline_result.get("sf")
    if sf is None:
        return

    plt.figure(figsize=(14, 10))
    res      = cumfreq(data, numbins=len(data))
    x_vals   = res.lowerlimit + res.binsize * (np.arange(res.cumcount.size) + 0.5)
    surv_emp = 1 - res.cumcount / res.cumcount[-1]
    t_range  = np.linspace(0, min(max(data) * 1.1, 20), 500)

    plt.step(x_vals, surv_emp, where="mid", label="Empirical", color="black", linewidth=2)

    try:
        surv_spline = sf.survival_function_at_times(t_range).flatten()
        plt.plot(t_range, surv_spline, color="#9B59B6", linewidth=4,
                 label="Spline", alpha=0.9, zorder=5)
    except Exception:
        pass

    from_state, to_state = transition_key
    plt.title(f"Spline: {from_state} → {to_state}", fontsize=16, fontweight="bold")
    plt.xlabel("Duration (s)", fontsize=14)
    plt.ylabel("Survival S(t)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, max(t_range))
    plt.ylim(0, 1.05)
    plt.tight_layout()

    filepath = Path(output_dir) / f"spline_{from_state}_{to_state}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logging.info(f"  Spline plot saved: {filepath}")


# ---------------------------------------------------------------------------
# Plot orchestration
# ---------------------------------------------------------------------------

def create_plots_for_transition(transition_key, data, analysis_result, output_dir="plots_distributions"):
    Path(output_dir).mkdir(exist_ok=True)

    if "best_model" not in analysis_result or "params" not in analysis_result:
        logging.warning(f"Missing data for plots of {transition_key}")
        return

    all_details = analysis_result.get("all_models_details", {})

    # top models sorted by QQ score for survival plots
    top_models = {}
    for model_name, details in get_top_distributions(all_details, top_n=3):
        top_models[model_name] = details["parameters"]

    if not top_models:
        top_models[analysis_result["best_model"]] = analysis_result["params"]

    plot_model_fit(transition_key, data, top_models, output_dir=output_dir)
    plot_specific_transition_zoom(transition_key, data, top_models, output_dir=output_dir)

    if all_details:
        plot_qq(transition_key, data, all_details, top_n=4, output_dir=output_dir)

    if (analysis_result.get("status") == "validated_spline"
            and "spline_model" in analysis_result):
        try:
            plot_spline_with_knots(transition_key, data,
                                   analysis_result["spline_model"],
                                   output_dir=output_dir)
        except Exception as e:
            logging.error(f"Spline plot error: {e}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def analyze_transition_workflow(
    transition_key, data, n_boot=100,
    fallback_to_spline=True, create_plots=True,
    output_dir="plots_distributions",
):
    n      = len(data)
    result = {"transition": list(transition_key), "n_obs": n}

    logging.info(f"  Observations: {n}")

    best = get_best_parametric_model(data)
    if best is None:
        logging.warning("  All models failed.")
        return {"transition": list(transition_key), "n_obs": n,
                "status": "fitting_failed", "best_model": None}

    model_name, params, score, all_models_details = best
    bic = all_models_details[model_name]["bic"]
    aic = all_models_details[model_name]["aic"]

    logging.info(f"  Best model (QQ): {model_name}  score={score:.5f}  BIC={bic:.2f}")
    display_qq_table(all_models_details)

    result.update({
        "best_model":        model_name,
        "params":            params,
        "qq_score":          score,
        "bic":               bic,
        "aic":               aic,
        "all_models_details": all_models_details,
    })

    # KS validation
    cdf_func = get_cdf_function(model_name, params)
    p_simple = ks_p_simple(data, cdf_func)
    result["ks_p_simple"] = p_simple

    if p_simple[1] > 0.05:
        result["status"]            = "validated_simple_ks"
        result["validation_method"] = "simple_ks"
    else:
        sampler = get_sampler_for_model(model_name, params)
        if sampler is None :
           print(" Error The sampler is not in the model params")
        else :
            try:
                ks_stat, p_boot = ks_bootstrap_parametric(data, cdf_func, sampler, n_boot=n_boot)
                result["ks_stat_bootstrap"] = ks_stat
                result["ks_p_bootstrap"]    = p_boot
                result["status"]            = "validated_bootstrap_ks" if p_boot > 0.05 else "failed"
                result["validation_method"] = "bootstrap_ks"           if p_boot > 0.05 else "none"
            except Exception as e:
                logging.error(f"  Bootstrap error: {e}")
                result["status"]            = "failed"
                result["validation_method"] = "none"

    if create_plots:
        create_plots_for_transition(transition_key, data, result, output_dir=output_dir)

    return result


def process_all_transitions(transitions_dict, min_obs=10, n_boot=100,
                             create_plots=True, output_dir="plots"):
    results         = {}
    valid_transitions = {k: v for k, v in transitions_dict.items() if len(v) >= min_obs}
    logging.info(f"Processing {len(valid_transitions)} transitions with ≥{min_obs} obs")

    if create_plots:
        Path(output_dir).mkdir(exist_ok=True)

    for transition_key, data in tqdm(valid_transitions.items(), desc="Transitions"):
        result                       = analyze_transition_workflow(
            transition_key, data, n_boot=n_boot,
            fallback_to_spline=True, create_plots=create_plots, output_dir=output_dir,
        )
        results[str(transition_key)] = result

    return results


# ---------------------------------------------------------------------------
# Reporting & serialisation (unchanged from original)
# ---------------------------------------------------------------------------

def generate_summary_report(results):
    status_counts = {}
    method_counts = {}
    model_counts  = {}
    bic_values    = []

    for r in results.values():
        status_counts[r.get("status",            "unknown")] = status_counts.get(r.get("status",            "unknown"), 0) + 1
        method_counts[r.get("validation_method", "none")]    = method_counts.get(r.get("validation_method", "none"),    0) + 1
        if r.get("best_model"):
            model_counts[r["best_model"]] = model_counts.get(r["best_model"], 0) + 1
        if r.get("bic") and np.isfinite(r["bic"]):
            bic_values.append(r["bic"])

    total = len(results)
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY REPORT")
    logging.info("=" * 80)
    logging.info(f"Total transitions: {total}")
    for status, count in sorted(status_counts.items()):
        logging.info(f"  {status}: {count} ({100*count/total:.1f}%)")
    logging.info("Most frequent models (QQ-selected):")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        logging.info(f"  {model}: {count} ({100*count/total:.1f}%)")


def create_detailed_tables(results):
    tables = {"summary_by_transition": {}, "validation_methods_summary": {}, "bic_rankings": {}}

    status_counts = {}
    method_counts = {}
    model_counts  = {}
    bic_data      = []

    for key_str, r in results.items():
        tables["summary_by_transition"][key_str] = {
            "n_observations":     r.get("n_obs", 0),
            "validation_status":  r.get("status", "unknown"),
            "validation_method":  r.get("validation_method", "none"),
            "best_model_qq":      r.get("best_model"),
            "qq_score":           r.get("qq_score"),
            "bic":                r.get("bic"),
            "all_models_details": r.get("all_models_details", {}),
        }
        status_counts[r.get("status",            "unknown")] = status_counts.get(r.get("status",            "unknown"), 0) + 1
        method_counts[r.get("validation_method", "none")]    = method_counts.get(r.get("validation_method", "none"),    0) + 1
        model_counts[r.get("best_model",         "none")]    = model_counts.get(r.get("best_model",         "none"),    0) + 1

        if r.get("bic") is not None:
            bic_data.append({"transition": key_str, "model": r.get("best_model"),
                             "bic": r["bic"], "qq_score": r.get("qq_score"), "n_obs": r.get("n_obs", 0)})

    bic_data.sort(key=lambda x: x.get("qq_score") or np.inf)
    tables["validation_methods_summary"] = {
        "status_distribution": status_counts,
        "method_distribution": method_counts,
        "model_distribution":  model_counts,
        "total_transitions":   len(results),
    }
    tables["qq_rankings"] = {"best_fits": bic_data[:10], "total_ranked": len(bic_data)}
    return tables


def serialize_for_json(obj):
    if isinstance(obj, np.ndarray):           return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)):   return int(obj)
    if isinstance(obj, (np.floating, np.float64)):return float(obj)
    if isinstance(obj, np.bool_):             return bool(obj)
    if isinstance(obj, tuple):                return list(obj)
    raise TypeError(f"Non-serialisable type: {type(obj)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging()
    logging.info("Starting distribution fitting with QQ-based model selection")

    df_final     = pd.read_csv(args.data_csv)
    with open(args.data_pkl, "rb") as f:
        all_sequences = pd.read_pickle(f)

    transitions_list = extract_transitions(all_sequences)
    duration_dict    = build_duration_dict(transitions_list)
    duration_dict    = filter_positive_durations(duration_dict)
    logging.info(f"{len(duration_dict)} unique transition types found")

    results = process_all_transitions(
        duration_dict,
        min_obs=args.min_obs,
        n_boot=args.n_boot,
        create_plots=True,
        output_dir="plots_distributions",
    )

    generate_summary_report(results)
    detailed_tables = create_detailed_tables(results)

    final_results = {
        "metadata": {
            "script_version":    "fits_dist_qq_selection_v1",
            "selection_criterion": "qq_score",
            "n_boot":            args.n_boot,
            "min_obs":           args.min_obs,
            "total_transitions": len(results),
            "data_files":        {"csv": args.data_csv, "pkl": args.data_pkl},
        },
        "results_by_transition": results,
        "detailed_tables":       detailed_tables,
    }

    logging.info(f"Saving results to {args.output_json}")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2, default=serialize_for_json)

    file_size = os.path.getsize(args.output_json) / (1024 * 1024)
    logging.info(f"Saved ({file_size:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())