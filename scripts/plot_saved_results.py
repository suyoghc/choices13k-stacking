#!/usr/bin/env python
"""Generate figures from saved Bayesian results (NetCDF files).

Two-level figure structure matching the paper narrative:
  Level 1 (4-model, classical only): CPT > PT >> EU ~ EV
  Level 2 (5-model, with context): Context >> all classical combined

Usage:
    # Copy results from Della (4-model already renamed locally):
    scp della:.../results/bayesian_stacking.nc results/bayesian_stacking_5model.nc
    scp della:.../results/hierarchical_bayesian.nc results/hierarchical_bayesian_5model.nc
    scp della:.../results/mot.nc results/mot_5model.nc
    scp della:.../results/summary.pkl results/summary_5model.pkl

    python scripts/plot_saved_results.py
"""

import pickle
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})

COLORS_4 = {"EV": "#1f77b4", "EU": "#ff7f0e", "PT": "#2ca02c", "CPT": "#d62728"}
COLORS_5 = {**COLORS_4, "Context": "#9467bd"}
CLASSICAL_NAMES = ["EV", "EU", "PT", "CPT"]
ALL_NAMES = ["EV", "EU", "PT", "CPT", "Context"]

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGDIR = RESULTS_DIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load both 4-model and 5-model results."""
    print("Loading saved results...")
    results = {}

    # 4-model (classical only)
    with open(RESULTS_DIR / "summary_4model.pkl", "rb") as f:
        results["summary_4"] = pickle.load(f)
    results["bayesian_4"] = az.from_netcdf(RESULTS_DIR / "bayesian_stacking_4model.nc")
    results["hierarchical_4"] = az.from_netcdf(RESULTS_DIR / "hierarchical_bayesian_4model.nc")
    results["mot_4"] = az.from_netcdf(RESULTS_DIR / "mot_4model.nc")

    # 5-model (with context)
    with open(RESULTS_DIR / "summary_5model.pkl", "rb") as f:
        results["summary_5"] = pickle.load(f)
    results["bayesian_5"] = az.from_netcdf(RESULTS_DIR / "bayesian_stacking_5model.nc")
    results["hierarchical_5"] = az.from_netcdf(RESULTS_DIR / "hierarchical_bayesian_5model.nc")
    results["mot_5"] = az.from_netcdf(RESULTS_DIR / "mot_5model.nc")

    return results


# =========================================================================
# Level 1: Classical theories
# =========================================================================

def fig1_classical_weights(results):
    """Fig 1: Classical theory Bayesian weights with HDI."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    summary = results["summary_4"]

    # Panel A: Frequentist weights
    ax = axes[0]
    weights = summary["freq_weights"]
    colors = [COLORS_4[m] for m in CLASSICAL_NAMES]
    bars = ax.bar(CLASSICAL_NAMES, weights * 100, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Stacking Weight (%)")
    ax.set_title("A. Frequentist Stacking Weights")
    ax.set_ylim(0, 75)
    for bar, w in zip(bars, weights):
        if w > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{w * 100:.1f}%", ha="center", fontsize=10)

    # Panel B: Bayesian weights with HDI
    ax = axes[1]
    means = summary["bayes_means"]
    lows = summary["bayes_hdi_low"]
    highs = summary["bayes_hdi_high"]

    x = np.arange(len(CLASSICAL_NAMES))
    bars = ax.bar(x, means * 100, color=colors, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means * 100,
                yerr=[means * 100 - lows * 100, highs * 100 - means * 100],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSICAL_NAMES)
    ax.set_ylabel("Bayesian Weight (%)")
    ax.set_title("B. Bayesian Posterior Mean (94% HDI)")
    ax.set_ylim(0, 75)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_classical_weights.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig1_classical_weights.png", bbox_inches="tight", dpi=300)
    print("  Saved: fig1_classical_weights")
    plt.close()


def fig2_classical_posteriors(results):
    """Fig 2: Full posterior densities for classical theories."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bayesian stacking posteriors
    ax = axes[0]
    weights = results["bayesian_4"].posterior["weights"].values
    weights_flat = weights.reshape(-1, weights.shape[-1])

    for i, model in enumerate(CLASSICAL_NAMES):
        sns.kdeplot(weights_flat[:, i] * 100, ax=ax, color=COLORS_4[model],
                    label=model, linewidth=2, fill=True, alpha=0.3)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Weight (%)")
    ax.set_ylabel("Density")
    ax.set_title("A. Bayesian Stacking Posteriors")
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 80)

    # Panel B: MOT posteriors
    ax = axes[1]
    pi = results["mot_4"].posterior["pi"].values
    pi_flat = pi.reshape(-1, pi.shape[-1])

    for i, model in enumerate(CLASSICAL_NAMES):
        sns.kdeplot(pi_flat[:, i] * 100, ax=ax, color=COLORS_4[model],
                    label=model, linewidth=2, fill=True, alpha=0.3)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Theory Usage (%)")
    ax.set_ylabel("Density")
    ax.set_title("B. Mixture of Theories Posteriors")
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 80)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig2_classical_posteriors.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig2_classical_posteriors.png", bbox_inches="tight", dpi=300)
    print("  Saved: fig2_classical_posteriors")
    plt.close()


def fig3_hierarchical(results):
    """Fig 3: Hierarchical feature effects (classical only)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    beta = results["hierarchical_4"].posterior["beta"].values
    beta_flat = beta.reshape(-1, beta.shape[-2], beta.shape[-1])

    feature_names = ["LotNumB", "Amb", "Corr"]
    feature_labels = ["# Lottery Outcomes", "Ambiguity", "Correlation"]
    ref_models = CLASSICAL_NAMES[:-1]  # exclude CPT (reference)

    y_pos = 0
    y_ticks = []
    y_labels = []

    for j, (fname, flabel) in enumerate(zip(feature_names, feature_labels)):
        for k, mname in enumerate(ref_models):
            samples = beta_flat[:, j, k]
            mean = samples.mean()
            lo, hi = np.percentile(samples, [3, 97])

            color = COLORS_4[mname]
            ax.plot([lo, hi], [y_pos, y_pos], color=color, linewidth=3, alpha=0.8)
            ax.scatter([mean], [y_pos], color=color, s=80, zorder=5, edgecolor="black")

            # Star if credibly non-zero
            if lo > 0 or hi < 0:
                ax.scatter([mean], [y_pos], color=color, s=150, marker="*", zorder=6)

            y_ticks.append(y_pos)
            y_labels.append(f"{flabel} → {mname}")
            y_pos += 1
        y_pos += 0.5

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Effect on Log-Odds (vs CPT reference)")
    ax.set_title("Feature Effects on Model Weights\n(* = 94% HDI excludes 0)")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_hierarchical.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig3_hierarchical.png", bbox_inches="tight", dpi=300)
    print("  Saved: fig3_hierarchical")
    plt.close()


# =========================================================================
# Level 2: Context vs classical
# =========================================================================

def fig4_context_dominance(results):
    """Fig 4: Context model dominance — 5-model Bayesian weights.

    Two panels: full scale (showing Context) + zoomed into classical theories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    summary = results["summary_5"]
    model_names = list(summary["model_names"])

    means = summary["bayes_means"]
    lows = summary["bayes_hdi_low"]
    highs = summary["bayes_hdi_high"]

    colors = [COLORS_5[m] for m in model_names]

    # Panel A: Full scale
    ax = axes[0]
    x = np.arange(len(model_names))
    bars = ax.bar(x, means * 100, color=colors, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means * 100,
                yerr=[means * 100 - lows * 100, highs * 100 - means * 100],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Bayesian Weight (%)")
    ax.set_title("A. 5-Model Bayesian Stacking")
    ax.set_ylim(0, 105)
    # Annotate Context weight
    ctx_idx = model_names.index("Context")
    ax.text(ctx_idx, means[ctx_idx] * 100 + 2,
            f"{means[ctx_idx] * 100:.1f}%\n[{lows[ctx_idx] * 100:.1f}, {highs[ctx_idx] * 100:.1f}]",
            ha="center", fontsize=9, fontweight="bold")

    # Panel B: Zoom into classical theories (0-5% range)
    ax = axes[1]
    classical_idx = [model_names.index(m) for m in CLASSICAL_NAMES]
    x_c = np.arange(len(CLASSICAL_NAMES))
    classical_means = means[classical_idx]
    classical_lows = lows[classical_idx]
    classical_highs = highs[classical_idx]
    classical_colors = [COLORS_4[m] for m in CLASSICAL_NAMES]

    bars = ax.bar(x_c, classical_means * 100, color=classical_colors, edgecolor="black", linewidth=0.8)
    ax.errorbar(x_c, classical_means * 100,
                yerr=[classical_means * 100 - classical_lows * 100,
                      classical_highs * 100 - classical_means * 100],
                fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x_c)
    ax.set_xticklabels(CLASSICAL_NAMES)
    ax.set_ylabel("Bayesian Weight (%)")
    ax.set_title("B. Classical Theories (Zoomed)")
    # Auto-scale y to show the small weights clearly
    max_hi = max(classical_highs) * 100
    ax.set_ylim(0, max(max_hi * 1.3, 5))

    for bar, m in zip(bars, classical_means):
        if m > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{m * 100:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig4_context_dominance.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig4_context_dominance.png", bbox_inches="tight", dpi=300)
    print("  Saved: fig4_context_dominance")
    plt.close()


def fig5_mot_comparison(results):
    """Fig 5: MOT comparison — 4-model vs 5-model.

    Shows how theory usage shifts when Context enters the race,
    plus kappa change (21.9 -> 143.5).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: 4-model MOT
    ax = axes[0]
    pi_4 = results["mot_4"].posterior["pi"].values
    pi_4_flat = pi_4.reshape(-1, pi_4.shape[-1])
    means_4 = pi_4_flat.mean(axis=0)
    colors_4 = [COLORS_4[m] for m in CLASSICAL_NAMES]

    bars = ax.bar(CLASSICAL_NAMES, means_4 * 100, color=colors_4, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Theory Usage (%)")
    ax.set_title("A. MOT: Classical Only")
    ax.set_ylim(0, 105)
    for bar, m in zip(bars, means_4):
        if m > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{m * 100:.1f}%", ha="center", fontsize=9)

    # Panel B: 5-model MOT
    ax = axes[1]
    pi_5 = results["mot_5"].posterior["pi"].values
    pi_5_flat = pi_5.reshape(-1, pi_5.shape[-1])
    model_names_5 = list(results["summary_5"]["model_names"])
    means_5 = pi_5_flat.mean(axis=0)
    colors_5 = [COLORS_5[m] for m in model_names_5]

    bars = ax.bar(model_names_5, means_5 * 100, color=colors_5, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Theory Usage (%)")
    ax.set_title("B. MOT: With Context Model")
    ax.set_ylim(0, 105)
    for bar, m, name in zip(bars, means_5, model_names_5):
        if m > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{m * 100:.1f}%", ha="center", fontsize=9)

    # Panel C: Kappa comparison
    ax = axes[2]
    kappa_4 = results["mot_4"].posterior["kappa"].values.flatten()
    kappa_5 = results["mot_5"].posterior["kappa"].values.flatten()

    ax.hist(kappa_4, bins=50, density=True, alpha=0.6, color="#d62728",
            edgecolor="black", linewidth=0.5, label=f"4-model (mean={kappa_4.mean():.1f})")
    ax.hist(kappa_5, bins=50, density=True, alpha=0.6, color="#9467bd",
            edgecolor="black", linewidth=0.5, label=f"5-model (mean={kappa_5.mean():.1f})")
    ax.set_xlabel(r"$\kappa$ (concentration)")
    ax.set_ylabel("Density")
    ax.set_title(r"C. Overdispersion $\kappa$")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig5_mot_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig5_mot_comparison.png", bbox_inches="tight", dpi=300)
    print("  Saved: fig5_mot_comparison")
    plt.close()


def fig6_diagnostics(results):
    """Fig 6: MCMC diagnostics (5-model)."""
    # Bayesian stacking traces
    axes_bayes = az.plot_trace(results["bayesian_5"], var_names=["weights"], compact=True)
    fig_bayes = axes_bayes.flatten()[0].get_figure()
    fig_bayes.suptitle("5-Model Bayesian Stacking: MCMC Diagnostics", fontsize=14)
    fig_bayes.tight_layout()
    fig_bayes.savefig(FIGDIR / "fig6a_bayes_diagnostics.pdf", bbox_inches="tight", dpi=300)
    fig_bayes.savefig(FIGDIR / "fig6a_bayes_diagnostics.png", bbox_inches="tight", dpi=300)
    plt.close(fig_bayes)

    # MOT traces
    axes_mot = az.plot_trace(results["mot_5"], var_names=["pi"], compact=True)
    fig_mot = axes_mot.flatten()[0].get_figure()
    fig_mot.suptitle("5-Model MOT: MCMC Diagnostics", fontsize=14)
    fig_mot.tight_layout()
    fig_mot.savefig(FIGDIR / "fig6b_mot_diagnostics.pdf", bbox_inches="tight", dpi=300)
    fig_mot.savefig(FIGDIR / "fig6b_mot_diagnostics.png", bbox_inches="tight", dpi=300)
    plt.close(fig_mot)

    print("  Saved: fig6a_bayes_diagnostics, fig6b_mot_diagnostics")


def summary_tables(results):
    """Print summary tables for both levels."""
    # Level 1
    s4 = results["summary_4"]
    print("\n" + "=" * 70)
    print("LEVEL 1: CLASSICAL THEORIES ONLY")
    print("=" * 70)
    print(f"\n{'Model':<8} {'Freq.':>10} {'Bayes Mean':>12} {'Bayes 94% HDI':>18} {'MOT':>10}")
    print("-" * 70)
    for i, m in enumerate(CLASSICAL_NAMES):
        freq = s4["freq_weights"][i] * 100
        bayes = s4["bayes_means"][i] * 100
        lo = s4["bayes_hdi_low"][i] * 100
        hi = s4["bayes_hdi_high"][i] * 100
        mot = s4["mot_pi"][i] * 100
        print(f"{m:<8} {freq:>9.1f}% {bayes:>11.1f}% [{lo:>5.1f}, {hi:>5.1f}] {mot:>9.1f}%")
    print(f"\nMOT Kappa: {s4['mot_kappa']:.1f}")

    # Level 2
    s5 = results["summary_5"]
    model_names = list(s5["model_names"])
    print("\n" + "=" * 70)
    print("LEVEL 2: WITH CONTEXT MODEL")
    print("=" * 70)
    print(f"\n{'Model':<10} {'Freq.':>10} {'Bayes Mean':>12} {'Bayes 94% HDI':>18} {'MOT':>10}")
    print("-" * 70)
    for i, m in enumerate(model_names):
        freq = s5["freq_weights"][i] * 100
        bayes = s5["bayes_means"][i] * 100
        lo = s5["bayes_hdi_low"][i] * 100
        hi = s5["bayes_hdi_high"][i] * 100
        mot = s5["mot_pi"][i] * 100
        print(f"{m:<10} {freq:>9.1f}% {bayes:>11.1f}% [{lo:>5.1f}, {hi:>5.1f}] {mot:>9.1f}%")
    print(f"\nMOT Kappa: {s5['mot_kappa']:.1f}")
    print(f"Total runtime: {s5['elapsed_minutes']:.1f} minutes")


def main():
    # Check all required files exist
    required_4 = [
        "summary_4model.pkl", "bayesian_stacking_4model.nc",
        "hierarchical_bayesian_4model.nc", "mot_4model.nc",
    ]
    required_5 = [
        "summary_5model.pkl", "bayesian_stacking_5model.nc",
        "hierarchical_bayesian_5model.nc", "mot_5model.nc",
    ]

    missing = [f for f in required_4 + required_5 if not (RESULTS_DIR / f).exists()]
    if missing:
        print(f"Missing files in {RESULTS_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print("\nFor 5-model files, copy from Della:")
        print("  scp della:.../results/bayesian_stacking.nc results/bayesian_stacking_5model.nc")
        print("  scp della:.../results/hierarchical_bayesian.nc results/hierarchical_bayesian_5model.nc")
        print("  scp della:.../results/mot.nc results/mot_5model.nc")
        print("  scp della:.../results/summary.pkl results/summary_5model.pkl")
        sys.exit(1)

    results = load_results()

    print("\nGenerating figures...")
    print("\n--- Level 1: Classical Theories ---")
    fig1_classical_weights(results)
    fig2_classical_posteriors(results)
    fig3_hierarchical(results)

    print("\n--- Level 2: Context vs Classical ---")
    fig4_context_dominance(results)
    fig5_mot_comparison(results)
    fig6_diagnostics(results)

    summary_tables(results)

    print(f"\nAll figures saved to: {FIGDIR}")


if __name__ == "__main__":
    main()
