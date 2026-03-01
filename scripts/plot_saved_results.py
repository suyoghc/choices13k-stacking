#!/usr/bin/env python
"""Generate figures from saved Bayesian results (NetCDF files).

Unlike make_figures.py, this doesn't re-run any analysis.
Just loads the saved posteriors and creates visualizations.

Usage:
    # First copy results from Della:
    scp della:/scratch/gpfs/SUYOGHC/choices13k-stacking/results/*.nc results/
    scp della:/scratch/gpfs/SUYOGHC/choices13k-stacking/results/*.pkl results/

    # Then run:
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
COLORS = {"EV": "#1f77b4", "EU": "#ff7f0e", "PT": "#2ca02c", "CPT": "#d62728"}
MODEL_NAMES = ["EV", "EU", "PT", "CPT"]

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGDIR = RESULTS_DIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load saved results."""
    print("Loading saved results...")

    results = {}

    # Load summary
    with open(RESULTS_DIR / "summary.pkl", "rb") as f:
        results["summary"] = pickle.load(f)

    # Load posteriors
    results["bayesian"] = az.from_netcdf(RESULTS_DIR / "bayesian_stacking.nc")
    results["hierarchical"] = az.from_netcdf(RESULTS_DIR / "hierarchical_bayesian.nc")
    results["mot"] = az.from_netcdf(RESULTS_DIR / "mot.nc")

    return results


def fig1_comparison(results):
    """Model comparison: MSE and weights."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    summary = results["summary"]

    # Panel A: Frequentist weights
    ax = axes[0]
    weights = summary["freq_weights"]
    colors = [COLORS[m] for m in MODEL_NAMES]
    bars = ax.bar(MODEL_NAMES, weights * 100, color=colors, edgecolor="black")
    ax.set_ylabel("Frequentist Weight (%)")
    ax.set_title("A. Optimal Stacking Weights")
    ax.set_ylim(0, 100)
    for bar, w in zip(bars, weights):
        if w > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{w*100:.1f}%", ha="center", fontsize=9)

    # Panel B: Bayesian weights
    ax = axes[1]
    means = summary["bayes_means"]
    lows = summary["bayes_hdi_low"]
    highs = summary["bayes_hdi_high"]

    x = np.arange(len(MODEL_NAMES))
    bars = ax.bar(x, means * 100, color=colors, edgecolor="black")
    ax.errorbar(x, means * 100,
                yerr=[means*100 - lows*100, highs*100 - means*100],
                fmt="none", color="black", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel("Bayesian Weight (%)")
    ax.set_title("B. Bayesian Posterior (94% HDI)")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig1_comparison.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: fig1_comparison.pdf")
    plt.close()


def fig2_posteriors(results):
    """Posterior distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bayesian stacking
    ax = axes[0]
    weights = results["bayesian"].posterior["weights"].values
    weights_flat = weights.reshape(-1, weights.shape[-1])

    for i, model in enumerate(MODEL_NAMES):
        sns.kdeplot(weights_flat[:, i] * 100, ax=ax, color=COLORS[model],
                    label=model, linewidth=2, fill=True, alpha=0.3)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Weight (%)")
    ax.set_ylabel("Density")
    ax.set_title("A. Bayesian Stacking Posteriors")
    ax.legend()
    ax.set_xlim(-5, 105)

    # Panel B: MOT
    ax = axes[1]
    pi = results["mot"].posterior["pi"].values
    pi_flat = pi.reshape(-1, pi.shape[-1])

    for i, model in enumerate(MODEL_NAMES):
        sns.kdeplot(pi_flat[:, i] * 100, ax=ax, color=COLORS[model],
                    label=model, linewidth=2, fill=True, alpha=0.3)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Theory Usage (%)")
    ax.set_ylabel("Density")
    ax.set_title("B. Mixture of Theories Posteriors")
    ax.legend()
    ax.set_xlim(-5, 105)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig2_posteriors.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig2_posteriors.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: fig2_posteriors.pdf")
    plt.close()


def fig3_hierarchical(results):
    """Hierarchical feature effects."""
    fig, ax = plt.subplots(figsize=(8, 6))

    beta = results["hierarchical"].posterior["beta"].values
    beta_flat = beta.reshape(-1, beta.shape[-2], beta.shape[-1])

    feature_names = ["LotNumB", "Amb", "Corr"]
    ref_models = MODEL_NAMES[:-1]  # exclude CPT (reference)

    y_pos = 0
    y_ticks = []
    y_labels = []

    for j, fname in enumerate(feature_names):
        for k, mname in enumerate(ref_models):
            samples = beta_flat[:, j, k]
            mean = samples.mean()
            lo, hi = np.percentile(samples, [3, 97])

            color = COLORS[mname]
            ax.plot([lo, hi], [y_pos, y_pos], color=color, linewidth=3)
            ax.scatter([mean], [y_pos], color=color, s=80, zorder=5, edgecolor="black")

            # Star if significant
            if lo > 0 or hi < 0:
                ax.scatter([mean], [y_pos], color=color, s=150, marker="*", zorder=6)

            y_ticks.append(y_pos)
            y_labels.append(f"{fname} → {mname}")
            y_pos += 1
        y_pos += 0.5

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Effect on Log-Odds (vs CPT)")
    ax.set_title("Feature Effects on Model Weights (* = 94% HDI excludes 0)")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_hierarchical.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig3_hierarchical.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: fig3_hierarchical.pdf")
    plt.close()


def fig4_mot_kappa(results):
    """MOT overdispersion posterior."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Pi posteriors as violins
    ax = axes[0]
    pi = results["mot"].posterior["pi"].values
    pi_flat = pi.reshape(-1, pi.shape[-1])

    parts = ax.violinplot([pi_flat[:, i] * 100 for i in range(4)],
                          positions=range(4), showmeans=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[MODEL_NAMES[i]])
        body.set_alpha(0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel("Theory Usage (%)")
    ax.set_title("A. Mixture Proportions")
    ax.set_ylim(-5, 105)

    # Panel B: Kappa posterior
    ax = axes[1]
    kappa = results["mot"].posterior["kappa"].values.flatten()
    ax.hist(kappa, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(kappa.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {kappa.mean():.1f}")
    ax.set_xlabel("Kappa (overdispersion)")
    ax.set_ylabel("Density")
    ax.set_title("B. Overdispersion Parameter")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig4_mot.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig4_mot.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: fig4_mot.pdf")
    plt.close()


def fig5_diagnostics(results):
    """MCMC diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Bayesian stacking trace
    az.plot_trace(results["bayesian"], var_names=["weights"], axes=axes[0:1, :].flatten()[:2])
    axes[0, 0].set_title("Bayesian Stacking: Trace")
    axes[0, 1].set_title("Bayesian Stacking: Posterior")

    # MOT trace
    az.plot_trace(results["mot"], var_names=["pi"], axes=axes[1:2, :].flatten()[:2])
    axes[1, 0].set_title("MOT: Trace")
    axes[1, 1].set_title("MOT: Posterior")

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig5_diagnostics.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig5_diagnostics.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: fig5_diagnostics.pdf")
    plt.close()


def summary_table(results):
    """Print summary table."""
    summary = results["summary"]

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Model':<8} {'Freq.':>10} {'Bayes Mean':>12} {'Bayes 94% HDI':>18} {'MOT':>10}")
    print("-" * 70)

    for i, m in enumerate(MODEL_NAMES):
        freq = summary["freq_weights"][i] * 100
        bayes = summary["bayes_means"][i] * 100
        lo = summary["bayes_hdi_low"][i] * 100
        hi = summary["bayes_hdi_high"][i] * 100
        mot = summary["mot_pi"][i] * 100
        print(f"{m:<8} {freq:>9.1f}% {bayes:>11.1f}% [{lo:>5.1f}, {hi:>5.1f}] {mot:>9.1f}%")

    print("-" * 70)
    print(f"\nMOT Kappa (overdispersion): {summary['mot_kappa']:.1f}")
    print(f"Total runtime: {summary['elapsed_minutes']:.1f} minutes")


def main():
    # Check files exist
    required = ["summary.pkl", "bayesian_stacking.nc", "hierarchical_bayesian.nc", "mot.nc"]
    missing = [f for f in required if not (RESULTS_DIR / f).exists()]
    if missing:
        print(f"Missing files in {RESULTS_DIR}:")
        for f in missing:
            print(f"  - {f}")
        print("\nCopy from Della:")
        print("  scp della:/scratch/gpfs/SUYOGHC/choices13k-stacking/results/*.nc results/")
        print("  scp della:/scratch/gpfs/SUYOGHC/choices13k-stacking/results/*.pkl results/")
        sys.exit(1)

    results = load_results()

    print("\nGenerating figures...")
    fig1_comparison(results)
    fig2_posteriors(results)
    fig3_hierarchical(results)
    fig4_mot_kappa(results)
    fig5_diagnostics(results)

    summary_table(results)

    print(f"\nFigures saved to: {FIGDIR}")


if __name__ == "__main__":
    main()
