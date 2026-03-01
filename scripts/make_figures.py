#!/usr/bin/env python
"""Generate paper and appendix figures for choices13k stacking analysis.

Creates:
  - Figure 1: Model comparison (MSE + frequentist weights)
  - Figure 2: Bayesian posterior distributions on weights
  - Figure 3: Feature effects on weights (hierarchical)
  - Figure 4: Per-problem weight variation by features
  - Appendix figures: diagnostics, convergence, etc.

Usage:
    pip install -e ".[bayesian]"
    pip install matplotlib seaborn
    python scripts/make_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.preprocessing import StandardScaler
from stacking.config import DataConfig, StackingConfig, ModelFitConfig
from stacking.data import load_selections, load_problems
from stacking.models import EVModel, EUModel, PTModel, CPTModel
from stacking.stacking import run_kfold_stacking
from stacking.bayesian import (
    run_bayesian_stacking,
    run_hierarchical_bayesian,
    BayesianStackingConfig,
    HierarchicalBayesianConfig,
)

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"EV": "#1f77b4", "EU": "#ff7f0e", "PT": "#2ca02c", "CPT": "#d62728"}
FIGDIR = Path(__file__).parent.parent / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load choices13k data."""
    print("Loading data...")
    config = DataConfig(data_dir=Path(__file__).parent.parent / "data")
    df = load_selections(config)
    problems = load_problems(config)
    return df, problems


def run_frequentist_stacking(df, problems):
    """Run K-fold stacking to get OOF predictions and weights."""
    print("Running frequentist stacking...")
    stacking_config = StackingConfig(n_folds=5, random_seed=42)
    fit_config = ModelFitConfig(n_restarts=3)

    results = run_kfold_stacking(
        model_classes=[EVModel, EUModel, PTModel, CPTModel],
        df=df,
        problems=problems,
        stacking_config=stacking_config,
        fit_config=fit_config,
    )
    return results


def run_bayesian(oof_predictions, df):
    """Run Bayesian stacking."""
    print("Running Bayesian stacking...")
    config = BayesianStackingConfig(
        n_samples=2000, n_tune=1000, n_chains=4, target_accept=0.9
    )
    return run_bayesian_stacking(
        oof_predictions=oof_predictions,
        observed_brate=df["bRate"].values,
        sample_sizes=df["n"].values,
        model_names=["EV", "EU", "PT", "CPT"],
        config=config,
    )


def run_hierarchical(oof_predictions, df):
    """Run hierarchical Bayesian stacking."""
    print("Running hierarchical Bayesian stacking...")

    # Prepare features
    feature_cols = ["LotNumB", "Amb", "Corr"]
    raw_features = df[feature_cols].values.astype(float)
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)

    config = HierarchicalBayesianConfig(
        n_samples=2000, n_tune=1000, n_chains=4, target_accept=0.95
    )
    return run_hierarchical_bayesian(
        oof_predictions=oof_predictions,
        observed_brate=df["bRate"].values,
        sample_sizes=df["n"].values,
        features=features,
        model_names=["EV", "EU", "PT", "CPT"],
        feature_names=feature_cols,
        config=config,
    )


# =============================================================================
# FIGURE 1: Model Comparison
# =============================================================================

def fig1_model_comparison(freq_results):
    """Bar chart showing MSE and stacking weights."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    models = ["EV", "EU", "PT", "CPT"]
    mses = freq_results["per_model_mse"]
    weights = freq_results["weights"]
    colors = [COLORS[m] for m in models]

    # Panel A: MSE
    ax = axes[0]
    bars = ax.bar(models, mses, color=colors, edgecolor="black", linewidth=1)
    ax.axhline(freq_results["stacked_mse"], color="black", linestyle="--",
               label=f"Stacked ({freq_results['stacked_mse']:.4f})")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("A. Prediction Error")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(mses) * 1.1)

    # Add value labels
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{mse:.4f}", ha="center", va="bottom", fontsize=9)

    # Panel B: Weights
    ax = axes[1]
    bars = ax.bar(models, weights * 100, color=colors, edgecolor="black", linewidth=1)
    ax.set_ylabel("Stacking Weight (%)")
    ax.set_title("B. Optimal Weights")
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, w in zip(bars, weights):
        if w > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{w*100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_model_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig1_model_comparison.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / 'fig1_model_comparison.pdf'}")
    plt.close(fig)


# =============================================================================
# FIGURE 2: Bayesian Posterior Distributions
# =============================================================================

def fig2_bayesian_posteriors(bayesian_results):
    """Ridge/violin plots of posterior weight distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Extract posterior samples
    weights = bayesian_results.idata.posterior["weights"].values
    weights_flat = weights.reshape(-1, weights.shape[-1])  # (samples, models)

    models = bayesian_results.model_names

    # Create violin plot
    parts = ax.violinplot(
        [weights_flat[:, i] * 100 for i in range(len(models))],
        positions=range(len(models)),
        showmeans=True,
        showmedians=False,
    )

    # Color the violins
    for i, (body, model) in enumerate(zip(parts["bodies"], models)):
        body.set_facecolor(COLORS[model])
        body.set_alpha(0.7)

    # Style the mean lines
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2)

    # Add HDI intervals as error bars
    for i, model in enumerate(models):
        mean = bayesian_results.weight_means[i] * 100
        lo = bayesian_results.weight_hdi_low[i] * 100
        hi = bayesian_results.weight_hdi_high[i] * 100
        ax.plot([i, i], [lo, hi], color="black", linewidth=3, solid_capstyle="round")
        ax.text(i + 0.15, mean, f"{mean:.1f}%\n[{lo:.1f}, {hi:.1f}]",
                va="center", fontsize=9)

    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Posterior Weight (%)")
    ax.set_xlabel("Decision Theory")
    ax.set_title("Bayesian Stacking: Posterior Weight Distributions (94% HDI)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig2_bayesian_posteriors.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig2_bayesian_posteriors.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / 'fig2_bayesian_posteriors.pdf'}")
    plt.close(fig)


# =============================================================================
# FIGURE 3: Feature Effects (Forest Plot)
# =============================================================================

def fig3_feature_effects(hier_results):
    """Forest plot showing beta coefficients with credible intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))

    features = hier_results.feature_names
    models = hier_results.model_names[:-1]  # exclude reference model

    n_features = len(features)
    n_models = len(models)

    # Extract beta posterior
    beta_post = hier_results.idata.posterior["beta"].values
    beta_flat = beta_post.reshape(-1, n_features, n_models)

    # Compute HDI for each beta
    y_positions = []
    y_labels = []

    y = 0
    for j, fname in enumerate(features):
        for k, mname in enumerate(models):
            samples = beta_flat[:, j, k]
            mean = samples.mean()
            lo, hi = np.percentile(samples, [3, 97])

            color = COLORS[mname]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.scatter([mean], [y], color=color, s=50, zorder=5, edgecolor="black")

            # Mark if credibly non-zero
            if lo > 0 or hi < 0:
                ax.scatter([mean], [y], color=color, s=100, marker="*", zorder=6)

            y_positions.append(y)
            y_labels.append(f"{fname} → {mname}")
            y += 1
        y += 0.5  # gap between features

    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Effect on Log-Odds (vs CPT)")
    ax.set_title("Feature Effects on Model Weights\n(★ = 94% HDI excludes zero)")
    ax.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=COLORS[m], linewidth=2, label=m)
                       for m in models]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_feature_effects.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig3_feature_effects.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / 'fig3_feature_effects.pdf'}")
    plt.close(fig)


# =============================================================================
# FIGURE 4: Per-Problem Weight Variation
# =============================================================================

def fig4_weight_variation(hier_results, df):
    """Scatter plots showing weight variation by features."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    weights = hier_results.weights_per_problem
    models = hier_results.model_names

    # Feature values (raw)
    features = {
        "LotNumB": df["LotNumB"].values,
        "Amb": df["Amb"].values.astype(float),
    }

    # Panel A: CPT weight vs LotNumB
    ax = axes[0, 0]
    ax.scatter(features["LotNumB"], weights[:, 3] * 100, alpha=0.1, s=5, c=COLORS["CPT"])
    # Bin means
    bins = np.arange(1, df["LotNumB"].max() + 2)
    bin_idx = np.digitize(features["LotNumB"], bins) - 1
    bin_means = [weights[bin_idx == i, 3].mean() * 100 for i in range(len(bins)-1)]
    ax.plot(bins[:-1] + 0.5, bin_means, "ko-", markersize=8, linewidth=2, label="Bin mean")
    ax.set_xlabel("Number of Lottery Outcomes")
    ax.set_ylabel("CPT Weight (%)")
    ax.set_title("A. CPT Weight vs Complexity")
    ax.legend()

    # Panel B: PT weight vs LotNumB
    ax = axes[0, 1]
    ax.scatter(features["LotNumB"], weights[:, 2] * 100, alpha=0.1, s=5, c=COLORS["PT"])
    bin_means = [weights[bin_idx == i, 2].mean() * 100 for i in range(len(bins)-1)]
    ax.plot(bins[:-1] + 0.5, bin_means, "ko-", markersize=8, linewidth=2, label="Bin mean")
    ax.set_xlabel("Number of Lottery Outcomes")
    ax.set_ylabel("PT Weight (%)")
    ax.set_title("B. PT Weight vs Complexity")
    ax.legend()

    # Panel C: EU weight by Ambiguity
    ax = axes[1, 0]
    amb_vals = [0, 1]
    eu_by_amb = [weights[features["Amb"] == a, 1] * 100 for a in amb_vals]
    parts = ax.violinplot(eu_by_amb, positions=amb_vals, showmeans=True)
    for body in parts["bodies"]:
        body.set_facecolor(COLORS["EU"])
        body.set_alpha(0.7)
    ax.set_xticks(amb_vals)
    ax.set_xticklabels(["No Ambiguity", "Ambiguity"])
    ax.set_ylabel("EU Weight (%)")
    ax.set_title("C. EU Weight by Ambiguity")

    # Panel D: Weight distribution summary
    ax = axes[1, 1]
    data_for_box = [weights[:, i] * 100 for i in range(len(models))]
    bp = ax.boxplot(data_for_box, labels=models, patch_artist=True)
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.7)
    ax.set_ylabel("Weight (%)")
    ax.set_title("D. Weight Distribution Across Problems")

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig4_weight_variation.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig4_weight_variation.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / 'fig4_weight_variation.pdf'}")
    plt.close(fig)


# =============================================================================
# APPENDIX: Diagnostics
# =============================================================================

def figA1_mcmc_diagnostics(bayesian_results):
    """Trace plots and R-hat for MCMC diagnostics."""
    import arviz as az

    fig = plt.figure(figsize=(12, 6))

    # Trace plot
    axes = az.plot_trace(
        bayesian_results.idata,
        var_names=["weights"],
        figsize=(12, 6),
    )

    plt.suptitle(f"MCMC Diagnostics (R-hat max: {bayesian_results.r_hat_max:.4f}, "
                 f"ESS min: {bayesian_results.ess_min:.0f})")
    plt.tight_layout()
    plt.savefig(FIGDIR / "figA1_mcmc_diagnostics.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(FIGDIR / "figA1_mcmc_diagnostics.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / 'figA1_mcmc_diagnostics.pdf'}")
    plt.close()


def figA2_baseline_weights(hier_results):
    """Posterior distribution of baseline (intercept-only) weights."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = hier_results.model_names

    # Extract baseline weights from posterior
    intercept = hier_results.idata.posterior["intercept"].values
    intercept_flat = intercept.reshape(-1, len(models) - 1)

    # Compute softmax for each sample
    logits = np.concatenate([intercept_flat, np.zeros((len(intercept_flat), 1))], axis=1)
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    baseline_weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Violin plot
    parts = ax.violinplot(
        [baseline_weights[:, i] * 100 for i in range(len(models))],
        positions=range(len(models)),
        showmeans=True,
    )

    for i, (body, model) in enumerate(zip(parts["bodies"], models)):
        body.set_facecolor(COLORS[model])
        body.set_alpha(0.7)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("Baseline Weight (%)")
    ax.set_title("Hierarchical Model: Baseline Weights (Intercept Only)")
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    fig.savefig(FIGDIR / "figA2_baseline_weights.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "figA2_baseline_weights.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / 'figA2_baseline_weights.pdf'}")
    plt.close(fig)


def figA3_results_table(freq_results, bayesian_results, hier_results):
    """Create a summary table as a figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Build table data
    models = ["EV", "EU", "PT", "CPT"]

    table_data = []
    headers = ["Model", "MSE", "Freq. Weight", "Bayes. Mean", "Bayes. 94% HDI",
               "Hier. Baseline", "Hier. Range"]

    for i, m in enumerate(models):
        row = [
            m,
            f"{freq_results['per_model_mse'][i]:.4f}",
            f"{freq_results['weights'][i]*100:.1f}%",
            f"{bayesian_results.weight_means[i]*100:.1f}%",
            f"[{bayesian_results.weight_hdi_low[i]*100:.1f}, {bayesian_results.weight_hdi_high[i]*100:.1f}]",
            f"{hier_results.baseline_weight_means[i]*100:.1f}%",
            f"[{hier_results.weights_per_problem[:,i].min()*100:.1f}, {hier_results.weights_per_problem[:,i].max()*100:.1f}]",
        ]
        table_data.append(row)

    # Add stacked row
    table_data.append([
        "Stacked", f"{freq_results['stacked_mse']:.4f}", "—", "—", "—", "—", "—"
    ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color the model name cells
    for i, m in enumerate(models):
        table[(i+1, 0)].set_facecolor(COLORS[m])
        table[(i+1, 0)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Summary of Stacking Results", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(FIGDIR / "figA3_results_table.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "figA3_results_table.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / 'figA3_results_table.pdf'}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    # Load data
    df, problems = load_data()

    # Run analyses
    freq_results = run_frequentist_stacking(df, problems)
    bayesian_results = run_bayesian(freq_results["oof_predictions"], df)
    hier_results = run_hierarchical(freq_results["oof_predictions"], df)

    # Generate figures
    print("\nGenerating figures...")

    print("\n[Figure 1] Model Comparison")
    fig1_model_comparison(freq_results)

    print("\n[Figure 2] Bayesian Posteriors")
    fig2_bayesian_posteriors(bayesian_results)

    print("\n[Figure 3] Feature Effects")
    fig3_feature_effects(hier_results)

    print("\n[Figure 4] Weight Variation")
    fig4_weight_variation(hier_results, df)

    print("\n[Appendix A1] MCMC Diagnostics")
    figA1_mcmc_diagnostics(bayesian_results)

    print("\n[Appendix A2] Baseline Weights")
    figA2_baseline_weights(hier_results)

    print("\n[Appendix A3] Results Table")
    figA3_results_table(freq_results, bayesian_results, hier_results)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
