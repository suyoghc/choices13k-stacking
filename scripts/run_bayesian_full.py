#!/usr/bin/env python
"""Run full Bayesian analysis pipeline on HPC.

This script runs all Bayesian analyses with production settings:
1. Frequentist stacking (to get OOF predictions)
2. Bayesian stacking (uniform weights)
3. Hierarchical Bayesian stacking (feature-varying weights)
4. Mixture of Theories (MOT)

Saves all results to results/ as pickle and NetCDF files.

Usage:
    python scripts/run_bayesian_full.py [--quick]

    --quick: Use fewer samples for testing (default: production settings)
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.preprocessing import StandardScaler
from stacking.config import DataConfig, StackingConfig, ModelFitConfig
from stacking.data import load_selections, load_problems
from stacking.models import EVModel, EUModel, PTModel, CPTModel, ContextModel
from stacking.stacking import run_kfold_stacking
from stacking.bayesian import (
    run_bayesian_stacking,
    run_hierarchical_bayesian,
    run_mot,
    print_bayesian_results,
    print_hierarchical_bayesian_results,
    print_mot_results,
    BayesianStackingConfig,
    HierarchicalBayesianConfig,
    MOTConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full Bayesian analysis")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer samples for testing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Settings
    if args.quick:
        print("=" * 60)
        print("QUICK MODE: Using reduced samples for testing")
        print("=" * 60)
        n_samples = 500
        n_tune = 250
        n_chains = 2
        n_folds = 3
        n_restarts = 2
    else:
        print("=" * 60)
        print("PRODUCTION MODE: Full MCMC sampling")
        print("=" * 60)
        n_samples = 4000
        n_tune = 2000
        n_chains = 4
        n_folds = 10
        n_restarts = 5

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    start_time = time.time()

    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("\n[1/5] Loading data...")
    data_config = DataConfig(data_dir=Path(__file__).parent.parent / "data")
    df = load_selections(data_config)
    problems = load_problems(data_config)
    print(f"  Loaded {len(df)} problems")

    # =========================================================================
    # 2. Frequentist stacking (for OOF predictions)
    # =========================================================================
    print(f"\n[2/5] Running {n_folds}-fold stacking...")
    stacking_config = StackingConfig(n_folds=n_folds, random_seed=42)
    fit_config = ModelFitConfig(n_restarts=n_restarts)

    freq_results = run_kfold_stacking(
        model_classes=[EVModel, EUModel, PTModel, CPTModel, ContextModel],
        df=df,
        problems=problems,
        stacking_config=stacking_config,
        fit_config=fit_config,
    )

    print(f"\n  Frequentist results:")
    print(f"    Stacked MSE: {freq_results['stacked_mse']:.4f}")
    print(f"    Weights: {freq_results['weights']}")

    # Save OOF predictions for reuse
    np.savez(
        results_dir / "oof_predictions.npz",
        oof_predictions=freq_results["oof_predictions"],
        model_names=freq_results["model_names"],
        brate=df["bRate"].values,
        sample_sizes=df["n"].values,
    )

    # =========================================================================
    # 3. Bayesian stacking (uniform weights)
    # =========================================================================
    print(f"\n[3/5] Running Bayesian stacking ({n_chains} chains x {n_samples} samples)...")
    bayes_config = BayesianStackingConfig(
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=0.9,
        random_seed=42,
    )

    bayes_results = run_bayesian_stacking(
        oof_predictions=freq_results["oof_predictions"],
        observed_brate=df["bRate"].values,
        sample_sizes=df["n"].values,
        model_names=freq_results["model_names"],
        config=bayes_config,
    )

    print_bayesian_results(bayes_results)

    # Save posterior
    bayes_results.idata.to_netcdf(results_dir / "bayesian_stacking.nc")

    # =========================================================================
    # 4. Hierarchical Bayesian stacking
    # =========================================================================
    print(f"\n[4/5] Running hierarchical Bayesian stacking...")

    # Prepare features
    feature_cols = ["LotNumB", "Amb", "Corr"]
    raw_features = df[feature_cols].values.astype(float)
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)

    hier_config = HierarchicalBayesianConfig(
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=0.95,
        random_seed=42,
    )

    hier_results = run_hierarchical_bayesian(
        oof_predictions=freq_results["oof_predictions"],
        observed_brate=df["bRate"].values,
        sample_sizes=df["n"].values,
        features=features,
        model_names=freq_results["model_names"],
        feature_names=feature_cols,
        config=hier_config,
    )

    print_hierarchical_bayesian_results(hier_results)

    # Save posterior
    hier_results.idata.to_netcdf(results_dir / "hierarchical_bayesian.nc")

    # =========================================================================
    # 5. Mixture of Theories (MOT)
    # =========================================================================
    print(f"\n[5/5] Running Mixture of Theories...")

    mot_config = MOTConfig(
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=0.9,
        use_overdispersion=True,
        random_seed=42,
    )

    mot_results = run_mot(
        oof_predictions=freq_results["oof_predictions"],
        observed_brate=df["bRate"].values,
        sample_sizes=df["n"].values,
        model_names=freq_results["model_names"],
        config=mot_config,
    )

    print_mot_results(mot_results)

    # Save posterior
    mot_results.idata.to_netcdf(results_dir / "mot.nc")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {results_dir}")
    print("  - oof_predictions.npz")
    print("  - bayesian_stacking.nc")
    print("  - hierarchical_bayesian.nc")
    print("  - mot.nc")

    # Save summary
    summary = {
        "freq_weights": freq_results["weights"],
        "freq_mse": freq_results["stacked_mse"],
        "bayes_means": bayes_results.weight_means,
        "bayes_hdi_low": bayes_results.weight_hdi_low,
        "bayes_hdi_high": bayes_results.weight_hdi_high,
        "hier_baseline": hier_results.baseline_weight_means,
        "hier_beta": hier_results.beta_means,
        "mot_pi": mot_results.pi_means,
        "mot_kappa": mot_results.kappa_mean,
        "model_names": freq_results["model_names"],
        "elapsed_minutes": elapsed / 60,
    }
    with open(results_dir / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print("\nKey results:")
    for i, name in enumerate(freq_results["model_names"]):
        print(f"  Bayesian {name:>7s} weight: {bayes_results.weight_means[i]*100:.1f}% "
              f"[{bayes_results.weight_hdi_low[i]*100:.1f}, "
              f"{bayes_results.weight_hdi_high[i]*100:.1f}]")
    print(f"  MOT kappa: {mot_results.kappa_mean:.1f}")


if __name__ == "__main__":
    main()
