"""MVP: K-fold stacking on choices13k.

Run EV, EU, PT on the feedback subset (~12k problems),
compute stacking weights, compare to best single model.

Usage: python scripts/run_stacking_mvp.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stacking.config import DataConfig, StackingConfig, ModelFitConfig
from stacking.data import load_selections, load_problems
from stacking.models import EVModel, EUModel, PTModel, MODEL_REGISTRY
from stacking.stacking import run_kfold_stacking, print_stacking_results


def main():
    # --- Configuration (Poldrack §7: no hardcoded paths) ---
    data_config = DataConfig(
        data_dir=Path(__file__).parent.parent / "data",
    )
    stacking_config = StackingConfig(
        n_folds=10,
        random_seed=42,
        loss="mse",
    )
    fit_config = ModelFitConfig(
        max_iterations=500,
        n_restarts=3,
        random_seed=42,
    )

    # --- Load and validate data ---
    print("Loading data...")
    df = load_selections(data_config)
    problems = load_problems(data_config)
    print(f"  {len(df)} problem-conditions loaded")
    print(f"  {df['Problem'].nunique()} unique problems")
    print(f"  bRate range: [{df['bRate'].min():.3f}, {df['bRate'].max():.3f}]")

    # --- MVP: start with 3 classic models ---
    model_classes = [EVModel, EUModel, PTModel]
    print(f"\nModels: {[m.name for m in model_classes]}")
    print(f"K-fold CV with K={stacking_config.n_folds}")

    # --- Run stacking ---
    print("\nRunning K-fold stacking...")
    results = run_kfold_stacking(
        model_classes=model_classes,
        df=df,
        problems=problems,
        stacking_config=stacking_config,
        fit_config=fit_config,
    )

    # --- Report results ---
    print_stacking_results(results)

    # --- Save results ---
    import numpy as np
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    np.savez(
        output_dir / "stacking_mvp_results.npz",
        weights=results["weights"],
        oof_predictions=results["oof_predictions"],
        per_model_mse=results["per_model_mse"],
        stacked_mse=results["stacked_mse"],
        model_names=results["model_names"],
    )
    print(f"\nResults saved to {output_dir / 'stacking_mvp_results.npz'}")


if __name__ == "__main__":
    main()
