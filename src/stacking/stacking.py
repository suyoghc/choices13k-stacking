"""K-fold stacking for decision theory model combination.

Implements Breiman (1996) stacking:
1. K-fold CV: fit each model on K-1 folds, predict on held-out fold
2. Collect out-of-fold predictions for all problems
3. Find optimal weights minimizing prediction error

Weights w_k ≥ 0, Σ w_k = 1 (simplex constraint).
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from .config import StackingConfig, ModelFitConfig
from .models import GambleData, prepare_gamble_data
from .fitting import fit_model, compute_mse_loss


def compute_stacking_weights(
    oof_predictions: np.ndarray,
    targets: np.ndarray,
    loss: str = "mse",
) -> np.ndarray:
    """Find optimal stacking weights on the simplex.

    Args:
        oof_predictions: (n_problems, n_models) out-of-fold predictions
        targets: (n_problems,) observed bRate
        loss: "mse" loss function

    Returns:
        weights: (n_models,) optimal stacking weights summing to 1
    """
    n_models = oof_predictions.shape[1]

    if n_models == 1:
        return np.array([1.0])

    def objective(w):
        blended = oof_predictions @ w
        return compute_mse_loss(blended, targets)

    # Simplex constraints: w >= 0, sum(w) = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * n_models
    x0 = np.ones(n_models) / n_models  # uniform start

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        # Fall back to uniform weights with warning
        import warnings
        warnings.warn(
            f"Stacking weight optimization did not converge: {result.message}. "
            "Falling back to uniform weights."
        )
        return np.ones(n_models) / n_models

    return result.x


def run_kfold_stacking(
    model_classes: list,
    df,
    problems: dict,
    stacking_config: StackingConfig,
    fit_config: ModelFitConfig,
) -> dict:
    """Full K-fold stacking pipeline.

    Args:
        model_classes: list of model classes (EVModel, EUModel, etc.)
        df: selections DataFrame
        problems: raw problem specifications
        stacking_config: K-fold and loss settings
        fit_config: model optimization settings

    Returns:
        dict with keys:
            weights: (n_models,) stacking weights
            oof_predictions: (n_problems, n_models) out-of-fold preds
            model_names: list of model names
            per_model_mse: (n_models,) MSE for each model alone
            stacked_mse: float, MSE of stacked ensemble
            fitted_params: dict[model_name -> list of params per fold]
    """
    n_problems = len(df)
    n_models = len(model_classes)
    model_names = [m.name for m in model_classes]

    # Out-of-fold prediction matrix
    oof_predictions = np.full((n_problems, n_models), np.nan)

    # Track fitted parameters per fold
    fitted_params = {name: [] for name in model_names}

    kf = KFold(
        n_splits=stacking_config.n_folds,
        shuffle=True,
        random_state=stacking_config.random_seed,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"  Fold {fold_idx + 1}/{stacking_config.n_folds}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_data = prepare_gamble_data(train_df, problems)
        val_data = prepare_gamble_data(val_df, problems)

        for model_idx, model_class in enumerate(model_classes):
            print(f"    Fitting {model_class.name}...", end=" ")

            params, train_loss = fit_model(
                model_class, train_data, fit_config, loss=stacking_config.loss
            )
            fitted_params[model_class.name].append(params)

            # Predict on held-out fold
            val_preds = model_class.predict(params, val_data)
            oof_predictions[val_idx, model_idx] = val_preds

            val_mse = compute_mse_loss(val_preds, val_data.brate)
            print(f"train_loss={train_loss:.4f}, val_mse={val_mse:.4f}")

    # Defensive check: no NaN in OOF predictions
    assert not np.any(np.isnan(oof_predictions)), (
        "NaN found in out-of-fold predictions — "
        "some problems were not covered by CV folds"
    )

    targets = df["bRate"].values

    # Per-model MSE
    per_model_mse = np.array([
        compute_mse_loss(oof_predictions[:, i], targets)
        for i in range(n_models)
    ])

    # Stacking weights
    weights = compute_stacking_weights(
        oof_predictions, targets, loss=stacking_config.loss
    )

    # Stacked ensemble MSE
    stacked_preds = oof_predictions @ weights
    stacked_mse = compute_mse_loss(stacked_preds, targets)

    return {
        "weights": weights,
        "oof_predictions": oof_predictions,
        "model_names": model_names,
        "per_model_mse": per_model_mse,
        "stacked_mse": stacked_mse,
        "fitted_params": fitted_params,
    }


def print_stacking_results(results: dict) -> None:
    """Pretty-print stacking results."""
    print("\n" + "=" * 60)
    print("STACKING RESULTS")
    print("=" * 60)

    print("\nPer-model MSE (out-of-fold):")
    for name, mse in zip(results["model_names"], results["per_model_mse"]):
        print(f"  {name:>6s}: {mse:.6f}")

    print(f"\nBest single model: "
          f"{results['model_names'][np.argmin(results['per_model_mse'])]}"
          f" (MSE={np.min(results['per_model_mse']):.6f})")

    print(f"\nStacked ensemble MSE: {results['stacked_mse']:.6f}")

    improvement = (
        np.min(results["per_model_mse"]) - results["stacked_mse"]
    ) / np.min(results["per_model_mse"]) * 100
    print(f"Improvement over best single: {improvement:.2f}%")

    print("\nStacking weights:")
    for name, w in zip(results["model_names"], results["weights"]):
        bar = "█" * int(w * 40)
        print(f"  {name:>6s}: {w:.4f} {bar}")
