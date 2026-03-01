"""Hierarchical stacking: weights that vary by problem features.

Standard stacking asks "which model is best overall?"
Hierarchical stacking asks "which model is best HERE?"

The idea (Yao et al. 2022, Bayesian Stacking):
  w_k(x) = softmax(X @ β_k)  for model k
  
  where X is a matrix of problem features and β are learned coefficients.
  When β = 0, this reduces to uniform weights — standard stacking.

This reveals which problem features determine model performance,
e.g., "PT is best for mixed gambles, but EV wins for dominated ones."
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from .fitting import compute_mse_loss


def softmax_rows(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def compute_hierarchical_weights(
    features: np.ndarray,
    beta: np.ndarray,
    n_models: int,
) -> np.ndarray:
    """Compute problem-specific stacking weights.

    Args:
        features: (n_problems, n_features) standardized problem features
        beta: (n_features, n_models-1) coefficient matrix
              Last model's coefficients are fixed at 0 (identifiability)
        n_models: total number of models

    Returns:
        weights: (n_problems, n_models) weights summing to 1 per row
    """
    # Logits for first n_models-1 models; last model is reference (logit=0)
    logits = np.zeros((features.shape[0], n_models))
    logits[:, :-1] = features @ beta
    return softmax_rows(logits)


def fit_hierarchical_stacking(
    oof_predictions: np.ndarray,
    targets: np.ndarray,
    features: np.ndarray,
    regularization: float = 0.1,
) -> dict:
    """Fit hierarchical stacking weights as function of features.

    Model: w_k(x) = softmax(intercept_k + X @ beta_k)
    
    The intercept learns the "average" weight structure (e.g., PT-dominated).
    Features then modulate around that baseline.
    Only feature coefficients are regularized — intercepts are free.

    Args:
        oof_predictions: (n, n_models) out-of-fold predictions
        targets: (n,) observed bRate
        features: (n, p) problem features (raw, will be standardized)
        regularization: L2 penalty on beta only (not intercept)

    Returns:
        dict with beta, intercept, scaler, feature_importance, etc.
    """
    n_problems, n_models = oof_predictions.shape
    
    # Standardize features (Poldrack §2: don't let scale differences
    # dominate the optimization)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_features = X.shape[1]

    # Parameters: intercept (n_models-1) + beta matrix (n_features × (n_models-1))
    # Last model is reference category (logit fixed at 0)
    n_intercept = n_models - 1
    n_beta = n_features * (n_models - 1)
    n_params = n_intercept + n_beta

    def _unpack(params_flat):
        intercept = params_flat[:n_intercept]
        beta = params_flat[n_intercept:].reshape(n_features, n_models - 1)
        return intercept, beta

    def _compute_weights(intercept, beta):
        logits = np.zeros((n_problems, n_models))
        logits[:, :-1] = intercept[np.newaxis, :] + X @ beta
        return softmax_rows(logits)

    def objective(params_flat):
        intercept, beta = _unpack(params_flat)
        weights = _compute_weights(intercept, beta)

        blended = np.sum(oof_predictions * weights, axis=1)
        loss = compute_mse_loss(blended, targets)

        # Regularize feature coefficients only, not intercepts
        penalty = regularization * np.sum(beta ** 2)
        return loss + penalty

    # Initialize: intercepts at 0 (uniform), beta at 0
    x0 = np.zeros(n_params)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    intercept, beta = _unpack(result.x)

    # Compute weights for all problems
    weights = _compute_weights(intercept, beta)
    blended = np.sum(oof_predictions * weights, axis=1)
    hierarchical_mse = compute_mse_loss(blended, targets)

    # Feature importance: magnitude of beta coefficients
    feature_importance = np.sqrt(np.sum(beta ** 2, axis=1))

    # Baseline weights (intercept only, no features)
    baseline_logits = np.zeros(n_models)
    baseline_logits[:-1] = intercept
    baseline_weights = np.exp(baseline_logits) / np.sum(np.exp(baseline_logits))

    return {
        "beta": beta,
        "intercept": intercept,
        "baseline_weights": baseline_weights,
        "scaler": scaler,
        "weights": weights,
        "hierarchical_mse": hierarchical_mse,
        "feature_importance": feature_importance,
        "optimization_result": result,
    }


def print_hierarchical_results(
    results: dict,
    feature_names: list[str],
    model_names: list[str],
    uniform_mse: float,
) -> None:
    """Pretty-print hierarchical stacking results."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL STACKING RESULTS")
    print("=" * 60)

    print(f"\nUniform stacking MSE:      {uniform_mse:.6f}")
    print(f"Hierarchical stacking MSE: {results['hierarchical_mse']:.6f}")
    improvement = (uniform_mse - results["hierarchical_mse"]) / uniform_mse * 100
    print(f"Improvement:               {improvement:.2f}%")

    # Baseline weights (intercept only)
    if "baseline_weights" in results:
        print("\nBaseline weights (intercept, no features):")
        for name, w in zip(model_names, results["baseline_weights"]):
            bar = "█" * int(w * 40)
            print(f"  {name:>6s}: {w:.4f} {bar}")

    # Feature importance ranking
    print("\nFeature importance (drives weight variation):")
    importance = results["feature_importance"]
    order = np.argsort(-importance)
    max_imp = importance.max() if importance.max() > 0 else 1.0
    for rank, idx in enumerate(order):
        bar = "█" * int(importance[idx] / max_imp * 30)
        print(f"  {rank+1}. {feature_names[idx]:>20s}: {importance[idx]:.4f} {bar}")

    # Weight distribution statistics
    weights = results["weights"]
    print(f"\nWeight distribution across problems:")
    for i, name in enumerate(model_names):
        w = weights[:, i]
        print(
            f"  {name:>6s}: mean={w.mean():.3f}, "
            f"min={w.min():.3f}, max={w.max():.3f}, "
            f"std={w.std():.3f}"
        )

    # Beta coefficients
    print(f"\nBeta coefficients (vs reference model: {model_names[-1]}):")
    beta = results["beta"]
    for j, fname in enumerate(feature_names):
        coefs = " ".join(
            f"{model_names[k]}={beta[j,k]:+.3f}"
            for k in range(beta.shape[1])
        )
        print(f"  {fname:>20s}: {coefs}")
