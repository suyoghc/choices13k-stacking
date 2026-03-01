"""Fit decision models to choice data via MLE.

Separate from model definitions (Poldrack §3: Single Responsibility).
Models define predict(); this module handles optimization.
"""

import numpy as np
from scipy.optimize import minimize

from .config import ModelFitConfig


def compute_mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error between predicted and observed bRate."""
    return np.mean((predictions - targets) ** 2)


def compute_cross_entropy_loss(
    predictions: np.ndarray, targets: np.ndarray
) -> float:
    """Binary cross-entropy (treating bRate as soft label)."""
    eps = 1e-10
    p = np.clip(predictions, eps, 1 - eps)
    return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))


# Theory-informed starting points for each model.
# First start is always the "textbook" parameterization.
# Remaining starts sample from a narrower region around plausible values.
# Rationale: random uniform over (0.01, 100) for temperature wastes
# most restarts in bad basins. Kahneman & Tversky (1992) and
# Peterson et al. (2021) give us strong priors on where params live.
THEORY_STARTS = {
    # EV: only temperature. Peterson et al. found τ ≈ 0.1-1.0
    "EV": [
        [0.5],
        [0.1],
        [1.0],
    ],
    # EU: [alpha, temperature]. Tversky & Kahneman (1992): α ≈ 0.88
    "EU": [
        [0.88, 0.5],
        [0.5, 0.1],
        [0.88, 1.0],
    ],
    # PT: [alpha, lambda_, gamma, temperature]. TK92: α≈0.88, λ≈2.25, γ≈0.61
    "PT": [
        [0.88, 2.25, 0.61, 0.5],
        [0.5, 1.0, 0.7, 0.1],
        [0.88, 2.25, 0.61, 1.0],
    ],
    # CPT: [alpha, lambda_, gamma_pos, gamma_neg, temperature]
    "CPT": [
        [0.88, 2.25, 0.61, 0.69, 0.5],
        [0.5, 1.0, 0.7, 0.7, 0.1],
        [0.88, 2.25, 0.61, 0.69, 1.0],
    ],
}


def _generate_starting_points(model_class, config: ModelFitConfig):
    """Yield theory-informed starts, then random restarts.

    Theory starts first (best priors), random fills remaining budget.
    """
    rng = np.random.RandomState(config.random_seed)

    # Theory-informed starts
    theory = THEORY_STARTS.get(model_class.name, [])
    for x0 in theory:
        yield np.array(x0, dtype=float)

    # Random restarts — sample from narrower "plausible" region
    # within bounds, log-uniform for temperature
    n_random = max(0, config.n_restarts - len(theory))
    for _ in range(n_random):
        x0 = []
        for i, (lo, hi) in enumerate(model_class.param_bounds):
            name = model_class.param_names[i] if hasattr(model_class, 'param_names') else ""
            if "temperature" in name:
                # Log-uniform for scale parameters
                x0.append(np.exp(rng.uniform(np.log(lo), np.log(min(hi, 10.0)))))
            else:
                x0.append(rng.uniform(lo, min(hi, 3.0)))
        yield np.array(x0)


def fit_model(model_class, gamble_data, config: ModelFitConfig, loss: str = "mse"):
    """Fit a decision model to data via multi-start optimization.

    Uses theory-informed starting points (Tversky & Kahneman 1992)
    followed by random restarts. Best result across all starts wins.

    Args:
        model_class: model with .predict(params, data), .param_bounds, .n_params
        gamble_data: GambleData with observed bRate
        config: optimization settings
        loss: "mse" or "cross_entropy"

    Returns:
        best_params: np.ndarray of fitted parameters
        best_loss: float
    """
    loss_fn = compute_mse_loss if loss == "mse" else compute_cross_entropy_loss
    targets = gamble_data.brate

    def objective(params):
        try:
            preds = model_class.predict(params, gamble_data)
            return loss_fn(preds, targets)
        except (ValueError, FloatingPointError, OverflowError):
            return 1e10  # penalty for invalid params

    best_params = None
    best_loss = np.inf

    for x0 in _generate_starting_points(model_class, config):
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=model_class.param_bounds,
            options={
                "maxiter": config.max_iterations,
                "ftol": config.tolerance,
            },
        )

        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x

    if best_params is None:
        raise RuntimeError(
            f"All optimization starts failed for {model_class.name}"
        )

    return best_params, best_loss
