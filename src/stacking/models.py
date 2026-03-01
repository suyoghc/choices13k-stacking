"""Decision theory models for risky choice.

Implements the classic hierarchy from Peterson et al. (2021):
EV ⊂ EU ⊂ PT ⊂ CPT

Each model computes V(gamble) → P(choose B) via softmax.
Fit via scipy.optimize (MLE on observed bRate).

Single Responsibility (Poldrack §3): each model predicts choice probabilities.
Fitting is handled separately by the optimizer module.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.special import expit as sigmoid


# --- Softmax choice rule ---
# P(B) = exp(τ·V(B)) / [exp(τ·V(A)) + exp(τ·V(B))]
# Equivalent to sigmoid(τ·(V(B) - V(A)))
# τ = inverse temperature (higher = more deterministic)

def predict_choice_probability(
    value_a: np.ndarray,
    value_b: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Predict P(choose B) via softmax choice rule.

    Returns array of probabilities in (0, 1).
    """
    # Using sigmoid for numerical stability
    return sigmoid(temperature * (value_b - value_a))


# --- Gamble valuation functions ---
# Each takes gamble outcomes/probabilities and model parameters,
# returns subjective value V(gamble).


def compute_expected_value(
    outcomes: list[list[float]],
) -> float:
    """EV: V = Σ p_i · x_i. No free parameters."""
    return sum(p * x for p, x in outcomes)


def apply_power_utility(x: float, alpha: float, lambda_: float) -> float:
    """Power utility with loss aversion (Tversky & Kahneman 1992).

    u(x) = x^α          if x ≥ 0
    u(x) = -λ·(-x)^α    if x < 0

    Parameters:
        alpha: diminishing sensitivity (0 < α ≤ 1)
        lambda_: loss aversion coefficient (λ > 0; λ > 1 = loss averse)
    """
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_ * ((-x) ** alpha)


def apply_probability_weight(p: float, gamma: float) -> float:
    """Kahneman-Tversky probability weighting function.

    w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)

    Overweights small probabilities when γ < 1.
    Linear (no distortion) when γ = 1.

    Parameters:
        gamma: curvature parameter (0 < γ ≤ 1 typically)
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    numerator = p ** gamma
    denominator = (numerator + (1 - p) ** gamma) ** (1 / gamma)
    return numerator / denominator


# --- Model classes ---
# Each wraps parameter vector → predicted bRate for all problems.
# Interface: params (flat array) + data → predictions (array of P(B)).


@dataclass
class GambleData:
    """Pre-extracted gamble data for vectorized model evaluation.

    Each field is array of shape (n_problems, max_outcomes).
    Padded with zeros where a gamble has fewer outcomes.
    """

    outcomes_a: np.ndarray  # (n, max_outcomes)
    probs_a: np.ndarray     # (n, max_outcomes)
    outcomes_b: np.ndarray  # (n, max_outcomes)
    probs_b: np.ndarray     # (n, max_outcomes)
    brate: np.ndarray       # (n,) observed choice rates


def prepare_gamble_data(df, problems: dict) -> GambleData:
    """Convert raw data into padded arrays for vectorized evaluation.

    Args:
        df: selections DataFrame (must have _json_idx column from load_selections)
        problems: dict keyed by CSV row index (string) -> {"A": [...], "B": [...]}
    """
    if "_json_idx" not in df.columns:
        raise ValueError(
            "DataFrame missing _json_idx column. "
            "Use load_selections() to create it."
        )

    n = len(df)

    # Find max number of outcomes across all gambles
    max_out = 1
    for _, row in df.iterrows():
        jkey = str(int(row["_json_idx"]))
        if jkey in problems:
            prob = problems[jkey]
            max_out = max(max_out, len(prob["A"]), len(prob["B"]))

    # Pre-allocate padded arrays
    outcomes_a = np.zeros((n, max_out))
    probs_a = np.zeros((n, max_out))
    outcomes_b = np.zeros((n, max_out))
    probs_b = np.zeros((n, max_out))

    for i, (_, row) in enumerate(df.iterrows()):
        jkey = str(int(row["_json_idx"]))
        prob = problems[jkey]
        for j, (p, x) in enumerate(prob["A"]):
            probs_a[i, j] = p
            outcomes_a[i, j] = x
        for j, (p, x) in enumerate(prob["B"]):
            probs_b[i, j] = p
            outcomes_b[i, j] = x

    # --- Validation: spot-check JSON ↔ selections alignment ---
    # Compare first outcome of gamble A against Ha/pHa columns.
    # This catches index-mapping bugs like the one that made
    # every model predict 0.5 (Poldrack §5: the conspiracy paper).
    sample_idx = min(10, n)
    for i in range(sample_idx):
        row = df.iloc[i]
        assert abs(outcomes_a[i, 0] - row["Ha"]) < 0.1 or abs(probs_a[i, 0] - row["pHa"]) < 0.01, (
            f"Row {i}: JSON/selections mismatch! "
            f"JSON A[0]=({probs_a[i,0]:.3f}, {outcomes_a[i,0]:.1f}), "
            f"selections Ha={row['Ha']}, pHa={row['pHa']}. "
            f"Check _json_idx mapping."
        )

    return GambleData(
        outcomes_a=outcomes_a,
        probs_a=probs_a,
        outcomes_b=outcomes_b,
        probs_b=probs_b,
        brate=df["bRate"].values,
    )


class EVModel:
    """Expected Value model. No free parameters except temperature.

    V(gamble) = Σ p_i · x_i
    P(B) = sigmoid(τ · (V(B) - V(A)))
    """

    name = "EV"
    n_params = 1  # temperature only
    param_names = ["temperature"]
    # Bounds for scipy.optimize (Poldrack §2: named constants)
    param_bounds = [(0.01, 100.0)]

    @staticmethod
    def predict(params: np.ndarray, gamble_data: GambleData) -> np.ndarray:
        temperature = params[0]
        va = np.sum(gamble_data.probs_a * gamble_data.outcomes_a, axis=1)
        vb = np.sum(gamble_data.probs_b * gamble_data.outcomes_b, axis=1)
        return sigmoid(temperature * (vb - va))


class EUModel:
    """Expected Utility model. Power utility, no probability weighting.

    V(gamble) = Σ p_i · u(x_i)
    u(x) = sign(x)·|x|^α  (symmetric power utility)
    """

    name = "EU"
    n_params = 2  # alpha, temperature
    param_names = ["alpha", "temperature"]
    param_bounds = [(0.01, 2.0), (0.01, 100.0)]

    @staticmethod
    def predict(params: np.ndarray, gamble_data: GambleData) -> np.ndarray:
        alpha, temperature = params

        def utility(x):
            return np.sign(x) * np.abs(x) ** alpha

        va = np.sum(gamble_data.probs_a * utility(gamble_data.outcomes_a), axis=1)
        vb = np.sum(gamble_data.probs_b * utility(gamble_data.outcomes_b), axis=1)
        return sigmoid(temperature * (vb - va))


class PTModel:
    """Prospect Theory. Power utility + probability weighting.

    V(gamble) = Σ w(p_i) · u(x_i)
    u(x) = x^α for x≥0, -λ(-x)^α for x<0
    w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
    """

    name = "PT"
    n_params = 4  # alpha, lambda_, gamma, temperature
    param_names = ["alpha", "lambda_", "gamma", "temperature"]
    param_bounds = [(0.01, 2.0), (0.1, 10.0), (0.1, 2.0), (0.01, 100.0)]

    @staticmethod
    def predict(params: np.ndarray, gamble_data: GambleData) -> np.ndarray:
        alpha, lambda_, gamma, temperature = params

        def utility(x):
            pos = np.maximum(x, 0) ** alpha
            neg = -lambda_ * np.maximum(-x, 0) ** alpha
            return np.where(x >= 0, pos, neg)

        def weight(p):
            # Vectorized KT weighting, handling p=0 and p=1
            p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
            num = p_clipped ** gamma
            denom = (num + (1 - p_clipped) ** gamma) ** (1 / gamma)
            w = num / denom
            # Restore exact values at boundaries
            w = np.where(p <= 0, 0.0, w)
            w = np.where(p >= 1, 1.0, w)
            return w

        va = np.sum(weight(gamble_data.probs_a) * utility(gamble_data.outcomes_a), axis=1)
        vb = np.sum(weight(gamble_data.probs_b) * utility(gamble_data.outcomes_b), axis=1)
        return sigmoid(temperature * (vb - va))


class CPTModel:
    """Cumulative Prospect Theory (Tversky & Kahneman 1992).

    Like PT but applies probability weighting to cumulative probabilities,
    with separate weighting for gains and losses.

    Simplified vectorized version: uses the same KT weighting as PT
    but applied to cumulative probabilities from each tail.
    For the padded array format, we approximate CPT by:
    1. Sorting outcomes per gamble
    2. Computing cumulative weights from each direction
    3. Deriving decision weights as differences
    """

    name = "CPT"
    n_params = 5  # alpha, lambda_, gamma_pos, gamma_neg, temperature
    param_names = ["alpha", "lambda_", "gamma_pos", "gamma_neg", "temperature"]
    param_bounds = [
        (0.01, 2.0), (0.1, 10.0), (0.1, 2.0), (0.1, 2.0), (0.01, 100.0)
    ]

    @staticmethod
    def predict(params: np.ndarray, gamble_data: GambleData) -> np.ndarray:
        alpha, lambda_, gamma_pos, gamma_neg, temperature = params

        def utility(x):
            pos = np.maximum(x, 0) ** alpha
            neg = -lambda_ * np.maximum(-x, 0) ** alpha
            return np.where(x >= 0, pos, neg)

        def cpt_value_batch(outcomes, probs):
            """Vectorized CPT values for a batch of gambles."""
            n = outcomes.shape[0]
            values = np.zeros(n)
            # Pre-sort each row by outcome value
            order = np.argsort(outcomes, axis=1)
            x_sorted = np.take_along_axis(outcomes, order, axis=1)
            p_sorted = np.take_along_axis(probs, order, axis=1)

            u_vals = utility(x_sorted)
            active = p_sorted > 0  # mask out padding

            for i in range(n):
                mask = active[i]
                if not np.any(mask):
                    continue
                xs = x_sorted[i, mask]
                ps = p_sorted[i, mask]
                us = u_vals[i, mask]
                k = len(xs)

                # Gains: outcomes >= 0, cumulate from the top (best first)
                gain_mask = xs >= 0
                # Losses: outcomes < 0, cumulate from the bottom (worst first)
                loss_mask = xs < 0

                # Decision weights for gains (reverse cumulative)
                if np.any(gain_mask):
                    gp = ps[gain_mask][::-1]
                    gu = us[gain_mask][::-1]
                    gc = np.clip(np.cumsum(gp), 0, 1)
                    wc = _kt_weight(gc, gamma_pos)
                    dw = np.empty_like(wc)
                    dw[0] = wc[0]
                    dw[1:] = wc[1:] - wc[:-1]
                    values[i] += np.sum(dw * gu)

                # Decision weights for losses (forward cumulative)
                if np.any(loss_mask):
                    lp = ps[loss_mask]
                    lu = us[loss_mask]
                    lc = np.clip(np.cumsum(lp), 0, 1)
                    wc = _kt_weight(lc, gamma_neg)
                    dw = np.empty_like(wc)
                    dw[0] = wc[0]
                    dw[1:] = wc[1:] - wc[:-1]
                    values[i] += np.sum(dw * lu)

            return values

        va = cpt_value_batch(gamble_data.outcomes_a, gamble_data.probs_a)
        vb = cpt_value_batch(gamble_data.outcomes_b, gamble_data.probs_b)
        return sigmoid(temperature * (vb - va))


def _kt_weight(p: np.ndarray, gamma: float) -> np.ndarray:
    """Kahneman-Tversky weighting, vectorized."""
    p_clip = np.clip(p, 1e-10, 1 - 1e-10)
    num = p_clip ** gamma
    denom = (num + (1 - p_clip) ** gamma) ** (1 / gamma)
    w = num / denom
    w = np.where(p <= 0, 0.0, w)
    w = np.where(p >= 1, 1.0, w)
    return w


# Registry of all available models
ALL_MODELS = [EVModel, EUModel, PTModel, CPTModel]

MODEL_REGISTRY = {m.name: m for m in ALL_MODELS}
