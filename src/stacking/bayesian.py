"""Bayesian stacking with Dirichlet prior over weights.

Implements Yao et al. (2018) "Using Stacking to Average Bayesian
Predictive Distributions" with full posterior uncertainty.

Provides posterior distributions on stacking weights, answering
"how confident are we that CPT beats PT?" — not just point estimates.

Requires: pip install 'choices13k-stacking[bayesian]'
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Optional imports — fail gracefully if not installed
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    pm = None
    az = None


@dataclass
class BayesianStackingConfig:
    """Configuration for Bayesian stacking MCMC."""

    # Prior
    dirichlet_concentration: float = 1.0  # 1.0 = uniform on simplex

    # MCMC settings
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 4
    target_accept: float = 0.9
    random_seed: int = 42


@dataclass
class BayesianStackingResults:
    """Results from Bayesian stacking."""

    # Posterior samples (full ArviZ InferenceData)
    idata: "az.InferenceData"

    # Summary statistics
    weight_means: np.ndarray      # (n_models,) posterior means
    weight_stds: np.ndarray       # (n_models,) posterior stds
    weight_hdi_low: np.ndarray    # (n_models,) 94% HDI lower
    weight_hdi_high: np.ndarray   # (n_models,) 94% HDI upper

    # Model info
    model_names: list

    # Diagnostics
    r_hat_max: float              # Should be < 1.01
    ess_min: float                # Effective sample size (should be > 400)


def _check_pymc_available():
    """Raise helpful error if PyMC not installed."""
    if not HAS_PYMC:
        raise ImportError(
            "Bayesian stacking requires PyMC and ArviZ. "
            "Install with: pip install 'choices13k-stacking[bayesian]'"
        )


def build_stacking_model(
    oof_predictions: np.ndarray,
    observed_brate: np.ndarray,
    sample_sizes: np.ndarray,
    config: BayesianStackingConfig,
) -> "pm.Model":
    """Build PyMC model for Bayesian stacking.

    Model:
        weights ~ Dirichlet(alpha)
        mu = oof_predictions @ weights  (convex combination)
        y ~ Binomial(n, mu)

    Args:
        oof_predictions: (n_problems, n_models) out-of-fold predictions in [0,1]
        observed_brate: (n_problems,) observed choice proportions
        sample_sizes: (n_problems,) number of participants per problem
        config: Prior and MCMC settings

    Returns:
        PyMC model (not yet sampled)
    """
    _check_pymc_available()

    n_problems, n_models = oof_predictions.shape

    # Validate inputs (defensive programming per Poldrack §5)
    assert oof_predictions.shape[0] == len(observed_brate) == len(sample_sizes), (
        f"Shape mismatch: predictions {oof_predictions.shape[0]}, "
        f"brate {len(observed_brate)}, n {len(sample_sizes)}"
    )
    assert np.all((oof_predictions >= 0) & (oof_predictions <= 1)), (
        "OOF predictions must be probabilities in [0,1]"
    )
    assert np.all(sample_sizes > 0), "Sample sizes must be positive"

    # Convert bRate to counts for Binomial likelihood
    observed_counts = np.round(observed_brate * sample_sizes).astype(int)

    with pm.Model() as model:
        # Dirichlet prior on weights (uniform when alpha=1)
        concentration = np.ones(n_models) * config.dirichlet_concentration
        weights = pm.Dirichlet("weights", a=concentration)

        # Stacked prediction: convex combination
        # Shape: (n_problems,)
        mu = pm.math.dot(oof_predictions, weights)

        # Likelihood: Binomial
        # Each problem i: y_i ~ Binomial(n_i, mu_i)
        pm.Binomial(
            "y",
            n=sample_sizes,
            p=mu,
            observed=observed_counts,
        )

    return model


def run_bayesian_stacking(
    oof_predictions: np.ndarray,
    observed_brate: np.ndarray,
    sample_sizes: np.ndarray,
    model_names: list,
    config: Optional[BayesianStackingConfig] = None,
) -> BayesianStackingResults:
    """Run Bayesian stacking and return posterior results.

    This is the main entry point for Bayesian stacking.

    Args:
        oof_predictions: (n_problems, n_models) out-of-fold predictions
        observed_brate: (n_problems,) observed choice proportions
        sample_sizes: (n_problems,) from 'n' column in data
        model_names: list of model names for labeling
        config: MCMC settings (uses defaults if None)

    Returns:
        BayesianStackingResults with posterior samples and summaries
    """
    _check_pymc_available()

    if config is None:
        config = BayesianStackingConfig()

    # Build model
    model = build_stacking_model(
        oof_predictions, observed_brate, sample_sizes, config
    )

    # Sample
    with model:
        idata = pm.sample(
            draws=config.n_samples,
            tune=config.n_tune,
            chains=config.n_chains,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    # Extract posterior summaries
    weights_posterior = idata.posterior["weights"].values  # (chains, draws, n_models)
    weights_flat = weights_posterior.reshape(-1, weights_posterior.shape[-1])

    weight_means = weights_flat.mean(axis=0)
    weight_stds = weights_flat.std(axis=0)

    # HDI (94% by convention in ArviZ)
    hdi = az.hdi(idata, var_names=["weights"], hdi_prob=0.94)
    weight_hdi_low = hdi["weights"].values[:, 0]
    weight_hdi_high = hdi["weights"].values[:, 1]

    # Diagnostics
    summary = az.summary(idata, var_names=["weights"])
    r_hat_max = summary["r_hat"].max()
    ess_min = summary["ess_bulk"].min()

    return BayesianStackingResults(
        idata=idata,
        weight_means=weight_means,
        weight_stds=weight_stds,
        weight_hdi_low=weight_hdi_low,
        weight_hdi_high=weight_hdi_high,
        model_names=list(model_names),
        r_hat_max=r_hat_max,
        ess_min=ess_min,
    )


def print_bayesian_results(results: BayesianStackingResults) -> None:
    """Pretty-print Bayesian stacking results."""
    print("\n" + "=" * 60)
    print("BAYESIAN STACKING RESULTS")
    print("=" * 60)

    print("\nPosterior weight distributions (94% HDI):")
    for i, name in enumerate(results.model_names):
        mean = results.weight_means[i]
        std = results.weight_stds[i]
        lo = results.weight_hdi_low[i]
        hi = results.weight_hdi_high[i]
        bar_len = int(mean * 40)
        bar = "|" + "=" * bar_len + " " * (40 - bar_len) + "|"
        print(f"  {name:>6s}: {mean:.3f} +/- {std:.3f}  [{lo:.3f}, {hi:.3f}]  {bar}")

    print(f"\nDiagnostics:")
    print(f"  Max R-hat: {results.r_hat_max:.4f} (should be < 1.01)")
    print(f"  Min ESS:   {results.ess_min:.0f} (should be > 400)")

    if results.r_hat_max > 1.01:
        print("  WARNING: R-hat too high - chains may not have converged")
    if results.ess_min < 400:
        print("  WARNING: ESS too low - consider more samples")

    # Check if any model is clearly dominant
    max_idx = np.argmax(results.weight_means)
    if results.weight_hdi_low[max_idx] > 0.5:
        print(f"\n  {results.model_names[max_idx]} clearly dominates "
              f"(HDI excludes 0.5)")
