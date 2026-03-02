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
    if not (oof_predictions.shape[0] == len(observed_brate) == len(sample_sizes)):
        raise ValueError(
            f"Shape mismatch: predictions {oof_predictions.shape[0]}, "
            f"brate {len(observed_brate)}, n {len(sample_sizes)}"
        )
    if not np.all((oof_predictions >= 0) & (oof_predictions <= 1)):
        raise ValueError(
            "OOF predictions must be probabilities in [0,1]"
        )
    if not np.all(sample_sizes > 0):
        raise ValueError("Sample sizes must be positive")

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


# =============================================================================
# HIERARCHICAL BAYESIAN STACKING
# Weights vary by problem features: w_k(x) = softmax(intercept + X @ beta)
# =============================================================================


@dataclass
class HierarchicalBayesianConfig:
    """Configuration for hierarchical Bayesian stacking."""

    # MCMC settings
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 4
    target_accept: float = 0.95  # higher for hierarchical models
    random_seed: int = 42

    # Prior settings
    intercept_sd: float = 2.0      # prior SD on intercepts
    beta_sd_prior: float = 1.0     # prior on sigma_beta (hierarchical shrinkage)


@dataclass
class HierarchicalBayesianResults:
    """Results from hierarchical Bayesian stacking."""

    # Posterior samples
    idata: "az.InferenceData"

    # Baseline weights (intercept only, averaged over posterior)
    baseline_weight_means: np.ndarray   # (n_models,)
    baseline_weight_stds: np.ndarray    # (n_models,)

    # Beta coefficients: how features shift weights
    beta_means: np.ndarray              # (n_features, n_models-1)
    beta_stds: np.ndarray               # (n_features, n_models-1)

    # Per-problem weights (posterior means)
    weights_per_problem: np.ndarray     # (n_problems, n_models)

    # Model info
    model_names: list
    feature_names: list

    # Diagnostics
    r_hat_max: float
    ess_min: float


def build_hierarchical_model(
    oof_predictions: np.ndarray,
    observed_brate: np.ndarray,
    sample_sizes: np.ndarray,
    features: np.ndarray,
    config: HierarchicalBayesianConfig,
) -> "pm.Model":
    """Build PyMC model for hierarchical Bayesian stacking.

    Model:
        sigma_beta ~ HalfNormal(beta_sd_prior)
        beta ~ Normal(0, sigma_beta)           # (n_features, n_models-1)
        intercept ~ Normal(0, intercept_sd)    # (n_models-1)
        logits = intercept + features @ beta   # last model is reference
        weights = softmax(logits)              # (n_problems, n_models)
        mu = sum(oof_predictions * weights)
        y ~ Binomial(n, mu)

    Args:
        oof_predictions: (n_problems, n_models) out-of-fold predictions
        observed_brate: (n_problems,) observed choice proportions
        sample_sizes: (n_problems,) number of participants
        features: (n_problems, n_features) standardized problem features
        config: Prior and MCMC settings

    Returns:
        PyMC model (not yet sampled)
    """
    _check_pymc_available()

    n_problems, n_models = oof_predictions.shape
    n_features = features.shape[1]

    # Validate inputs
    if not (oof_predictions.shape[0] == len(observed_brate) == len(sample_sizes)):
        raise ValueError("Shape mismatch in inputs")
    if features.shape[0] != n_problems:
        raise ValueError(
            f"Features have {features.shape[0]} rows, expected {n_problems}"
        )
    if not np.all((oof_predictions >= 0) & (oof_predictions <= 1)):
        raise ValueError("OOF predictions must be probabilities in [0,1]")
    if not np.all(sample_sizes > 0):
        raise ValueError("Sample sizes must be positive")

    observed_counts = np.round(observed_brate * sample_sizes).astype(int)

    with pm.Model() as model:
        # Hierarchical shrinkage prior on beta
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=config.beta_sd_prior)

        # Beta coefficients: (n_features, n_models-1)
        # Last model is reference (beta = 0)
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=sigma_beta,
            shape=(n_features, n_models - 1),
        )

        # Intercepts: (n_models-1)
        intercept = pm.Normal(
            "intercept",
            mu=0,
            sigma=config.intercept_sd,
            shape=(n_models - 1,),
        )

        # Compute logits: (n_problems, n_models)
        # Last column is 0 (reference model)
        logits_free = intercept + pm.math.dot(features, beta)  # (n, K-1)
        logits = pm.math.concatenate(
            [logits_free, pm.math.zeros((n_problems, 1))],
            axis=1,
        )

        # Softmax to get weights per problem
        weights = pm.math.softmax(logits, axis=1)

        # Stacked prediction
        mu = pm.math.sum(oof_predictions * weights, axis=1)

        # Likelihood
        pm.Binomial("y", n=sample_sizes, p=mu, observed=observed_counts)

        # Track baseline weights (intercept only, no features)
        baseline_logits = pm.math.concatenate([intercept, pm.math.zeros((1,))])
        pm.Deterministic(
            "baseline_weights",
            pm.math.softmax(baseline_logits),
        )

    return model


def run_hierarchical_bayesian(
    oof_predictions: np.ndarray,
    observed_brate: np.ndarray,
    sample_sizes: np.ndarray,
    features: np.ndarray,
    model_names: list,
    feature_names: list,
    config: Optional[HierarchicalBayesianConfig] = None,
) -> HierarchicalBayesianResults:
    """Run hierarchical Bayesian stacking.

    Args:
        oof_predictions: (n_problems, n_models) out-of-fold predictions
        observed_brate: (n_problems,) observed choice proportions
        sample_sizes: (n_problems,) from 'n' column
        features: (n_problems, n_features) STANDARDIZED problem features
        model_names: list of model names
        feature_names: list of feature names
        config: MCMC settings

    Returns:
        HierarchicalBayesianResults with posteriors on beta, intercept, weights
    """
    _check_pymc_available()

    if config is None:
        config = HierarchicalBayesianConfig()

    n_problems, n_models = oof_predictions.shape
    n_features = features.shape[1]

    # Build and sample
    model = build_hierarchical_model(
        oof_predictions, observed_brate, sample_sizes, features, config
    )

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

    # Extract beta posteriors
    beta_posterior = idata.posterior["beta"].values  # (chains, draws, n_feat, n_models-1)
    beta_flat = beta_posterior.reshape(-1, n_features, n_models - 1)
    beta_means = beta_flat.mean(axis=0)
    beta_stds = beta_flat.std(axis=0)

    # Extract intercept posteriors
    intercept_posterior = idata.posterior["intercept"].values  # (chains, draws, n_models-1)
    intercept_flat = intercept_posterior.reshape(-1, n_models - 1)

    # Compute baseline weights from intercept posterior
    baseline_logits = np.concatenate(
        [intercept_flat, np.zeros((intercept_flat.shape[0], 1))],
        axis=1,
    )
    baseline_weights_samples = _softmax_numpy(baseline_logits)
    baseline_weight_means = baseline_weights_samples.mean(axis=0)
    baseline_weight_stds = baseline_weights_samples.std(axis=0)

    # Compute per-problem weights using posterior mean of beta and intercept
    intercept_mean = intercept_flat.mean(axis=0)
    logits_mean = np.concatenate([
        intercept_mean + features @ beta_means,
        np.zeros((n_problems, 1)),
    ], axis=1)
    weights_per_problem = _softmax_numpy(logits_mean)

    # Diagnostics
    summary = az.summary(idata, var_names=["intercept", "sigma_beta"])
    r_hat_max = summary["r_hat"].max()
    ess_min = summary["ess_bulk"].min()

    return HierarchicalBayesianResults(
        idata=idata,
        baseline_weight_means=baseline_weight_means,
        baseline_weight_stds=baseline_weight_stds,
        beta_means=beta_means,
        beta_stds=beta_stds,
        weights_per_problem=weights_per_problem,
        model_names=list(model_names),
        feature_names=list(feature_names),
        r_hat_max=r_hat_max,
        ess_min=ess_min,
    )


def _softmax_numpy(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# =============================================================================
# MIXTURE OF THEORIES (MOT)
# Interprets weights as "proportion of participants using each theory"
# =============================================================================


@dataclass
class MOTConfig:
    """Configuration for Mixture of Theories model."""

    # MCMC settings
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 4
    target_accept: float = 0.9
    random_seed: int = 42

    # Model variant
    use_overdispersion: bool = True  # Beta-Binomial vs Binomial
    kappa_prior_sd: float = 10.0     # Prior on overdispersion


@dataclass
class MOTResults:
    """Results from Mixture of Theories model."""

    # Posterior samples
    idata: "az.InferenceData"

    # Mixture proportions (what % use each theory)
    pi_means: np.ndarray       # (n_models,)
    pi_stds: np.ndarray        # (n_models,)
    pi_hdi_low: np.ndarray     # (n_models,)
    pi_hdi_high: np.ndarray    # (n_models,)

    # Overdispersion (if used)
    kappa_mean: Optional[float]
    kappa_std: Optional[float]

    # Model info
    model_names: list

    # Diagnostics
    r_hat_max: float
    ess_min: float


def build_mot_model(
    oof_predictions: np.ndarray,
    observed_brate: np.ndarray,
    sample_sizes: np.ndarray,
    config: MOTConfig,
) -> "pm.Model":
    """Build Mixture of Theories model.

    Generative story:
        For each problem i with n_i participants:
        - Each participant j uses theory k with probability π_k
        - Given theory k, they choose B with probability pred_{i,k}
        - Marginalizing: P(B) = Σ_k π_k * pred_{i,k}
        - Observed: y_i ~ Binomial(n_i, P(B)) or Beta-Binomial with overdispersion

    The overdispersion κ captures individual heterogeneity beyond the mixture.

    Args:
        oof_predictions: (n_problems, n_models) predictions from each theory
        observed_brate: (n_problems,) observed choice proportions
        sample_sizes: (n_problems,) participants per problem
        config: Model settings

    Returns:
        PyMC model
    """
    _check_pymc_available()

    n_problems, n_models = oof_predictions.shape
    observed_counts = np.round(observed_brate * sample_sizes).astype(int)

    # Validate
    if not np.all((oof_predictions >= 0) & (oof_predictions <= 1)):
        raise ValueError("Predictions must be probabilities in [0,1]")
    if not np.all(sample_sizes > 0):
        raise ValueError("Sample sizes must be positive")

    with pm.Model() as model:
        # Mixture proportions: what fraction use each theory
        pi = pm.Dirichlet("pi", a=np.ones(n_models))

        # Expected choice probability under mixture
        # P(B) = Σ_k π_k * pred_{i,k}
        mu = pm.math.dot(oof_predictions, pi)

        if config.use_overdispersion:
            # Beta-Binomial: extra variance from individual differences
            # Var(y) > n*p*(1-p) when people disagree more than binomial predicts
            kappa = pm.HalfNormal("kappa", sigma=config.kappa_prior_sd)

            # Reparameterize Beta in terms of mean and concentration
            # Beta(α, β) where α = μ*κ, β = (1-μ)*κ
            # Higher κ = less overdispersion (approaches Binomial)
            alpha = mu * kappa
            beta = (1 - mu) * kappa

            pm.BetaBinomial(
                "y",
                alpha=alpha,
                beta=beta,
                n=sample_sizes,
                observed=observed_counts,
            )
        else:
            # Standard Binomial (no overdispersion)
            pm.Binomial("y", n=sample_sizes, p=mu, observed=observed_counts)

    return model


def run_mot(
    oof_predictions: np.ndarray,
    observed_brate: np.ndarray,
    sample_sizes: np.ndarray,
    model_names: list,
    config: Optional[MOTConfig] = None,
) -> MOTResults:
    """Run Mixture of Theories model.

    Args:
        oof_predictions: (n_problems, n_models) predictions
        observed_brate: (n_problems,) observed proportions
        sample_sizes: (n_problems,) participants per problem
        model_names: names of theories
        config: MCMC settings

    Returns:
        MOTResults with posterior on mixture proportions
    """
    _check_pymc_available()

    if config is None:
        config = MOTConfig()

    model = build_mot_model(
        oof_predictions, observed_brate, sample_sizes, config
    )

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

    # Extract pi posteriors
    pi_posterior = idata.posterior["pi"].values
    pi_flat = pi_posterior.reshape(-1, pi_posterior.shape[-1])

    pi_means = pi_flat.mean(axis=0)
    pi_stds = pi_flat.std(axis=0)

    hdi = az.hdi(idata, var_names=["pi"], hdi_prob=0.94)
    pi_hdi_low = hdi["pi"].values[:, 0]
    pi_hdi_high = hdi["pi"].values[:, 1]

    # Extract kappa if used
    kappa_mean = None
    kappa_std = None
    if config.use_overdispersion:
        kappa_posterior = idata.posterior["kappa"].values.flatten()
        kappa_mean = kappa_posterior.mean()
        kappa_std = kappa_posterior.std()

    # Diagnostics
    var_names = ["pi"] + (["kappa"] if config.use_overdispersion else [])
    summary = az.summary(idata, var_names=var_names)
    r_hat_max = summary["r_hat"].max()
    ess_min = summary["ess_bulk"].min()

    return MOTResults(
        idata=idata,
        pi_means=pi_means,
        pi_stds=pi_stds,
        pi_hdi_low=pi_hdi_low,
        pi_hdi_high=pi_hdi_high,
        kappa_mean=kappa_mean,
        kappa_std=kappa_std,
        model_names=list(model_names),
        r_hat_max=r_hat_max,
        ess_min=ess_min,
    )


def print_mot_results(results: MOTResults) -> None:
    """Pretty-print MOT results."""
    print("\n" + "=" * 60)
    print("MIXTURE OF THEORIES RESULTS")
    print("=" * 60)

    print("\nTheory usage proportions (what % of participants use each):")
    for i, name in enumerate(results.model_names):
        mean = results.pi_means[i] * 100
        std = results.pi_stds[i] * 100
        lo = results.pi_hdi_low[i] * 100
        hi = results.pi_hdi_high[i] * 100
        bar_len = int(mean / 100 * 40)
        bar = "|" + "=" * bar_len + " " * (40 - bar_len) + "|"
        print(f"  {name:>6s}: {mean:.1f}% +/- {std:.1f}%  [{lo:.1f}%, {hi:.1f}%]  {bar}")

    if results.kappa_mean is not None:
        print(f"\nOverdispersion (kappa):")
        print(f"  kappa = {results.kappa_mean:.2f} +/- {results.kappa_std:.2f}")
        print(f"  (Higher = less individual variation beyond mixture)")
        print(f"  (Lower = more heterogeneity, people differ beyond theory choice)")

    print(f"\nDiagnostics:")
    print(f"  Max R-hat: {results.r_hat_max:.4f} (should be < 1.01)")
    print(f"  Min ESS:   {results.ess_min:.0f} (should be > 400)")

    # Interpretation
    max_idx = np.argmax(results.pi_means)
    print(f"\nInterpretation:")
    print(f"  {results.pi_means[max_idx]*100:.0f}% of participants use "
          f"{results.model_names[max_idx]} as their decision rule")


def print_hierarchical_bayesian_results(
    results: HierarchicalBayesianResults,
) -> None:
    """Pretty-print hierarchical Bayesian stacking results."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL BAYESIAN STACKING RESULTS")
    print("=" * 60)

    # Baseline weights
    print("\nBaseline weights (intercept only, no features):")
    for i, name in enumerate(results.model_names):
        mean = results.baseline_weight_means[i]
        std = results.baseline_weight_stds[i]
        bar_len = int(mean * 40)
        bar = "|" + "=" * bar_len + " " * (40 - bar_len) + "|"
        print(f"  {name:>6s}: {mean:.3f} +/- {std:.3f}  {bar}")

    # Feature effects (beta coefficients)
    print(f"\nFeature effects on weights (vs {results.model_names[-1]}):")
    print("  (positive = increases weight, negative = decreases)")
    for j, fname in enumerate(results.feature_names):
        effects = []
        for k, mname in enumerate(results.model_names[:-1]):
            mean = results.beta_means[j, k]
            std = results.beta_stds[j, k]
            # Check if credibly non-zero (HDI excludes 0)
            sig = "*" if abs(mean) > 2 * std else ""
            effects.append(f"{mname}={mean:+.3f}{sig}")
        print(f"  {fname:>20s}: {', '.join(effects)}")

    # Weight variation
    weights = results.weights_per_problem
    print(f"\nWeight variation across {weights.shape[0]} problems:")
    for i, name in enumerate(results.model_names):
        w = weights[:, i]
        print(
            f"  {name:>6s}: mean={w.mean():.3f}, "
            f"range=[{w.min():.3f}, {w.max():.3f}], "
            f"std={w.std():.3f}"
        )

    # Diagnostics
    print(f"\nDiagnostics:")
    print(f"  Max R-hat: {results.r_hat_max:.4f} (should be < 1.01)")
    print(f"  Min ESS:   {results.ess_min:.0f} (should be > 400)")

    if results.r_hat_max > 1.01:
        print("  WARNING: R-hat too high - chains may not have converged")
    if results.ess_min < 400:
        print("  WARNING: ESS too low - consider more samples")
