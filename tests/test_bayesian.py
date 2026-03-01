"""Tests for Bayesian stacking module.

TDD (Poldrack §4): tests verify behavior against ground truth.
Skip tests if PyMC not installed.
"""

import numpy as np
import pytest

# Skip all tests in this module if PyMC not available
pymc = pytest.importorskip("pymc")


class TestBayesianStackingWeights:
    """Test posterior properties."""

    def test_weights_sum_to_one(self):
        """Posterior samples must live on simplex."""
        from stacking.bayesian import run_bayesian_stacking, BayesianStackingConfig

        rng = np.random.RandomState(42)
        n, k = 100, 3
        preds = rng.rand(n, k)
        # Normalize to valid probabilities
        preds = np.clip(preds, 0.01, 0.99)
        targets = rng.rand(n)
        sample_sizes = rng.randint(20, 30, size=n)

        config = BayesianStackingConfig(
            n_samples=200, n_tune=100, n_chains=2  # fast for testing
        )
        results = run_bayesian_stacking(
            preds, targets, sample_sizes, ["M1", "M2", "M3"], config
        )

        # All posterior samples should sum to 1
        weights_posterior = results.idata.posterior["weights"].values
        sums = weights_posterior.sum(axis=-1)  # sum over models
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_weights_nonnegative(self):
        """Dirichlet samples are always non-negative."""
        from stacking.bayesian import run_bayesian_stacking, BayesianStackingConfig

        rng = np.random.RandomState(42)
        n, k = 100, 3
        preds = np.clip(rng.rand(n, k), 0.01, 0.99)
        targets = rng.rand(n)
        sample_sizes = rng.randint(20, 30, size=n)

        config = BayesianStackingConfig(n_samples=200, n_tune=100, n_chains=2)
        results = run_bayesian_stacking(
            preds, targets, sample_sizes, ["M1", "M2", "M3"], config
        )

        weights_posterior = results.idata.posterior["weights"].values
        assert np.all(weights_posterior >= 0)

    def test_perfect_model_gets_high_weight(self):
        """If one model perfectly predicts, it should dominate posterior."""
        from stacking.bayesian import run_bayesian_stacking, BayesianStackingConfig

        rng = np.random.RandomState(42)
        n = 200
        targets = np.clip(rng.rand(n), 0.1, 0.9)  # avoid boundaries
        sample_sizes = np.full(n, 25)

        preds = np.column_stack([
            targets,                        # perfect model
            np.ones(n) * 0.5,               # constant model
            np.clip(rng.rand(n), 0.1, 0.9), # noise model
        ])

        config = BayesianStackingConfig(n_samples=500, n_tune=200, n_chains=2)
        results = run_bayesian_stacking(
            preds, targets, sample_sizes, ["Perfect", "Constant", "Noise"], config
        )

        # Perfect model should get most posterior mass
        assert results.weight_means[0] > 0.7, (
            f"Perfect model should dominate, got {results.weight_means}"
        )


class TestBayesianStackingDiagnostics:
    """Test MCMC convergence diagnostics."""

    def test_r_hat_computed(self):
        """Should report R-hat diagnostic."""
        from stacking.bayesian import run_bayesian_stacking, BayesianStackingConfig

        rng = np.random.RandomState(42)
        n, k = 100, 2
        preds = np.clip(rng.rand(n, k), 0.01, 0.99)
        targets = rng.rand(n)
        sample_sizes = rng.randint(20, 30, size=n)

        config = BayesianStackingConfig(n_samples=200, n_tune=100, n_chains=2)
        results = run_bayesian_stacking(
            preds, targets, sample_sizes, ["M1", "M2"], config
        )

        assert hasattr(results, "r_hat_max")
        assert results.r_hat_max > 0  # some value computed

    def test_ess_computed(self):
        """Should report effective sample size."""
        from stacking.bayesian import run_bayesian_stacking, BayesianStackingConfig

        rng = np.random.RandomState(42)
        n, k = 100, 2
        preds = np.clip(rng.rand(n, k), 0.01, 0.99)
        targets = rng.rand(n)
        sample_sizes = rng.randint(20, 30, size=n)

        config = BayesianStackingConfig(n_samples=200, n_tune=100, n_chains=2)
        results = run_bayesian_stacking(
            preds, targets, sample_sizes, ["M1", "M2"], config
        )

        assert hasattr(results, "ess_min")
        assert results.ess_min > 0


class TestBayesianStackingValidation:
    """Test input validation."""

    def test_rejects_invalid_predictions(self):
        """OOF predictions must be in [0,1]."""
        from stacking.bayesian import build_stacking_model, BayesianStackingConfig

        preds = np.array([[1.5, 0.3], [0.2, 0.8]])  # 1.5 is invalid
        targets = np.array([0.5, 0.5])
        sample_sizes = np.array([25, 25])

        with pytest.raises(AssertionError, match="probabilities"):
            build_stacking_model(
                preds, targets, sample_sizes, BayesianStackingConfig()
            )

    def test_rejects_nonpositive_sample_sizes(self):
        """Sample sizes must be positive."""
        from stacking.bayesian import build_stacking_model, BayesianStackingConfig

        preds = np.array([[0.5, 0.3], [0.2, 0.8]])
        targets = np.array([0.5, 0.5])
        sample_sizes = np.array([25, 0])  # 0 is invalid

        with pytest.raises(AssertionError, match="positive"):
            build_stacking_model(
                preds, targets, sample_sizes, BayesianStackingConfig()
            )


class TestPyMCMissing:
    """Test graceful failure when PyMC not installed."""

    def test_helpful_error_message(self, monkeypatch):
        """Should give install instructions when PyMC missing."""
        import stacking.bayesian as bayesian_module

        # Simulate PyMC not being available
        monkeypatch.setattr(bayesian_module, "HAS_PYMC", False)

        with pytest.raises(ImportError, match="pip install"):
            bayesian_module._check_pymc_available()
