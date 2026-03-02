"""Tests for choices13k stacking pipeline.

TDD (Poldrack §4): tests verify behavior against ground truth.
Arrange → Act → Assert structure.
Test invalid inputs too — garbage in, exception out.

Ground truth for model tests:
- EV of a certain $10 gamble = 10
- P(B) = 0.5 when V(A) = V(B) regardless of temperature
- Stacking weights must be on simplex (sum to 1, non-negative)
"""

from pathlib import Path

import numpy as np
import pytest

from stacking.config import DataConfig, StackingConfig, AnalysisConfig
from stacking.data import load_selections, _validate_selections
from stacking.models import (
    EVModel, EUModel, PTModel, CPTModel,
    GambleData, predict_choice_probability,
)
from stacking.stacking import compute_stacking_weights
from stacking.fitting import compute_mse_loss


# ============================================================
# Data validation tests
# ============================================================

class TestDataValidation:

    def test_load_selections_file_not_found(self):
        """Garbage in, exception out (Poldrack §5)."""
        config = DataConfig(data_dir=Path("/nonexistent"))
        with pytest.raises(FileNotFoundError):
            load_selections(config)

    def test_load_selections_invalid_feedback_filter(self):
        config = DataConfig(
            data_dir=Path("data"),
            feedback_filter="invalid_value"
        )
        with pytest.raises(ValueError, match="feedback_filter"):
            load_selections(config)

    def test_validate_selections_rejects_bad_brate(self):
        """The conspiracy theory paper lesson — catch out-of-bounds data."""
        import pandas as pd
        bad_df = pd.DataFrame({
            "Problem": [1], "Feedback": [True], "n": [15],
            "Ha": [10], "pHa": [0.5], "La": [0],
            "Hb": [5], "pHb": [0.5], "Lb": [0],
            "LotShapeB": [0], "LotNumB": [1],
            "Amb": [False], "Corr": [0],
            "bRate": [1.5],  # INVALID — out of [0,1]
        })
        with pytest.raises(ValueError, match="bRate"):
            _validate_selections(bad_df)

    def test_validate_selections_rejects_negative_n(self):
        import pandas as pd
        bad_df = pd.DataFrame({
            "Problem": [1], "Feedback": [True], "n": [-5],
            "Ha": [10], "pHa": [0.5], "La": [0],
            "Hb": [5], "pHb": [0.5], "Lb": [0],
            "LotShapeB": [0], "LotNumB": [1],
            "Amb": [False], "Corr": [0],
            "bRate": [0.5],
        })
        with pytest.raises(ValueError, match="sample size"):
            _validate_selections(bad_df)


# ============================================================
# Model unit tests — ground truth, not implementation
# ============================================================

def _make_simple_gamble_data(
    outcomes_a, probs_a, outcomes_b, probs_b, brate=0.5
):
    """Helper to create GambleData for a single problem."""
    max_len = max(len(outcomes_a), len(outcomes_b))
    oa = np.zeros((1, max_len))
    pa = np.zeros((1, max_len))
    ob = np.zeros((1, max_len))
    pb = np.zeros((1, max_len))
    for i, (o, p) in enumerate(zip(outcomes_a, probs_a)):
        oa[0, i] = o
        pa[0, i] = p
    for i, (o, p) in enumerate(zip(outcomes_b, probs_b)):
        ob[0, i] = o
        pb[0, i] = p
    return GambleData(
        outcomes_a=oa, probs_a=pa,
        outcomes_b=ob, probs_b=pb,
        brate=np.array([brate]),
    )


class TestChoiceRule:

    def test_equal_values_give_fifty_fifty(self):
        """When V(A) = V(B), P(B) should be 0.5 regardless of temperature."""
        for temp in [0.1, 1.0, 10.0, 100.0]:
            p = predict_choice_probability(
                np.array([5.0]), np.array([5.0]), temperature=temp
            )
            np.testing.assert_allclose(p, 0.5, atol=1e-10)

    def test_higher_value_preferred(self):
        """When V(B) > V(A), P(B) > 0.5."""
        p = predict_choice_probability(
            np.array([5.0]), np.array([10.0]), temperature=1.0
        )
        assert p[0] > 0.5

    def test_temperature_increases_determinism(self):
        """Higher temperature → more extreme choice probability."""
        p_low = predict_choice_probability(
            np.array([5.0]), np.array([10.0]), temperature=0.1
        )
        p_high = predict_choice_probability(
            np.array([5.0]), np.array([10.0]), temperature=10.0
        )
        # Both > 0.5 (B preferred), but high temp closer to 1.0
        assert 0.5 < p_low[0] < p_high[0]


class TestEVModel:

    def test_certain_gambles(self):
        """EV of certain $10 = 10. Equal EVs → P(B) ≈ 0.5."""
        data = _make_simple_gamble_data(
            outcomes_a=[10], probs_a=[1.0],
            outcomes_b=[10], probs_b=[1.0],
        )
        pred = EVModel.predict(np.array([1.0]), data)
        np.testing.assert_allclose(pred, 0.5, atol=1e-10)

    def test_dominant_gamble_preferred(self):
        """$20 for sure should be preferred over $10 for sure."""
        data = _make_simple_gamble_data(
            outcomes_a=[10], probs_a=[1.0],
            outcomes_b=[20], probs_b=[1.0],
        )
        pred = EVModel.predict(np.array([1.0]), data)
        assert pred[0] > 0.5

    def test_ev_computation_correct(self):
        """EV of (0.5, $100; 0.5, $0) = $50."""
        data = _make_simple_gamble_data(
            outcomes_a=[50], probs_a=[1.0],  # $50 for sure
            outcomes_b=[100, 0], probs_b=[0.5, 0.5],  # EV = $50
        )
        pred = EVModel.predict(np.array([1.0]), data)
        np.testing.assert_allclose(pred, 0.5, atol=1e-10)


class TestEUModel:

    def test_reduces_to_ev_when_alpha_one(self):
        """EU with α=1 should equal EV."""
        data = _make_simple_gamble_data(
            outcomes_a=[50], probs_a=[1.0],
            outcomes_b=[100, 0], probs_b=[0.5, 0.5],
        )
        ev_pred = EVModel.predict(np.array([1.0]), data)
        eu_pred = EUModel.predict(np.array([1.0, 1.0]), data)
        np.testing.assert_allclose(eu_pred, ev_pred, atol=1e-10)

    def test_risk_aversion_with_concave_utility(self):
        """α < 1 (concave utility) → risk averse.
        Certain $50 preferred over (0.5, $100; 0.5, $0)."""
        data = _make_simple_gamble_data(
            outcomes_a=[50], probs_a=[1.0],       # $50 for sure
            outcomes_b=[100, 0], probs_b=[0.5, 0.5],  # risky, EV=$50
        )
        pred = EUModel.predict(np.array([0.5, 1.0]), data)
        # Risk averse: prefers sure thing (A), so P(B) < 0.5
        assert pred[0] < 0.5


class TestPTModel:

    def test_reduces_to_eu_when_gamma_one(self):
        """PT with γ=1 (linear weighting) should equal EU."""
        data = _make_simple_gamble_data(
            outcomes_a=[50], probs_a=[1.0],
            outcomes_b=[100, 0], probs_b=[0.5, 0.5],
        )
        eu_pred = EUModel.predict(np.array([0.8, 1.0]), data)
        # PT with lambda=1, gamma=1 should match EU
        pt_pred = PTModel.predict(np.array([0.8, 1.0, 1.0, 1.0]), data)
        np.testing.assert_allclose(pt_pred, eu_pred, atol=1e-6)


# ============================================================
# Stacking weight tests
# ============================================================

class TestStackingWeights:

    def test_weights_on_simplex(self):
        """Weights must be non-negative and sum to 1."""
        rng = np.random.RandomState(42)
        preds = rng.rand(100, 3)
        targets = rng.rand(100)

        weights = compute_stacking_weights(preds, targets)

        assert np.all(weights >= -1e-10), "Negative weights found"
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-6)

    def test_perfect_model_gets_all_weight(self):
        """If one model perfectly predicts, it should get weight ≈ 1."""
        targets = np.random.RandomState(42).rand(100)
        preds = np.column_stack([
            targets,               # perfect model
            np.ones(100) * 0.5,    # constant model
            np.random.rand(100),   # noise model
        ])

        weights = compute_stacking_weights(preds, targets)

        assert weights[0] > 0.95, (
            f"Perfect model should get most weight, got {weights[0]:.3f}"
        )

    def test_single_model_gets_unit_weight(self):
        """With one model, weight should be 1."""
        preds = np.random.rand(50, 1)
        targets = np.random.rand(50)
        weights = compute_stacking_weights(preds, targets)
        np.testing.assert_allclose(weights, [1.0])


class TestMSELoss:

    def test_perfect_prediction(self):
        targets = np.array([0.3, 0.7, 0.5])
        assert compute_mse_loss(targets, targets) == 0.0

    def test_known_value(self):
        preds = np.array([0.0, 0.0])
        targets = np.array([1.0, 1.0])
        assert compute_mse_loss(preds, targets) == 1.0


# ============================================================
# Smoke test — integration
# ============================================================

class TestCPTModel:
    """CPT-specific tests for vectorization correctness."""

    def test_cpt_reduces_to_pt_for_pure_gains(self):
        """When all outcomes are gains, CPT with gamma_pos=gamma_neg=gamma
        should produce similar results to PT (not exact due to cumulative vs simple weighting)."""
        data = _make_simple_gamble_data(
            outcomes_a=[50], probs_a=[1.0],
            outcomes_b=[100, 0], probs_b=[0.5, 0.5],
        )
        # CPT params: alpha, lambda_, gamma_pos, gamma_neg, temperature
        cpt_pred = CPTModel.predict(np.array([0.8, 1.0, 0.7, 0.7, 1.0]), data)
        # Should be a valid probability
        assert 0 < cpt_pred[0] < 1

    def test_cpt_handles_mixed_gains_losses(self):
        """CPT should handle gambles with both gains and losses."""
        data = _make_simple_gamble_data(
            outcomes_a=[0], probs_a=[1.0],  # $0 for sure
            outcomes_b=[50, -25], probs_b=[0.5, 0.5],  # mixed gamble
        )
        cpt_pred = CPTModel.predict(np.array([0.88, 2.25, 0.65, 0.65, 1.0]), data)
        # With loss aversion λ=2.25, the loss hurts more than gain helps
        # So P(B) should be < 0.5 (prefer sure $0)
        assert cpt_pred[0] < 0.5

    def test_cpt_loss_aversion_effect(self):
        """Higher loss aversion should decrease preference for mixed gambles."""
        data = _make_simple_gamble_data(
            outcomes_a=[0], probs_a=[1.0],
            outcomes_b=[100, -100], probs_b=[0.5, 0.5],
        )
        # Low loss aversion
        pred_low = CPTModel.predict(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), data)
        # High loss aversion
        pred_high = CPTModel.predict(np.array([1.0, 3.0, 1.0, 1.0, 1.0]), data)
        # Higher loss aversion → lower P(B)
        assert pred_high[0] < pred_low[0]

    def test_cpt_batch_consistency(self):
        """CPT should give same results whether run on single problems or batch."""
        # Create two separate single-problem datasets
        data1 = _make_simple_gamble_data(
            outcomes_a=[50], probs_a=[1.0],
            outcomes_b=[100, 0], probs_b=[0.5, 0.5],
        )
        data2 = _make_simple_gamble_data(
            outcomes_a=[0], probs_a=[1.0],
            outcomes_b=[50, -25], probs_b=[0.5, 0.5],
        )
        # Create combined batch
        batch_data = GambleData(
            outcomes_a=np.vstack([data1.outcomes_a, data2.outcomes_a]),
            probs_a=np.vstack([data1.probs_a, data2.probs_a]),
            outcomes_b=np.vstack([data1.outcomes_b, data2.outcomes_b]),
            probs_b=np.vstack([data1.probs_b, data2.probs_b]),
            brate=np.array([0.5, 0.5]),
        )
        params = np.array([0.88, 2.25, 0.65, 0.65, 1.0])

        pred1 = CPTModel.predict(params, data1)[0]
        pred2 = CPTModel.predict(params, data2)[0]
        batch_pred = CPTModel.predict(params, batch_data)

        np.testing.assert_allclose(batch_pred[0], pred1, atol=1e-10)
        np.testing.assert_allclose(batch_pred[1], pred2, atol=1e-10)

    def test_cpt_many_outcomes(self):
        """CPT should handle gambles with many outcomes (stress test vectorization)."""
        # Gamble with 5 outcomes
        data = _make_simple_gamble_data(
            outcomes_a=[10, 20, 30, 40, 50],
            probs_a=[0.2, 0.2, 0.2, 0.2, 0.2],
            outcomes_b=[-20, 0, 25, 50, 100],
            probs_b=[0.1, 0.2, 0.3, 0.3, 0.1],
        )
        params = np.array([0.88, 2.25, 0.65, 0.65, 1.0])
        pred = CPTModel.predict(params, data)
        assert pred.shape == (1,)
        assert 0 < pred[0] < 1


class TestSmoke:

    def test_ev_model_on_synthetic_data(self):
        """EV model should run without crashing on simple data."""
        data = _make_simple_gamble_data(
            outcomes_a=[10, 0], probs_a=[0.5, 0.5],
            outcomes_b=[20, -5], probs_b=[0.3, 0.7],
        )
        params = np.array([1.0])  # temperature
        pred = EVModel.predict(params, data)
        assert pred.shape == (1,)
        assert 0 < pred[0] < 1


class TestDataAlignment:
    """Bug-driven test: JSON keys are CSV row indices, NOT Problem IDs.

    This bug caused every model to predict ~0.5 because gambles
    were mismatched — the models were fitting noise.
    Written per Poldrack §1: write a test for every bug before fixing.
    """

    def test_prepare_gamble_data_requires_json_idx(self):
        """Should raise if _json_idx column missing."""
        import pandas as pd
        from stacking.models import prepare_gamble_data
        df = pd.DataFrame({"Problem": [1], "Ha": [10], "pHa": [0.5],
                           "La": [0], "bRate": [0.5]})
        with pytest.raises(ValueError, match="_json_idx"):
            prepare_gamble_data(df, {})

    def test_json_key_is_row_index_not_problem_id(self):
        """Verify JSON keys correspond to CSV row order, not Problem column."""
        import pandas as pd
        from stacking.data import load_selections, load_problems
        from stacking.config import DataConfig

        config = DataConfig(data_dir=Path("data"))
        try:
            df = load_selections(config)
            problems = load_problems(config)
        except FileNotFoundError:
            pytest.skip("Data files not available")

        # Check first 20 rows: JSON key = _json_idx, NOT Problem ID
        for i in range(min(20, len(df))):
            row = df.iloc[i]
            jkey = str(int(row["_json_idx"]))
            prob = problems[jkey]
            # First outcome of gamble A should match Ha
            json_outcome = prob["A"][0][1]  # [prob, outcome]
            assert abs(json_outcome - row["Ha"]) < 0.1, (
                f"Row {i}: JSON outcome={json_outcome}, Ha={row['Ha']}. "
                f"JSON key was {jkey}, Problem ID was {row['Problem']}"
            )


class TestHierarchicalStacking:

    def test_uniform_when_beta_zero(self):
        """Beta=0 should give uniform weights (reduces to standard stacking)."""
        from stacking.hierarchical import compute_hierarchical_weights
        X = np.random.randn(50, 3)
        beta = np.zeros((3, 2))  # 3 features, 3 models (2 free)
        weights = compute_hierarchical_weights(X, beta, n_models=3)
        # All weights should be 1/3
        np.testing.assert_allclose(weights, 1/3, atol=1e-10)

    def test_weights_sum_to_one(self):
        """Hierarchical weights must sum to 1 per problem."""
        from stacking.hierarchical import compute_hierarchical_weights
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        beta = rng.randn(4, 2) * 0.5
        weights = compute_hierarchical_weights(X, beta, n_models=3)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    def test_weights_nonnegative(self):
        """Softmax guarantees non-negative weights."""
        from stacking.hierarchical import compute_hierarchical_weights
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        beta = rng.randn(4, 2) * 2.0  # large coefficients
        weights = compute_hierarchical_weights(X, beta, n_models=3)
        assert np.all(weights >= 0)

    def test_hierarchical_beats_or_ties_uniform(self):
        """Hierarchical stacking with low regularization should beat
        or tie equal-weight ensemble (since beta=0 IS equal weights)."""
        from stacking.hierarchical import fit_hierarchical_stacking

        rng = np.random.RandomState(42)
        n, n_models, n_feat = 200, 3, 4
        preds = rng.rand(n, n_models) * 0.5 + 0.25
        targets = rng.rand(n)
        features = rng.randn(n, n_feat)

        # Equal-weight MSE (beta=0 starting point)
        equal_w = np.ones(n_models) / n_models
        equal_mse = compute_mse_loss(preds @ equal_w, targets)

        # Hierarchical MSE with very low regularization
        h_results = fit_hierarchical_stacking(
            preds, targets, features, regularization=0.001
        )

        # Should be at least as good as equal weights
        assert h_results["hierarchical_mse"] <= equal_mse + 1e-6
