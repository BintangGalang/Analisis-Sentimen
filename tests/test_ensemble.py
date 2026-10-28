"""
tests/test_ensemble.py
======================
Unit tests untuk src/models/ensemble.py (Stacking & Weighted Soft Voting).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ensemble import (
    CLASS_NAMES,
    StackingEnsemble,
    WeightedSoftVotingEnsemble,
    compute_metrics,
)


@pytest.fixture
def mock_probabilities() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixture probabilitas dummy untuk 3 model dan target label (20 sampel)."""
    np.random.seed(42)
    n = 30

    # Model A: condong ke kelas 0 dan 1
    p_a = np.random.dirichlet((2.0, 2.0, 1.0), size=n)
    # Model B: condong ke kelas 1 dan 2
    p_b = np.random.dirichlet((1.0, 3.0, 2.0), size=n)
    # Model C: condong ke kelas 2
    p_c = np.random.dirichlet((1.0, 1.0, 3.0), size=n)

    y = np.random.choice([0, 1, 2], size=n, p=[0.2, 0.4, 0.4])

    return p_a, p_b, p_c, y


class TestStackingEnsemble:
    def test_fit_and_predict_shape(self, mock_probabilities) -> None:
        p_a, p_b, p_c, y = mock_probabilities
        stacker = StackingEnsemble(random_state=42)

        assert not stacker.is_fitted
        with pytest.raises(RuntimeError):
            stacker.predict_proba(p_a, p_b, p_c)

        stacker.fit(p_a, p_b, p_c, y)
        assert stacker.is_fitted

        probs = stacker.predict_proba(p_a, p_b, p_c)
        preds = stacker.predict(p_a, p_b, p_c)

        assert probs.shape == (len(y), 3)
        assert preds.shape == (len(y),)
        assert np.all(preds >= 0) and np.all(preds < 3)

        # Probabilitas per baris mendekati 1.0
        row_sums = np.sum(probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_meta_coefficients(self, mock_probabilities) -> None:
        p_a, p_b, p_c, y = mock_probabilities
        stacker = StackingEnsemble(random_state=42)
        stacker.fit(p_a, p_b, p_c, y)

        df_coef = stacker.get_meta_coefficients()
        assert df_coef.shape == (3, 9)  # 3 kelas x 9 fitur meta (3 model * 3 proba)
        assert "lexicon_positif" in df_coef.columns
        assert "classical_netral" in df_coef.columns
        assert "indobert_negatif" in df_coef.columns


class TestWeightedSoftVoting:
    def test_fit_and_weights_normalized(self, mock_probabilities) -> None:
        p_a, p_b, p_c, y = mock_probabilities
        voter = WeightedSoftVotingEnsemble()
        voter.fit(p_a, p_b, p_c, y, step=0.05)

        assert voter.is_fitted
        w = voter.weights
        total_w = sum(w.values())
        assert np.isclose(total_w, 1.0, atol=1e-3)
        for k, val in w.items():
            assert val >= 0.0

    def test_predict_proba_and_pred(self, mock_probabilities) -> None:
        p_a, p_b, p_c, y = mock_probabilities
        voter = WeightedSoftVotingEnsemble(weights={"lexicon": 0.2, "classical": 0.5, "bert": 0.3})
        probs = voter.predict_proba(p_a, p_b, p_c)
        preds = voter.predict(p_a, p_b, p_c)

        assert probs.shape == (len(y), 3)
        assert preds.shape == (len(y),)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


class TestMetricsCalculation:
    def test_compute_metrics_keys(self) -> None:
        y_t = np.array([0, 1, 2, 1, 0])
        y_p = np.array([0, 1, 2, 0, 0])
        m = compute_metrics(y_t, y_p)

        assert "accuracy" in m
        assert "precision_macro" in m
        assert "recall_macro" in m
        assert "f1_macro" in m
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_macro"] <= 1.0
