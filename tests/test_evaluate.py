"""
tests/test_evaluate.py
======================
Unit tests untuk modul evaluasi:
- src/evaluate/metrics.py
- src/evaluate/significance.py
- src/evaluate/ablation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate.ablation import evaluate_two_model_soft_voting, run_ablation_study
from src.evaluate.metrics import (
    calculate_comprehensive_metrics,
    format_metrics_table,
)
from src.evaluate.significance import (
    mcnemar_test,
    paired_cv_t_test,
    run_pairwise_significance_matrix,
)


class TestMetrics:
    def test_calculate_comprehensive_metrics(self) -> None:
        y_true = ["negatif", "netral", "positif", "netral", "positif"]
        y_pred = ["negatif", "netral", "positif", "positif", "positif"]

        res = calculate_comprehensive_metrics(y_true, y_pred)
        assert "accuracy" in res
        assert "f1_macro" in res
        assert "per_class" in res
        assert "confusion_matrix" in res
        assert res["confusion_matrix"].shape == (3, 3)
        assert res["total_samples"] == 5

    def test_format_metrics_table(self) -> None:
        y_true = [0, 1, 2, 1, 2]
        y_pred = [0, 1, 2, 1, 1]
        res = calculate_comprehensive_metrics(y_true, y_pred)
        table_str = format_metrics_table(res, model_name="TestModel")
        assert "TestModel" in table_str
        assert "Accuracy" in table_str


class TestSignificance:
    def test_mcnemar_identical_predictions(self) -> None:
        y_true = [0, 1, 2, 1, 0, 2]
        y_pred_a = [0, 1, 2, 1, 0, 2]
        y_pred_b = [0, 1, 2, 1, 0, 2]

        res = mcnemar_test(y_true, y_pred_a, y_pred_b, "M1", "M2")
        assert res["statistic"] == 0.0
        assert res["p_value"] == 1.0
        assert not res["is_significant"]

    def test_mcnemar_significant_difference(self) -> None:
        # Model B jauh lebih akurat daripada Model A
        y_true = [1] * 50
        y_pred_a = [0] * 50  # 0% akurasi
        y_pred_b = [1] * 50  # 100% akurasi

        res = mcnemar_test(y_true, y_pred_a, y_pred_b, "Poor", "Good")
        assert res["p_value"] < 0.001
        assert res["is_significant"]
        assert "Good terbukti lebih unggul" in res["conclusion"]

    def test_paired_cv_t_test(self) -> None:
        scores_a = [0.70, 0.72, 0.71, 0.73, 0.70]
        scores_b = [0.85, 0.86, 0.87, 0.86, 0.88]
        res = paired_cv_t_test(scores_a, scores_b, "Model A", "Model B")
        assert res["is_significant"]
        assert res["mean_diff"] > 0

    def test_pairwise_significance_matrix(self) -> None:
        y_true = np.array([0, 1, 2, 1, 0])
        preds = {
            "M1": np.array([0, 1, 2, 1, 0]),
            "M2": np.array([0, 1, 1, 1, 0]),
            "M3": np.array([1, 1, 1, 1, 1]),
        }
        df_mat = run_pairwise_significance_matrix(preds, y_true)
        assert df_mat.shape == (3, 3)
        assert df_mat.iloc[0, 0] == "-"


class TestAblation:
    def test_evaluate_two_model_soft_voting(self) -> None:
        np.random.seed(42)
        n = 20
        p1 = np.random.dirichlet((1, 2, 1), size=n)
        p2 = np.random.dirichlet((2, 1, 2), size=n)
        y = np.random.choice([0, 1, 2], size=n)

        p_comb, metrics, w = evaluate_two_model_soft_voting(p1, p2, y)
        assert p_comb.shape == (n, 3)
        assert "f1_macro" in metrics
        assert 0.0 <= w <= 1.0

    def test_run_ablation_study_structure(self, tmp_path) -> None:
        n = 20
        np.random.seed(42)
        bundle = {
            "oof_a": np.random.dirichlet((1, 1, 1), size=n),
            "oof_b": np.random.dirichlet((2, 2, 1), size=n),
            "oof_c": np.random.dirichlet((1, 2, 2), size=n),
            "oof_soft_vote": np.random.dirichlet((2, 2, 2), size=n),
            "oof_stacking": np.random.dirichlet((2, 2, 2), size=n),
            "y_true": np.random.choice([0, 1, 2], size=n),
        }
        df_ab = run_ablation_study(bundle, tables_dir=tmp_path)
        assert len(df_ab) == 8
        assert "Konfigurasi Model" in df_ab.columns
        assert "F1-Macro" in df_ab.columns
