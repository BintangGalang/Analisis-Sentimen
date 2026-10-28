"""
tests/test_label.py
===================
Unit test untuk modul pelabelan dan evaluasi kesepakatan Cohen's Kappa:
- src/label/annotation_tool.py
- src/label/kappa.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.label.annotation_tool import (
    ANNOTATION_COLUMNS,
    create_annotation_sample,
    export_double_annotation_templates,
)
from src.label.kappa import (
    calculate_agreement_metrics,
    interpret_kappa,
    reconcile_annotations,
)


@pytest.fixture
def sample_processed_df() -> pd.DataFrame:
    """Fixture DataFrame hasil preprocessing dummy."""
    data = {
        "id_anon": [f"hash_{i}" for i in range(20)],
        "platform": ["youtube"] * 20,
        "text_raw": [f"Komentar asli ke-{i} tentang Undiksha kampus kita tercinta!" for i in range(20)],
        "text_clean": [f"komentar asli ke-{i} tentang undiksha kampus kita tercinta" for i in range(20)],
        "text_tokens": [f"komentar asli undiksha kampus cinta" for i in range(20)],
        "text_stemmed": [f"komentar asli undiksha kampus cinta" for i in range(20)],
        "emoji_replaced": [False] * 20,
        "has_codemix": [False] * 20,
        "en_words": ["[]"] * 20,
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────
# Test Annotation Tool
# ─────────────────────────────────────────────────────────────

class TestAnnotationTool:
    def test_create_annotation_sample_shape(self, sample_processed_df: pd.DataFrame) -> None:
        sample = create_annotation_sample(sample_processed_df, sample_size=10, seed=42)
        assert len(sample) == 10
        for col in ANNOTATION_COLUMNS:
            assert col in sample.columns

    def test_create_annotation_sample_empty_raises(self) -> None:
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            create_annotation_sample(empty_df, sample_size=5)

    def test_export_double_annotation_templates(
        self, sample_processed_df: pd.DataFrame, tmp_path
    ) -> None:
        sample = create_annotation_sample(sample_processed_df, sample_size=5, seed=42)
        path_a, path_b = export_double_annotation_templates(
            sample, output_dir=tmp_path, platform="youtube"
        )
        assert path_a.exists()
        assert path_b.exists()

        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        assert len(df_a) == 5
        assert len(df_b) == 5
        assert "sentiment" in df_a.columns
        assert "aspect" in df_a.columns


# ─────────────────────────────────────────────────────────────
# Test Kappa & Agreement
# ─────────────────────────────────────────────────────────────

class TestKappaMetrics:
    def test_perfect_agreement(self) -> None:
        y_a = ["positif", "netral", "negatif", "positif", "netral"]
        y_b = ["positif", "netral", "negatif", "positif", "netral"]
        metrics = calculate_agreement_metrics(y_a, y_b)
        assert metrics["cohen_kappa"] == 1.0
        assert metrics["observed_agreement"] == 1.0
        assert metrics["disagreements_count"] == 0
        assert "Almost Perfect" in metrics["interpretation"]

    def test_substantial_agreement(self) -> None:
        # 9 dari 10 sama, 1 beda
        y_a = ["positif", "netral", "negatif", "positif", "netral", "positif", "netral", "negatif", "positif", "netral"]
        y_b = ["positif", "netral", "negatif", "positif", "netral", "positif", "netral", "negatif", "positif", "positif"]
        metrics = calculate_agreement_metrics(y_a, y_b)
        assert metrics["cohen_kappa"] >= 0.60
        assert metrics["disagreements_count"] == 1

    def test_interpret_kappa_categories(self) -> None:
        assert "Poor" in interpret_kappa(-0.1)
        assert "Fair" in interpret_kappa(0.35)
        assert "Moderate" in interpret_kappa(0.55)
        assert "Substantial" in interpret_kappa(0.75)
        assert "Almost Perfect" in interpret_kappa(0.95)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_agreement_metrics(["positif"], ["positif", "netral"])


# ─────────────────────────────────────────────────────────────
# Test Reconciliation
# ─────────────────────────────────────────────────────────────

class TestReconciliation:
    def test_reconcile_with_conflict(self) -> None:
        df_a = pd.DataFrame({
            "id_anon": ["id1", "id2", "id3"],
            "sentiment": ["positif", "netral", "negatif"],
            "aspect": ["akademik", "biaya", "fasilitas"],
        })
        df_b = pd.DataFrame({
            "id_anon": ["id1", "id2", "id3"],
            "sentiment": ["positif", "negatif", "negatif"],  # id2 konflik
            "aspect": ["akademik", "layanan", "fasilitas"],
        })
        reconciled = reconcile_annotations(df_a, df_b)
        assert len(reconciled) == 3
        assert reconciled.loc[reconciled["id_anon"] == "id1", "is_conflict"].values[0] == False
        assert reconciled.loc[reconciled["id_anon"] == "id1", "sentiment_final"].values[0] == "positif"
        assert reconciled.loc[reconciled["id_anon"] == "id2", "is_conflict"].values[0] == True
        assert reconciled.loc[reconciled["id_anon"] == "id2", "sentiment_final"].values[0] == "KONFLIK"
        assert reconciled.loc[reconciled["id_anon"] == "id3", "is_conflict"].values[0] == False
        assert reconciled.loc[reconciled["id_anon"] == "id3", "sentiment_final"].values[0] == "negatif"
