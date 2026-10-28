"""
tests/test_features_models.py
=============================
Unit tests untuk:
- src/features/tfidf_vectorizer.py (Anti-leakage split, fit/transform, SMOTE)
- src/models/classical.py (SVM, Naive Bayes, Random Forest, OOF)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.tfidf_vectorizer import (
    TextFeatureExtractor,
    prepare_features,
    split_dataset,
)
from src.models.classical import (
    build_classical_model,
    evaluate_predictions,
    generate_oof_predictions,
)


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """Fixture dataset berlabel kecil untuk unit test."""
    texts = [
        "kampus undiksha sangat bagus dan nyaman",
        "siakad rusak lelet sekali down",
        "jadwal krs dibuka besok senin",
        "bangga sekali menjadi mahasiswa undiksha mantap",
        "biaya ukt mahal sekali kecewa parah",
        "info pendaftaran jalur mandiri undiksha",
        "fasilitas perpustakaan lengkap dan asri",
        "server down saat war krs menyebalkan",
        "selamat atas wisuda sarjana undiksha",
        "kapan pengumuman hasil seleksi snbt",
    ] * 5  # 50 baris
    sentiments = ["positif", "negatif", "netral", "positif", "negatif", "netral", "positif", "negatif", "positif", "netral"] * 5
    return pd.DataFrame({"text_clean": texts, "sentiment": sentiments})


class TestTfidfAndAntiLeakage:
    def test_split_dataset_proportions(self, mock_dataset: pd.DataFrame) -> None:
        df_train, df_val, df_test = split_dataset(
            mock_dataset, test_size=0.20, val_size=0.20, seed=42
        )
        assert len(df_train) + len(df_val) + len(df_test) == len(mock_dataset)
        assert len(df_test) == 10   # 20% dari 50
        assert len(df_val) == 10    # 20% dari 50
        assert len(df_train) == 30  # 60% dari 50

    def test_anti_leakage_vectorizer(self, mock_dataset: pd.DataFrame) -> None:
        train_texts = ["kampus bagus keren", "dosen hebat"]
        test_texts = ["gedung baru mewah"]  # kata 'mewah' tidak ada di train

        extractor = TextFeatureExtractor()
        X_tr = extractor.fit_transform(train_texts)
        assert extractor.is_fitted
        assert "bagus" in extractor.vectorizer.vocabulary_
        assert "mewah" not in extractor.vectorizer.vocabulary_

        X_te = extractor.transform(test_texts)
        assert X_te.shape[1] == X_tr.shape[1]

    def test_prepare_features_bundle(self, mock_dataset: pd.DataFrame) -> None:
        config = {
            "seed": 42,
            "split": {"test_size": 0.2, "val_size": 0.2},
            "tfidf": {"max_features": 500, "ngram_range": [1, 2], "sublinear_tf": True},
            "imbalance": {"method": "none"},
        }
        bundle = prepare_features(mock_dataset, config, use_smote=False)
        assert "X_train" in bundle
        assert "X_val" in bundle
        assert "X_test" in bundle
        assert bundle["X_train"].shape[0] == len(bundle["df_train"])
        assert bundle["X_val"].shape[0] == len(bundle["df_val"])
        assert bundle["X_test"].shape[0] == len(bundle["df_test"])


class TestClassicalModels:
    @pytest.mark.parametrize("model_name", ["svm", "naive_bayes", "random_forest"])
    def test_build_and_fit_models(self, model_name: str, mock_dataset: pd.DataFrame) -> None:
        config = {
            "seed": 42,
            "split": {"test_size": 0.2, "val_size": 0.2},
            "tfidf": {"max_features": 500, "ngram_range": [1, 2], "sublinear_tf": True},
            "imbalance": {"method": "none"},
            "classical_models": {
                "svm": {"kernel": "linear", "C": 1.0, "class_weight": None, "probability": True},
                "naive_bayes": {"alpha": 1.0},
                "random_forest": {"n_estimators": 10, "class_weight": None},
            },
        }
        bundle = prepare_features(mock_dataset, config, use_smote=False)
        clf = build_classical_model(model_name, config)
        clf.fit(bundle["X_train"], bundle["y_train"])

        preds = clf.predict(bundle["X_test"])
        assert len(preds) == len(bundle["y_test"])

        metrics = evaluate_predictions(bundle["y_test"], preds)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0

    def test_generate_oof_predictions(self, mock_dataset: pd.DataFrame) -> None:
        config = {
            "seed": 42,
            "split": {"test_size": 0.2, "val_size": 0.2},
            "tfidf": {"max_features": 500, "ngram_range": [1, 2], "sublinear_tf": True},
            "imbalance": {"method": "none"},
            "classical_models": {
                "svm": {"kernel": "linear", "C": 1.0, "class_weight": None, "probability": True},
                "naive_bayes": {"alpha": 1.0},
                "random_forest": {"n_estimators": 10, "class_weight": None},
            },
        }
        bundle = prepare_features(mock_dataset, config, use_smote=False)
        oof_probs, oof_f1 = generate_oof_predictions(
            "naive_bayes", bundle["X_train"], bundle["y_train"], config, n_splits=3
        )
        assert oof_probs.shape == (bundle["X_train"].shape[0], 3)
        assert 0.0 <= oof_f1 <= 1.0
