"""
tests/test_lexicon.py
=====================
Unit tests untuk Komponen A: InSet Lexicon Model (src/models/lexicon.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.lexicon import InSetLexiconModel


@pytest.fixture
def lexicon_model() -> InSetLexiconModel:
    return InSetLexiconModel(
        lexicon_path="data/interim/inset_lexicon.tsv",
        neutral_threshold=0.5,
    )


class TestInSetLexiconModel:
    def test_lexicon_loaded(self, lexicon_model: InSetLexiconModel) -> None:
        assert len(lexicon_model.lexicon) > 0
        assert "bagus" in lexicon_model.lexicon
        assert lexicon_model.lexicon["bagus"] > 0
        assert "rusak" in lexicon_model.lexicon
        assert lexicon_model.lexicon["rusak"] < 0

    def test_positive_text_prediction(self, lexicon_model: InSetLexiconModel) -> None:
        text = "undiksha kampusnya sangat bagus dan dosennya ramah hebat mantap"
        pred = lexicon_model.predict_one(text)
        assert pred == "positif"

    def test_negative_text_prediction(self, lexicon_model: InSetLexiconModel) -> None:
        text = "siakad error lelet rusak parah kecewa banget"
        pred = lexicon_model.predict_one(text)
        assert pred == "negatif"

    def test_neutral_text_prediction(self, lexicon_model: InSetLexiconModel) -> None:
        text = "jadwal pendaftaran dibuka hari senin pukul sepuluh pagi"
        pred = lexicon_model.predict_one(text)
        assert pred == "netral"

    def test_predict_proba_shape_and_sum(self, lexicon_model: InSetLexiconModel) -> None:
        texts = [
            "keren banget undiksha",
            "info jadwal kuliah",
            "server siakad down terus parah",
        ]
        probas = lexicon_model.predict_proba(texts)
        assert probas.shape == (3, 3)
        # Total probabilitas setiap baris mendekati 1.0
        for row in probas:
            assert np.isclose(np.sum(row), 1.0, atol=1e-5)
            assert np.all(row >= 0.0)
