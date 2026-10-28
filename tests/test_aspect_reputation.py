"""
tests/test_aspect_reputation.py
===============================
Unit tests untuk:
- src/evaluate/aspect.py
- src/evaluate/reputation_index.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluate.aspect import (
    classify_aspect,
    classify_aspects_multi,
    compute_aspect_sentiment_breakdown,
)
from src.evaluate.reputation_index import (
    calculate_reputation_index,
    compute_reputation_by_group,
    interpret_reputation_index,
)


class TestAspectClassification:
    def test_classify_aspect_akademik(self) -> None:
        text = "dosen di fakultas ftk sangat ramah dan kurikulum jelas"
        assert classify_aspect(text) == "akademik"

    def test_classify_aspect_fasilitas(self) -> None:
        text = "ruang kelas panas dan ac mati di perpustakaan"
        assert classify_aspect(text) == "fasilitas"

    def test_classify_aspect_biaya(self) -> None:
        text = "biaya ukt mahal dan kapan pendaftaran beasiswa kip"
        assert classify_aspect(text) == "penerimaan_biaya"

    def test_classify_aspect_layanan(self) -> None:
        text = "siakad down terus pas jadwal war krs"
        assert classify_aspect(text) == "layanan_administrasi"

    def test_classify_aspect_empty(self) -> None:
        assert classify_aspect("") == "lainnya"

    def test_classify_aspects_multi(self) -> None:
        text = "dosen ramah tapi gedung lab rusak"
        matched = classify_aspects_multi(text)
        assert "akademik" in matched
        assert "fasilitas" in matched

    def test_compute_aspect_sentiment_breakdown(self) -> None:
        df = pd.DataFrame({
            "aspect": ["akademik", "akademik", "fasilitas"],
            "sentiment": ["positif", "netral", "negatif"],
        })
        ct = compute_aspect_sentiment_breakdown(df)
        assert "positif" in ct.columns
        assert "Total" in ct.columns
        assert ct.loc["akademik", "Total"] == 2


class TestReputationIndex:
    def test_calculate_reputation_index_formula(self) -> None:
        # pos=80, neg=20, total=100 -> (80 - 20) / 100 = +0.60
        idx = calculate_reputation_index(80, 20, 100)
        assert idx == 0.60
        assert "Sangat Positif" in interpret_reputation_index(idx)

    def test_calculate_reputation_index_negative(self) -> None:
        # pos=10, neg=50, total=100 -> (10 - 50) / 100 = -0.40
        idx = calculate_reputation_index(10, 50, 100)
        assert idx == -0.40
        assert "Negatif" in interpret_reputation_index(idx)

    def test_calculate_reputation_index_zero_total(self) -> None:
        assert calculate_reputation_index(0, 0, 0) == 0.0

    def test_compute_reputation_by_group(self) -> None:
        df = pd.DataFrame({
            "aspect": ["akademik", "akademik", "fasilitas", "fasilitas"],
            "sentiment": ["positif", "positif", "negatif", "netral"],
        })
        rep_df = compute_reputation_by_group(df, group_col="aspect")
        assert len(rep_df) == 2
        assert "Indeks_Reputasi" in rep_df.columns

        row_akademik = rep_df[rep_df["aspect"] == "akademik"].iloc[0]
        assert row_akademik["Indeks_Reputasi"] == 1.0  # 2 pos, 0 neg, total 2
