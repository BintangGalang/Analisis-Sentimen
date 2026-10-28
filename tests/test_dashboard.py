"""
tests/test_dashboard.py
======================
Unit test untuk komponen dan fungsionalitas dashboard Streamlit.
"""

from pathlib import Path
import pytest
import pandas as pd

from pathlib import Path
import pytest
import pandas as pd

from src.dashboard.components import get_sentiment_badge_html, render_metric_card


def test_sentiment_badge_html():
    """Menguji rendering badge HTML sentimen untuk ketiga kelas."""
    badge_pos = get_sentiment_badge_html("positif")
    assert "POSITIF" in badge_pos
    assert "#2ecc71" in badge_pos
    assert "<span" in badge_pos

    badge_net = get_sentiment_badge_html("netral")
    assert "NETRAL" in badge_net
    assert "#95a5a6" in badge_net

    badge_neg = get_sentiment_badge_html("negatif")
    assert "NEGATIF" in badge_neg
    assert "#e74c3c" in badge_neg

    # Unknown fallback
    badge_other = get_sentiment_badge_html("lainnya")
    assert "LAINNYA" in badge_other
    assert "#34495e" in badge_other


def test_dashboard_required_data_artifacts():
    """Memastikan semua artefak tabel dan gambar yang dibutuhkan dashboard tersedia."""
    root = Path(__file__).resolve().parent.parent
    
    # 1. Tabel data
    labeled_csv = root / "data" / "labeled" / "youtube_labeled.csv"
    assert labeled_csv.exists(), f"File {labeled_csv} harus tersedia untuk dashboard."
    
    # 2. Tabel perbandingan model & indeks reputasi
    model_comp = root / "reports" / "tables" / "model_comparison.csv"
    assert model_comp.exists(), f"File {model_comp} harus tersedia."
    
    rep_aspect = root / "reports" / "tables" / "reputation_index_aspect.csv"
    assert rep_aspect.exists(), f"File {rep_aspect} harus tersedia."
    
    # 3. Model SVM & InSet Lexicon
    svm_path = root / "models" / "svm.joblib"
    assert svm_path.exists(), f"Model {svm_path} harus tersedia."
    
    inset_path = root / "data" / "interim" / "inset_lexicon.tsv"
    assert inset_path.exists(), f"Lexicon {inset_path} harus tersedia."
    
    # 4. Figures
    fig_dir = root / "reports" / "figures"
    expected_figs = [
        "sentiment_distribution.png",
        "aspect_sentiment_breakdown.png",
        "reputation_index_aspect.png",
        "model_comparison_chart.png",
        "wordcloud_combined.png",
    ]
    for fig_name in expected_figs:
        assert (fig_dir / fig_name).exists(), f"Gambar {fig_name} harus ada di {fig_dir}."
