"""
tests/test_weekly_issues.py
===========================
Unit & Integration Tests untuk Fitur Pemantauan Isu Mingguan (PRD Langkah 2, 3, 4, 5).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.collect.weekly import WeeklyDataCollector, anonymize_text, anonymize_username, generate_id_anonim
from src.features.issues import WeeklyIssueDetector
from src.pipeline.weekly_run import WeeklyPipelineRunner
from src.dashboard.flask_app import app


@pytest.fixture
def test_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_weekly_collector_anonymization():
    """Menguji fungsi anonimisasi teks dan username."""
    # Test anonymize_text
    t1 = "Terima kasih @rektor_undiksha dan @bintang atas bimbingannya!"
    anon1 = anonymize_text(t1)
    assert "@pengguna" in anon1
    assert "@bintang" not in anon1

    # Official mention preserved
    t2 = "Akun resmi @undiksha merilis jadwal pendaftaran!"
    anon2 = anonymize_text(t2)
    assert "@undiksha" in anon2

    # Test anonymize_username
    u1 = "bintang_mhs"
    anon_u1 = anonymize_username(u1)
    assert anon_u1.startswith("user_")
    assert u1 not in anon_u1

    # Official username preserved
    u_off = "undiksha.official"
    assert anonymize_username(u_off) == u_off

    # Test unique id generation
    id1 = generate_id_anonim("youtube", "komentar kampus bagus", "2026-11-23")
    id2 = generate_id_anonim("youtube", "komentar kampus bagus", "2026-11-23")
    id3 = generate_id_anonim("twitter", "komentar kampus bagus", "2026-11-23")
    assert id1 == id2
    assert id1 != id3


def test_issue_detector_calculation():
    """Menguji rumus lonjakan (Growth Ratio) dan Skor Isu."""
    detector = WeeklyIssueDetector()

    # Laplace smoothing growth ratio
    growth = detector.calculate_growth_ratio(vol_curr=10, past_volumes=[2, 2, 2, 2])
    # (10 + 1) / (2 + 1) = 11 / 3 = 3.67
    assert round(growth, 2) == 3.67

    # Issue score: w_vol=0.30, w_neg=0.45, w_growth=0.25
    # High negative + high growth should yield high issue score
    score = detector.calculate_issue_score(vol=20, max_vol=20, neg_ratio=0.8, growth_ratio=4.0)
    assert score >= 65.0  # Tingkat perhatian Tinggi


def test_api_weekly_monitoring_data(test_client):
    """Menguji endpoint /api/weekly_monitoring_data."""
    response = test_client.get("/api/weekly_monitoring_data")
    assert response.status_code == 200
    data = response.get_json()
    assert data["available"] is True
    assert "summary" in data
    assert "top_5_issues" in data
    assert "daily_trend" in data
    assert "platform_dist" in data
    assert "heatmap" in data
    assert "top_words" in data
    assert "multi_week_trend" in data
    assert "posts" in data


def test_api_weekly_issues_history(test_client):
    """Menguji endpoint /api/weekly_issues_history."""
    response = test_client.get("/api/weekly_issues_history")
    assert response.status_code == 200
    data = response.get_json()
    assert data["available"] is True
    assert "issues" in data
    assert "recurring_analysis" in data
    assert len(data["issues"]) > 0


def test_api_predict_multicomponent(test_client):
    """Menguji endpoint /api/predict_multicomponent."""
    payload = {
        "text": "Perpustakaan kampus Undiksha sekarang sejuk dan fasilitas koleksi bukunya sangat lengkap!"
    }
    response = test_client.post("/api/predict_multicomponent", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "components" in data
    assert "lexicon" in data["components"]
    assert "svm" in data["components"]
    assert "indobert" in data["components"]
    assert "ensemble" in data
    assert "stacking" in data["ensemble"]
    assert "soft_voting" in data["ensemble"]
    assert "final_verdict" in data
    assert data["detected_aspect"] == "fasilitas"
    assert data["final_verdict"]["sentiment"] in ["positif", "netral", "negatif"]


def test_download_endpoints(test_client):
    """Menguji endpoint download CSV."""
    # Download ringkasan isu
    res_issues = test_client.get("/api/download_weekly_issues")
    assert res_issues.status_code == 200
    assert "text/csv" in res_issues.content_type

    # Download prediksi
    res_preds = test_client.get("/api/download_weekly_predictions")
    assert res_preds.status_code == 200
    assert "text/csv" in res_preds.content_type


def test_issue_detector_prioritize_negative():
    """Menguji prioritas deteksi isu negatif dan ekstraksi frasa keluhan."""
    detector = WeeklyIssueDetector()
    df_dummy = pd.DataFrame([
        {"minggu": "2026-W47", "aspek": "fasilitas", "sentimen": "negatif", "teks_bersih": "wifi lemot rusak parah mati", "teks_asli_anonim": "wifi lemot banget"},
        {"minggu": "2026-W47", "aspek": "fasilitas", "sentimen": "negatif", "teks_bersih": "ac gedung kelas mati panas", "teks_asli_anonim": "ac kelas mati"},
        {"minggu": "2026-W47", "aspek": "fasilitas", "sentimen": "positif", "teks_bersih": "gedung baru megah mantap", "teks_asli_anonim": "gedung megah"},
        {"minggu": "2026-W47", "aspek": "akademik", "sentimen": "positif", "teks_bersih": "dosen baik materi jelas", "teks_asli_anonim": "dosen baik"},
        {"minggu": "2026-W47", "aspek": "akademik", "sentimen": "positif", "teks_bersih": "kurikulum mantap keren", "teks_asli_anonim": "kurikulum keren"},
        {"minggu": "2026-W47", "aspek": "akademik", "sentimen": "netral", "teks_bersih": "jadwal kuliah keluar", "teks_asli_anonim": "jadwal keluar"},
    ])

    issues = detector.detect_issues_for_week(df_dummy, target_week="2026-W47")
    assert not issues.empty
    fasilitas_issue = issues[issues["aspek"] == "fasilitas"].iloc[0]
    assert fasilitas_issue["persen_negatif"] > 50.0
    assert fasilitas_issue["is_prioritas_negatif"] is True or fasilitas_issue["is_prioritas_negatif"] == 1
    # Isu dengan sentimen negatif tinggi harus berada di ranking teratas
    assert issues.iloc[0]["aspek"] == "fasilitas"


def test_api_fetch_weekly_parameter_parsing(test_client, monkeypatch):
    """Menguji endpoint /api/fetch_weekly menerima rentang hari, kuota post, dan kata kunci."""
    # Mock subprocess.run agar test cepat tanpa memicu live scraping nyata
    import subprocess
    recorded_cmd = []

    def mock_run(cmd, *args, **kwargs):
        recorded_cmd.extend(cmd)
        class DummyProc:
            returncode = 0
            stdout = '{"final": {"success": true, "added": 2, "message": "OK"}}\n'
            stderr = ""
        return DummyProc()

    monkeypatch.setattr(subprocess, "run", mock_run)

    payload = {
        "days": 14,
        "max_videos": 20,
        "max_comments": 80,
        "max_posts": 4,
        "keywords": "undiksha ukt, undiksha siakad"
    }

    res = test_client.post("/api/fetch_weekly", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["days"] == 14
    assert data["max_videos"] == 20
    assert data["max_comments"] == 80
    assert data["max_posts"] == 4

    # Verifikasi CLI argument diteruskan dengan benar ke subprocess
    cmd_str = " ".join(recorded_cmd)
    assert "--days 14" in cmd_str
    assert "--max-videos 20" in cmd_str
    assert "--max-comments 80" in cmd_str
    assert "--max-posts 4" in cmd_str
    assert "undiksha ukt, undiksha siakad" in cmd_str


