"""
tests/test_flask_dashboard.py
=============================
Unit test untuk Flask web application dashboard reputasi Undiksha.
"""

from pathlib import Path
import pytest
from src.dashboard.flask_app import app


@pytest.fixture
def client():
    """Membuat test client Flask."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_flask_index_page(client):
    """Menguji halaman utama dashboard memuat dengan status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Sistem Pemantauan Reputasi Undiksha" in response.data
    assert b"Chart.js" in response.data or b"chart-sentiment-donut" in response.data


def test_flask_api_stats(client):
    """Menguji endpoint /api/stats mengembalikan JSON agregasi sentimen dan indeks reputasi."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert "total" in data
    assert data["total"] > 0
    assert "positif" in data
    assert "netral" in data
    assert "negatif" in data
    assert "reputation_index" in data
    assert "reputation_index_pct" in data
    assert "predikat" in data
    assert "aspect_stats" in data


def test_flask_api_predict(client):
    """Menguji endpoint /api/predict untuk inferensi langsung opini."""
    sample_text = "Fasilitas kampus Undiksha sekarang sangat lengkap, perpustakaannya nyaman sekali!"
    response = client.post("/api/predict", json={"text": sample_text})
    assert response.status_code == 200
    data = response.get_json()
    assert data["sentiment"] in ["positif", "netral", "negatif"]
    assert "aspect" in data
    assert "probabilities" in data
    assert "positif" in data["probabilities"]
    assert "netral" in data["probabilities"]
    assert "negatif" in data["probabilities"]


def test_flask_api_data_explorer(client):
    """Menguji endpoint /api/data untuk eksplorasi korpus komentar."""
    response = client.get("/api/data?limit=10")
    assert response.status_code == 200
    data = response.get_json()
    assert "data" in data
    assert len(data["data"]) <= 10
    assert "total" in data


def test_flask_api_reports(client):
    """Menguji endpoint /api/reports mengembalikan data evaluasi model."""
    response = client.get("/api/reports")
    assert response.status_code == 200
    data = response.get_json()
    assert "model_comparison" in data
    assert "ablation_study" in data
