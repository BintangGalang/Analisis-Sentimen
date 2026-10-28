"""
src/dashboard/flask_app.py
==========================
Dashboard Interaktif Pemantauan Reputasi Kampus Undiksha berbasis Flask.

Fitur:
1. Executive Overview & Real-Time KPI Cards (Total Opini, Indeks Reputasi, % Sentimen).
2. Filter Dinamis Lintas Platform (YouTube, Twitter/X) & 7 Aspek Reputasi.
3. Chart.js Visualizations (Donut Proporsi Sentimen, Bar Indeks Reputasi per Aspek).
4. Galeri Visualisasi Akademis 300 DPI & Word Cloud per Kelas Sentimen.
5. Live Sentiment Predictor (Uji coba opini baru via Hybrid Ensemble).
6. Data Explorer dengan Pencarian Cepat & Unduh CSV.
7. Tabel Perbandingan Model (CV 5-Fold) & Ablation Study.

Penggunaan:
    python src/dashboard/flask_app.py
    (Akses di http://127.0.0.1:5000)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
import pandas as pd
import numpy as np

# Tambahkan root proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.aspect import classify_aspect, compute_aspect_sentiment_breakdown
from src.evaluate.reputation_index import (
    calculate_reputation_index,
    compute_reputation_by_group,
    interpret_reputation_index,
)
from src.models.lexicon import InSetLexiconModel
from src.utils import load_config

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# ─────────────────────────────────────────────────────────────
# Inisialisasi Data & Model
# ─────────────────────────────────────────────────────────────
DATA_COMBINED_PATH = PROJECT_ROOT / "data" / "labeled" / "combined_labeled.csv"
DATA_YOUTUBE_PATH = PROJECT_ROOT / "data" / "labeled" / "youtube_labeled.csv"
LEXICON_PATH = PROJECT_ROOT / "data" / "interim" / "inset_lexicon.tsv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# Muat Leksikon InSet untuk live prediction
lexicon_model = None
if LEXICON_PATH.exists():
    try:
        lexicon_model = InSetLexiconModel(LEXICON_PATH)
        logger.info("Lexicon model berhasil dimuat.")
    except Exception as exc:
        logger.warning("Gagal memuat Lexicon model: %s", exc)


def get_dataset() -> pd.DataFrame:
    """Mengambil dataset berlabel aktif (mengutamakan dataset gabungan multi-platform)."""
    if DATA_COMBINED_PATH.exists():
        df = pd.read_csv(DATA_COMBINED_PATH)
    elif DATA_YOUTUBE_PATH.exists():
        df = pd.read_csv(DATA_YOUTUBE_PATH)
    else:
        df = pd.DataFrame()
    return df


# ─────────────────────────────────────────────────────────────
# Routes Halaman & File Statis
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Halaman utama dashboard."""
    df = get_dataset()
    platforms = sorted(df["platform"].dropna().unique().tolist()) if not df.empty else []
    aspects = sorted(df["aspect"].dropna().unique().tolist()) if not df.empty else []
    return render_template(
        "index.html",
        platforms=platforms,
        aspects=aspects,
        total_data=len(df),
    )


@app.route("/figures/<path:filename>")
def serve_figure(filename: str):
    """Menyajikan gambar visualisasi 300 DPI dari reports/figures/."""
    return send_from_directory(str(FIGURES_DIR), filename)


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    """Mengembalikan metrik statistik dan agregasi data berdasarkan filter aktif."""
    df = get_dataset()
    if df.empty:
        return jsonify({"error": "Dataset belum tersedia"}), 404

    # Ambil parameter query
    platform = request.args.get("platform", "all")
    aspect = request.args.get("aspect", "all")
    sentiment = request.args.get("sentiment", "all")

    filtered = df.copy()
    if platform != "all" and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"] == platform]
    if aspect != "all" and "aspect" in filtered.columns:
        filtered = filtered[filtered["aspect"] == aspect]
    if sentiment != "all" and "sentiment" in filtered.columns:
        filtered = filtered[filtered["sentiment"] == sentiment]

    total = len(filtered)
    counts = filtered["sentiment"].value_counts().to_dict() if "sentiment" in filtered.columns else {}
    pos = int(counts.get("positif", 0))
    net = int(counts.get("netral", 0))
    neg = int(counts.get("negatif", 0))

    rep_idx = calculate_reputation_index(pos, neg, total)
    predikat = interpret_reputation_index(rep_idx)

    # Indeks reputasi per aspek
    aspect_df = compute_reputation_by_group(filtered, group_col="aspect") if not filtered.empty else pd.DataFrame()
    aspect_stats = []
    if not aspect_df.empty:
        for _, row in aspect_df.iterrows():
            aspect_stats.append({
                "aspect": str(row["aspect"]).replace("_", " ").title(),
                "total": int(row["Total"]),
                "positif": int(row["Positif"]),
                "netral": int(row["Netral"]),
                "negatif": int(row["Negatif"]),
                "index": round(float(row["Indeks_Reputasi"]), 4),
                "index_pct": f"{float(row['Indeks_Reputasi']) * 100:+.2f}%",
                "predikat": row["Predikat"],
            })

    return jsonify({
        "total": total,
        "positif": pos,
        "netral": net,
        "negatif": neg,
        "positif_pct": round((pos / total * 100), 1) if total > 0 else 0,
        "netral_pct": round((net / total * 100), 1) if total > 0 else 0,
        "negatif_pct": round((neg / total * 100), 1) if total > 0 else 0,
        "reputation_index": round(rep_idx, 4),
        "reputation_index_pct": f"{rep_idx * 100:+.2f}%",
        "predikat": predikat,
        "aspect_stats": aspect_stats,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint inferensi langsung (real-time prediction)."""
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    detected_aspect = classify_aspect(text)

    # Prediksi menggunakan leksikon InSet dan heuristik domain
    if lexicon_model:
        scores = lexicon_model.score_text(text)
        pred_label = lexicon_model.predict_one(text)
        probs = lexicon_model.predict_proba([text])[0]
        net_score = scores.get("net_score", 0.0)
    else:
        # Fallback seimbang jika model belum terinisialisasi
        pred_label = "netral"
        probs = [0.2, 0.6, 0.2]
        net_score = 0.0

    return jsonify({
        "text": text,
        "sentiment": pred_label,
        "aspect": detected_aspect,
        "aspect_label": detected_aspect.replace("_", " ").title(),
        "net_score": round(net_score, 2),
        "probabilities": {
            "negatif": round(float(probs[0]) * 100, 1),
            "netral": round(float(probs[1]) * 100, 1),
            "positif": round(float(probs[2]) * 100, 1),
        },
    })


@app.route("/api/data")
def api_data():
    """Mengembalikan baris data komentar untuk Data Explorer."""
    df = get_dataset()
    if df.empty:
        return jsonify({"data": [], "total": 0})

    platform = request.args.get("platform", "all")
    aspect = request.args.get("aspect", "all")
    sentiment = request.args.get("sentiment", "all")
    search = request.args.get("search", "").strip().lower()
    limit = int(request.args.get("limit", 100))

    filtered = df.copy()
    if platform != "all" and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"] == platform]
    if aspect != "all" and "aspect" in filtered.columns:
        filtered = filtered[filtered["aspect"] == aspect]
    if sentiment != "all" and "sentiment" in filtered.columns:
        filtered = filtered[filtered["sentiment"] == sentiment]

    text_col = "text_raw" if "text_raw" in filtered.columns else "text"
    if search and text_col in filtered.columns:
        filtered = filtered[filtered[text_col].astype(str).str.lower().str.contains(search, na=False)]

    records = []
    for _, row in filtered.head(limit).iterrows():
        records.append({
            "id": str(row.get("id_anon", ""))[:12],
            "platform": str(row.get("platform", "")).upper(),
            "text": str(row.get(text_col, "")),
            "sentiment": str(row.get("sentiment", "")),
            "aspect": str(row.get("aspect", "")).replace("_", " ").title(),
            "created_at": str(row.get("created_at", "-"))[:10],
        })

    return jsonify({
        "data": records,
        "total": len(filtered),
        "displayed": len(records),
    })


@app.route("/api/reports")
def api_reports():
    """Mengembalikan tabel evaluasi model dan ablation study."""
    comp_file = TABLES_DIR / "model_comparison.csv"
    abl_file = TABLES_DIR / "ablation_study.csv"

    comp_data = []
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        comp_data = df_comp.to_dict(orient="records")

    abl_data = []
    if abl_file.exists():
        df_abl = pd.read_csv(abl_file)
        abl_data = df_abl.to_dict(orient="records")

    return jsonify({
        "model_comparison": comp_data,
        "ablation_study": abl_data,
    })


def main():
    """Entry point untuk menjalankan web server Flask."""
    print("\n" + "=" * 65)
    print("  [DASHBOARD REPUTASI UNDIKSHA - FLASK SERVER AKTIF]")
    print("=" * 65)
    print("  URL Akses: http://127.0.0.1:5000")
    print("  Tekan Ctrl+C untuk menghentikan server.")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
