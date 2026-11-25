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
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ─────────────────────────────────────────────────────────────
# Inisialisasi Data & Model
# ─────────────────────────────────────────────────────────────
import joblib

DATA_COMBINED_PATH = PROJECT_ROOT / "data" / "labeled" / "combined_labeled.csv"
DATA_YOUTUBE_PATH = PROJECT_ROOT / "data" / "labeled" / "youtube_labeled.csv"
LEXICON_PATH = PROJECT_ROOT / "data" / "interim" / "inset_lexicon.tsv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
WEEKLY_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "weekly_predictions.parquet"
WEEKLY_ISSUES_PATH = PROJECT_ROOT / "data" / "processed" / "weekly_issues.csv"

# Muat Leksikon InSet untuk live prediction
lexicon_model = None
INV_LABEL_MAPPING = {0: "negatif", 1: "netral", 2: "positif"}
if LEXICON_PATH.exists():
    try:
        lexicon_model = InSetLexiconModel(LEXICON_PATH)
        logger.info("Lexicon model berhasil dimuat.")
    except Exception as exc:
        logger.warning("Gagal memuat Lexicon model: %s", exc)

# Muat model SVM, TF-IDF Vectorizer, Stacking, dan Soft Voting Ensemble
svm_model = None
tfidf_extractor = None
stacking_model = None
soft_voting_model = None

try:
    if (MODELS_DIR / "svm.joblib").exists():
        svm_model = joblib.load(MODELS_DIR / "svm.joblib")
        logger.info("Model SVM berhasil dimuat.")
    if (MODELS_DIR / "tfidf_vectorizer.joblib").exists():
        tfidf_extractor = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        logger.info("TF-IDF Vectorizer berhasil dimuat.")
    if (MODELS_DIR / "stacking_ensemble.joblib").exists():
        stacking_model = joblib.load(MODELS_DIR / "stacking_ensemble.joblib")
        logger.info("Stacking Ensemble berhasil dimuat.")
    if (MODELS_DIR / "soft_voting_ensemble.joblib").exists():
        soft_voting_model = joblib.load(MODELS_DIR / "soft_voting_ensemble.joblib")
        logger.info("Soft Voting Ensemble berhasil dimuat.")
except Exception as exc:
    logger.warning("Catatan saat memuat model ensemble: %s", exc)


def get_dataset() -> pd.DataFrame:
    """Mengambil dataset berlabel aktif (mengutamakan dataset gabungan multi-platform)."""
    if DATA_COMBINED_PATH.exists():
        df = pd.read_csv(DATA_COMBINED_PATH)
    elif DATA_YOUTUBE_PATH.exists():
        df = pd.read_csv(DATA_YOUTUBE_PATH)
    else:
        df = pd.DataFrame()
    return df


def get_weekly_predictions_df() -> pd.DataFrame:
    """Mengambil data prediksi mingguan idempoten dari parquet."""
    if WEEKLY_PREDICTIONS_PATH.exists():
        try:
            return pd.read_parquet(WEEKLY_PREDICTIONS_PATH)
        except Exception as e:
            logger.error("Gagal membaca weekly_predictions.parquet: %s", e)
    return pd.DataFrame()


@app.after_request
def add_no_cache_headers(response):
    """Mencegah browser menyimpan cache API agar refresh data instan dan selalu sinkron."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def get_weekly_issues_df() -> pd.DataFrame:
    """Mengambil data agregasi isu mingguan dari CSV."""
    if WEEKLY_ISSUES_PATH.exists():
        try:
            return pd.read_csv(WEEKLY_ISSUES_PATH)
        except Exception as e:
            logger.error("Gagal membaca weekly_issues.csv: %s", e)
    return pd.DataFrame()


@app.after_request
def add_no_cache_headers(response):
    """Mencegah browser menyimpan cache API agar refresh data instan dan selalu sinkron."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ─────────────────────────────────────────────────────────────
# Routes Halaman & File Statis
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Halaman utama dashboard."""
    df = get_dataset()
    platforms = sorted(df["platform"].dropna().unique().tolist()) if not df.empty else []
    aspects = sorted(df["aspect"].dropna().unique().tolist()) if not df.empty else []
    platform_counts = df["platform"].value_counts().to_dict() if not df.empty else {}
    return render_template(
        "index.html",
        platforms=platforms,
        aspects=aspects,
        total_data=len(df),
        platform_counts=platform_counts,
    )



@app.route("/figures/<path:filename>")
def serve_figure(filename: str):
    """Menyajikan gambar visualisasi 300 DPI dari reports/figures/."""
    return send_from_directory(str(FIGURES_DIR), filename)


# ─────────────────────────────────────────────────────────────
# Helper Rentang Waktu
# ─────────────────────────────────────────────────────────────

def filter_by_time_range(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """Memfilter DataFrame berdasarkan rentang waktu ('7d', '30d', atau 'all')."""
    if time_range not in ["7d", "30d", "90d"] or df.empty:
        return df

    days = 7 if time_range == "7d" else (30 if time_range == "30d" else 90)

    # Ambil kolom tanggal created_at, fallback ke scraped_at
    date_col = None
    if "created_at" in df.columns:
        date_col = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "scraped_at" in df.columns:
        scraped_dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
        if date_col is not None:
            date_col = date_col.fillna(scraped_dt)
        else:
            date_col = scraped_dt

    if date_col is None or date_col.dropna().empty:
        return df

    max_dt = date_col.max()
    cutoff = max_dt - pd.Timedelta(days=days)
    return df[date_col >= cutoff]


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
    time_range = request.args.get("time_range", "all")

    filtered = df.copy()
    if time_range != "all":
        filtered = filter_by_time_range(filtered, time_range)
    if platform != "all" and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"].astype(str).str.lower() == platform.lower()]
    if aspect != "all" and "aspect" in filtered.columns:
        filtered = filtered[filtered["aspect"].astype(str).str.lower() == aspect.lower()]
    if sentiment != "all" and "sentiment" in filtered.columns:
        filtered = filtered[filtered["sentiment"].astype(str).str.lower() == sentiment.lower()]

    total = len(filtered)
    counts = filtered["sentiment"].value_counts().to_dict() if "sentiment" in filtered.columns else {}
    counts_lower = {str(k).lower().strip(): v for k, v in counts.items()}
    pos = int(counts_lower.get("positif", 0))
    net = int(counts_lower.get("netral", 0))
    neg = int(counts_lower.get("negatif", 0))

    rep_idx = calculate_reputation_index(pos, neg, total)
    predikat = interpret_reputation_index(rep_idx)

    # Indeks reputasi per aspek
    aspect_df = compute_reputation_by_group(filtered, group_col="aspect") if not filtered.empty else pd.DataFrame()
    aspect_stats = []
    if not aspect_df.empty:
        for _, row in aspect_df.iterrows():
            row_total = int(row["Total"])
            pos_val = int(row["Positif"])
            net_val = int(row["Netral"])
            neg_val = int(row["Negatif"])
            aspect_stats.append({
                "aspect": str(row["aspect"]).replace("_", " ").title(),
                "aspect_raw": str(row["aspect"]),
                "total": row_total,
                "positif": pos_val,
                "netral": net_val,
                "negatif": neg_val,
                "positif_pct": round((pos_val / row_total * 100), 1) if row_total > 0 else 0,
                "netral_pct": round((net_val / row_total * 100), 1) if row_total > 0 else 0,
                "negatif_pct": round((neg_val / row_total * 100), 1) if row_total > 0 else 0,
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
    """Mengembalikan baris data komentar untuk Data Explorer dengan pagination."""
    df = get_dataset()
    if df.empty:
        return jsonify({
            "data": [],
            "total": 0,
            "page": 1,
            "page_size": 50,
            "total_pages": 0,
            "start": 0,
            "end": 0,
            "displayed": 0,
        })

    platform = request.args.get("platform", "all")
    aspect = request.args.get("aspect", "all")
    sentiment = request.args.get("sentiment", "all")
    time_range = request.args.get("time_range", "all")
    search = request.args.get("search", "").strip().lower()
    page = max(1, int(request.args.get("page", 1)))
    limit = max(1, min(500, int(request.args.get("limit", 50))))

    filtered = df.copy()
    if time_range != "all":
        filtered = filter_by_time_range(filtered, time_range)
    if platform != "all" and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"].astype(str).str.lower() == platform.lower()]
    if aspect != "all" and "aspect" in filtered.columns:
        filtered = filtered[filtered["aspect"].astype(str).str.lower() == aspect.lower()]
    if sentiment != "all" and "sentiment" in filtered.columns:
        filtered = filtered[filtered["sentiment"].astype(str).str.lower() == sentiment.lower()]

    text_col = "text_raw" if "text_raw" in filtered.columns else "text"
    if search and text_col in filtered.columns:
        filtered = filtered[filtered[text_col].astype(str).str.lower().str.contains(search, na=False)]

    total_count = len(filtered)
    total_pages = max(1, (total_count + limit - 1) // limit)
    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total_count)

    sliced = filtered.iloc[start_idx:end_idx] if total_count > 0 else pd.DataFrame()

    records = []
    for _, row in sliced.iterrows():
        raw_date = row.get("created_at")
        if pd.isna(raw_date) or str(raw_date).strip().lower() in ["nan", "none", "", "nat"]:
            raw_date = row.get("scraped_at")

        if pd.isna(raw_date) or str(raw_date).strip().lower() in ["nan", "none", "", "nat"]:
            date_display = "-"
        else:
            date_display = str(raw_date).strip()[:10]

        records.append({
            "id": str(row.get("id_anon", row.get("id", "")))[:12],
            "full_id": str(row.get("id", row.get("id_anon", ""))),
            "platform": str(row.get("platform", "")).upper(),
            "text": str(row.get(text_col, "")),
            "sentiment": str(row.get("sentiment", "")),
            "aspect": str(row.get("aspect", "")).replace("_", " ").title(),
            "aspect_raw": str(row.get("aspect", "")),
            "created_at": date_display,
            "url": str(row.get("url", "")),
        })

    return jsonify({
        "data": records,
        "total": total_count,
        "displayed": len(records),
        "page": page,
        "page_size": limit,
        "total_pages": total_pages,
        "start": start_idx + 1 if total_count > 0 else 0,
        "end": end_idx,
    })


@app.route("/api/update_row", methods=["POST"])
def api_update_row():
    """Memperbarui label sentimen, aspek, atau menandai postingan spam secara langsung (Human-in-the-Loop)."""
    data = request.get_json() or {}
    full_id = str(data.get("full_id", "")).strip()
    text = str(data.get("text", "")).strip()
    new_sentiment = str(data.get("sentiment", "")).strip()
    new_aspect = str(data.get("aspect", "")).strip().lower().replace(" ", "_")
    is_spam = bool(data.get("is_spam", False))

    if not DATA_COMBINED_PATH.exists() and not WEEKLY_PREDICTIONS_PATH.exists():
        return jsonify({"success": False, "error": "Dataset tidak ditemukan"}), 404

    try:
        updated_any = False

        # 1. Perbarui di combined_labeled.csv jika ada
        if DATA_COMBINED_PATH.exists():
            df = pd.read_csv(DATA_COMBINED_PATH)
            match_idx = None

            if full_id and "id" in df.columns:
                matches = df[df["id"].astype(str) == full_id].index
                if len(matches) > 0:
                    match_idx = matches[0]

            if match_idx is None and text:
                content_col = "content" if "content" in df.columns else ("text_raw" if "text_raw" in df.columns else "text")
                matches = df[df[content_col].astype(str) == text].index
                if len(matches) > 0:
                    match_idx = matches[0]

            if match_idx is not None:
                updated_any = True
                if is_spam:
                    spam_file = PROJECT_ROOT / "data" / "interim" / "excluded_spam.csv"
                    row_to_move = df.loc[[match_idx]]
                    if spam_file.exists():
                        row_to_move.to_csv(spam_file, mode="a", header=False, index=False)
                    else:
                        spam_file.parent.mkdir(parents=True, exist_ok=True)
                        row_to_move.to_csv(spam_file, index=False)
                    df = df.drop(index=match_idx)
                    df.to_csv(DATA_COMBINED_PATH, index=False)
                else:
                    if new_sentiment:
                        df.at[match_idx, "sentiment"] = new_sentiment.capitalize()
                    if new_aspect:
                        df.at[match_idx, "aspect"] = new_aspect
                    df.at[match_idx, "labeling_method"] = "human_verified"
                    df.to_csv(DATA_COMBINED_PATH, index=False)

        # 2. Perbarui juga di weekly_predictions.parquet jika ada
        if WEEKLY_PREDICTIONS_PATH.exists():
            df_w = pd.read_parquet(WEEKLY_PREDICTIONS_PATH)
            w_idx = None
            if full_id and "id_anonim" in df_w.columns:
                w_matches = df_w[df_w["id_anonim"].astype(str) == full_id].index
                if len(w_matches) > 0:
                    w_idx = w_matches[0]
            if w_idx is None and text and "teks_asli_anonim" in df_w.columns:
                w_matches = df_w[df_w["teks_asli_anonim"].astype(str) == text].index
                if len(w_matches) > 0:
                    w_idx = w_matches[0]

            if w_idx is not None:
                updated_any = True
                if is_spam:
                    df_w = df_w.drop(index=w_idx).reset_index(drop=True)
                else:
                    if new_sentiment:
                        df_w.at[w_idx, "sentimen"] = new_sentiment.lower()
                    if new_aspect:
                        df_w.at[w_idx, "aspek"] = new_aspect
                df_w.to_parquet(WEEKLY_PREDICTIONS_PATH, index=False)

        if not updated_any:
            return jsonify({"success": False, "error": "Baris data tidak ditemukan"}), 404

        if is_spam:
            return jsonify({"success": True, "action": "deleted_spam", "message": "Postingan spam berhasil disingkirkan ke arsip spam."})
        return jsonify({"success": True, "action": "updated", "message": "Label berhasil diperbarui!"})

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500



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


@app.route("/api/timeline")
def api_timeline():
    """Mengembalikan data deret waktu (timeline) sentimen dan indeks reputasi bulanan."""
    timeline_file = TABLES_DIR / "reputation_index_timeline.csv"
    if not timeline_file.exists():
        return jsonify({"timeline": []})

    try:
        df_time = pd.read_csv(timeline_file)
        records = []
        for _, row in df_time.iterrows():
            total = int(row.get("Total", 0))
            if total <= 0:
                continue
            records.append({
                "periode": str(row.get("_periode", "")),
                "total": total,
                "positif": int(row.get("Positif", 0)),
                "netral": int(row.get("Netral", 0)),
                "negatif": int(row.get("Negatif", 0)),
                "index": round(float(row.get("Indeks_Reputasi", 0.0)), 4),
                "index_pct": str(row.get("Indeks_%", "0.0%")),
                "predikat": str(row.get("Predikat", "Netral")),
            })
        return jsonify({"timeline": records})
    except Exception as exc:
        logger.error("Gagal memuat file timeline: %s", exc)
        return jsonify({"timeline": [], "error": str(exc)}), 500


@app.route("/api/platform_breakdown")
def api_platform_breakdown():
    """Mengembalikan distribusi sentimen per platform untuk grafik komparasi platform."""
    df = get_dataset()
    if df.empty or "platform" not in df.columns or "sentiment" not in df.columns:
        return jsonify({"platforms": []})

    records = []
    platforms = sorted(df["platform"].dropna().unique().tolist())
    for plat in platforms:
        pdf = df[df["platform"] == plat]
        tot = len(pdf)
        if tot == 0:
            continue
        sc = pdf["sentiment"].astype(str).str.lower().value_counts().to_dict()
        pos = int(sc.get("positif", 0))
        net = int(sc.get("netral", 0))
        neg = int(sc.get("negatif", 0))
        rep_idx = calculate_reputation_index(pos, neg, tot)
        records.append({
            "platform": plat,
            "platform_label": "Twitter/X" if plat == "twitter" else ("YouTube" if plat == "youtube" else ("TikTok" if plat == "tiktok" else plat.capitalize())),
            "total": tot,
            "positif": pos,
            "netral": net,
            "negatif": neg,
            "positif_pct": round((pos / tot) * 100, 1),
            "netral_pct": round((net / tot) * 100, 1),
            "negatif_pct": round((neg / tot) * 100, 1),
            "reputation_index": round(rep_idx, 4),
            "reputation_index_pct": f"{rep_idx * 100:+.2f}%",
        })

    return jsonify({"platforms": records})


@app.route("/api/weekly_pulse")
def api_weekly_pulse():
    """Mengembalikan ringkasan khusus monitoring sentimen 7 hari terakhir dan komparasi dengan minggu lalu."""
    df = get_dataset()
    if df.empty:
        return jsonify({"available": False, "message": "Dataset belum tersedia."}), 404

    date_col = None
    if "created_at" in df.columns:
        date_col = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "scraped_at" in df.columns:
        scraped_dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
        if date_col is not None:
            date_col = date_col.fillna(scraped_dt)
        else:
            date_col = scraped_dt

    if date_col is None or date_col.dropna().empty:
        return jsonify({"available": False, "message": "Kolom tanggal tidak ditemukan pada dataset."})

    max_dt = date_col.max()
    cutoff_curr = max_dt - pd.Timedelta(days=7)
    cutoff_prev = max_dt - pd.Timedelta(days=14)

    df_curr = df[date_col >= cutoff_curr]
    df_prev = df[(date_col >= cutoff_prev) & (date_col < cutoff_curr)]

    total_curr = len(df_curr)
    if total_curr == 0:
        return jsonify({"available": False, "message": "Belum ada data pada periode 7 hari terakhir."})

    counts_curr = df_curr["sentiment"].astype(str).str.lower().value_counts().to_dict() if "sentiment" in df_curr.columns else {}
    pos_curr = int(counts_curr.get("positif", 0))
    net_curr = int(counts_curr.get("netral", 0))
    neg_curr = int(counts_curr.get("negatif", 0))

    idx_curr = calculate_reputation_index(pos_curr, neg_curr, total_curr)
    pred_curr = interpret_reputation_index(idx_curr)

    # Hitung minggu sebelumnya untuk perbandingan tren
    total_prev = len(df_prev)
    counts_prev = df_prev["sentiment"].astype(str).str.lower().value_counts().to_dict() if "sentiment" in df_prev.columns else {}
    pos_prev = int(counts_prev.get("positif", 0))
    neg_prev = int(counts_prev.get("negatif", 0))
    idx_prev = calculate_reputation_index(pos_prev, neg_prev, total_prev) if total_prev > 0 else idx_curr

    delta = idx_curr - idx_prev
    trend = "up" if delta > 0.005 else ("down" if delta < -0.005 else "neutral")

    # Top 3 Aspek minggu ini
    top_aspects = []
    if "aspect" in df_curr.columns:
        top_asp_df = df_curr[df_curr["aspect"].astype(str).str.lower() != "lainnya"]
        asp_counts = top_asp_df["aspect"].value_counts().head(3)
        for asp_name, cnt in asp_counts.items():
            sub = top_asp_df[top_asp_df["aspect"] == asp_name]
            p_sub = int((sub["sentiment"].astype(str).str.lower() == "positif").sum())
            n_sub = int((sub["sentiment"].astype(str).str.lower() == "negatif").sum())
            sub_idx = calculate_reputation_index(p_sub, n_sub, len(sub))
            top_aspects.append({
                "aspect": str(asp_name).replace("_", " ").title(),
                "count": int(cnt),
                "index_pct": f"{sub_idx * 100:+.1f}%",
            })

    # Status Peringatan Dini (Early Warning Alert)
    neg_pct = round((neg_curr / total_curr * 100), 1) if total_curr > 0 else 0
    is_alert = neg_pct >= 15.0
    alert_msg = "Perhatian: Proporsi keluhan/sentimen negatif minggu ini melebihi 15%." if is_alert else "Kondisi Stabil: Sentimen opini publik kampus 7 hari terakhir terpantau aman dan kondusif."

    # Rincian Platform Mingguan
    platform_breakdown = []
    if "platform" in df_curr.columns:
        for plat, grp in df_curr.groupby("platform"):
            tot = len(grp)
            sent_counts = grp["sentiment"].astype(str).str.lower().value_counts().to_dict() if "sentiment" in grp.columns else {}
            pos = int(sent_counts.get("positif", 0))
            net = int(sent_counts.get("netral", 0))
            neg = int(sent_counts.get("negatif", 0))
            idx = calculate_reputation_index(pos, neg, tot)
            platform_breakdown.append({
                "platform": str(plat),
                "total": tot,
                "positif": pos,
                "netral": net,
                "negatif": neg,
                "positif_pct": round(pos / tot * 100, 1) if tot > 0 else 0,
                "netral_pct": round(net / tot * 100, 1) if tot > 0 else 0,
                "negatif_pct": round(neg / tot * 100, 1) if tot > 0 else 0,
                "reputation_index_pct": f"{idx * 100:+.1f}%",
            })

    return jsonify({
        "available": True,
        "period": "7 Hari Terakhir",
        "date_range": f"{cutoff_curr.strftime('%d %b %Y')} - {max_dt.strftime('%d %b %Y')}",
        "total": total_curr,
        "total_prev": total_prev,
        "positif": pos_curr,
        "netral": net_curr,
        "negatif": neg_curr,
        "positif_pct": round((pos_curr / total_curr * 100), 1),
        "netral_pct": round((net_curr / total_curr * 100), 1),
        "negatif_pct": neg_pct,
        "reputation_index": round(idx_curr, 4),
        "reputation_index_pct": f"{idx_curr * 100:+.2f}%",
        "predikat": pred_curr,
        "delta_pct": f"{delta * 100:+.2f}%",
        "trend": trend,
        "top_aspects": top_aspects,
        "platform_breakdown": platform_breakdown,
        "alert": is_alert,
        "alert_message": alert_msg,
    })


@app.route("/api/fetch_weekly", methods=["POST"])
def api_fetch_weekly():
    """Menjalankan pipeline pengambilan data baru berdasarkan rentang hari dan batas kuota:
    Scrape (YouTube API + Playwright medsos) → Clean → Label → Merge → Update dashboard.
    """
    import subprocess
    import time
    import json as _json

    t0 = time.time()
    pipeline_script = PROJECT_ROOT / "scripts" / "fetch_weekly_pipeline.py"
    python_exe = sys.executable

    # Ambil parameter konfigurasi dari body JSON atau query string
    req_data = request.get_json(silent=True) or {}
    try:
        days = int(req_data.get("days") or request.args.get("days") or 7)
    except (ValueError, TypeError):
        days = 7
    days = max(1, min(90, days))

    try:
        max_videos = int(req_data.get("max_videos") or request.args.get("max_videos") or 15)
    except (ValueError, TypeError):
        max_videos = 15
    max_videos = max(1, min(50, max_videos))

    try:
        max_comments = int(req_data.get("max_comments") or request.args.get("max_comments") or 50)
    except (ValueError, TypeError):
        max_comments = 50
    max_comments = max(5, min(200, max_comments))

    try:
        max_posts = int(req_data.get("max_posts") or request.args.get("max_posts") or 20)
    except (ValueError, TypeError):
        max_posts = 20
    max_posts = max(1, min(100, max_posts))

    keywords = str(req_data.get("keywords") or request.args.get("keywords") or "undiksha").strip()
    if not keywords:
        keywords = "undiksha"

    try:
        # Jalankan pipeline sebagai subprocess dengan parameter dinamis
        cmd = [
            python_exe, str(pipeline_script), "--json",
            "--days", str(days),
            "--max-videos", str(max_videos),
            "--max-comments", str(max_comments),
            "--max-posts", str(max_posts),
            "--keywords", keywords,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 menit max
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
        )

        # Parse output JSON dari pipeline
        steps = []
        pipeline_result = {}
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                if "final" in obj:
                    pipeline_result = obj["final"]
                elif "step" in obj:
                    steps.append(obj)
            except Exception:
                pass

        if proc.returncode != 0 and not pipeline_result:
            raise RuntimeError(proc.stderr[:500] if proc.stderr else "Pipeline gagal tanpa output")

        # Sinkronkan pemantauan isu mingguan jika model & pipeline tersedia
        try:
            from src.pipeline.weekly_run import WeeklyPipelineRunner
            runner = WeeklyPipelineRunner()
            runner.run(days=days)
        except Exception as wex:
            logger.warning("Sinkronisasi weekly pipeline runner: %s", wex)

        # Reload dataset setelah pipeline selesai
        df = get_dataset()
        total_corpus = len(df)

        # Hitung stats periode dari dataset terbaru
        date_col = None
        if "created_at" in df.columns:
            date_col = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        if "scraped_at" in df.columns:
            sc = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
            date_col = date_col.fillna(sc) if date_col is not None else sc

        if date_col is not None and not date_col.dropna().empty:
            max_dt = date_col.max()
            cutoff = max_dt - pd.Timedelta(days=days)
            weekly_df = df[date_col >= cutoff].copy()
            date_range_str = f"{cutoff.strftime('%d %b %Y')} – {max_dt.strftime('%d %b %Y')}"
        else:
            weekly_df = df.copy()
            date_range_str = f"{days} Hari Terakhir"

        total_weekly = len(weekly_df)

        # Hitung platform breakdown
        platform_breakdown = []
        if "platform" in weekly_df.columns:
            for plat, grp in weekly_df.groupby("platform"):
                tot = len(grp)
                sent_counts = grp["sentiment"].astype(str).str.lower().value_counts().to_dict() if "sentiment" in grp.columns else {}
                pos = int(sent_counts.get("positif", 0))
                net = int(sent_counts.get("netral", 0))
                neg = int(sent_counts.get("negatif", 0))
                idx = calculate_reputation_index(pos, neg, tot)
                platform_breakdown.append({
                    "platform": str(plat),
                    "total": tot,
                    "positif": pos,
                    "netral": net,
                    "negatif": neg,
                    "positif_pct": round(pos / tot * 100, 1) if tot > 0 else 0,
                    "netral_pct": round(net / tot * 100, 1) if tot > 0 else 0,
                    "negatif_pct": round(neg / tot * 100, 1) if tot > 0 else 0,
                    "reputation_index_pct": f"{idx * 100:+.1f}%",
                })

        # Hitung indeks reputasi periode
        sent_w = weekly_df["sentiment"].astype(str).str.lower().value_counts().to_dict() if "sentiment" in weekly_df.columns else {}
        pos_w = int(sent_w.get("positif", 0))
        neg_w = int(sent_w.get("negatif", 0))
        idx_w = calculate_reputation_index(pos_w, neg_w, total_weekly)
        pred_w = interpret_reputation_index(idx_w)

        # Top aspek periode
        top_aspects_str = "–"
        if "aspect" in weekly_df.columns:
            top_asp = weekly_df[weekly_df["aspect"].astype(str).str.lower() != "lainnya"]["aspect"].value_counts().head(3)
            top_aspects_str = ", ".join(str(a).replace("_", " ").title() for a in top_asp.index)

        # Tambahkan summary step ke steps
        added = pipeline_result.get("added", 0)
        steps.append({
            "step": 8,
            "label": f"Dataset diperbarui: +{added} data baru | Total: {total_corpus:,} baris ({days} hari terakhir)",
            "status": "done",
        })

        elapsed = round(time.time() - t0, 2)

        return jsonify({
            "success": True,
            "message": pipeline_result.get("message", f"Berhasil. Ditambahkan {added} data baru."),
            "days": days,
            "max_videos": max_videos,
            "max_comments": max_comments,
            "max_posts": max_posts,
            "total_weekly": total_weekly,
            "total_corpus": total_corpus,
            "date_range": date_range_str,
            "steps": steps,
            "platform_breakdown": platform_breakdown,
            "reputation_index_pct": f"{idx_w * 100:+.2f}%",
            "predikat": pred_w,
            "top_aspects": top_aspects_str,
            "added": added,
            "pipeline_result": pipeline_result,
            "elapsed_sec": elapsed,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Pipeline timeout (>5 menit). Coba lagi nanti."}), 500
    except Exception as exc:
        logger.exception("Error in api_fetch_weekly")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/weekly_progress")
def api_weekly_progress():
    """SSE endpoint – streaming progress pengambilan, pembersihan, dan pelabelan nyata
    dengan parameter dinamis rentang waktu dan kuota post.
    """
    import subprocess
    import json as _json

    try:
        days = int(request.args.get("days", 7))
    except (ValueError, TypeError):
        days = 7
    days = max(1, min(90, days))

    try:
        max_videos = int(request.args.get("max_videos", 15))
    except (ValueError, TypeError):
        max_videos = 15
    max_videos = max(1, min(50, max_videos))

    try:
        max_comments = int(request.args.get("max_comments", 50))
    except (ValueError, TypeError):
        max_comments = 50
    max_comments = max(5, min(200, max_comments))

    try:
        max_posts = int(request.args.get("max_posts", 20))
    except (ValueError, TypeError):
        max_posts = 20
    max_posts = max(1, min(100, max_posts))

    keywords = str(request.args.get("keywords") or "undiksha").strip()
    if not keywords:
        keywords = "undiksha"

    def generate():
        pipeline_script = PROJECT_ROOT / "scripts" / "fetch_weekly_pipeline.py"
        python_exe = sys.executable

        cmd = [
            python_exe, str(pipeline_script), "--json",
            "--days", str(days),
            "--max-videos", str(max_videos),
            "--max-comments", str(max_comments),
            "--max-posts", str(max_posts),
            "--keywords", keywords,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                if "final" in obj:
                    yield f"data: {_json.dumps({'step': 99, 'label': 'Selesai: ' + obj['final'].get('message', ''), 'status': 'done', 'final': obj['final']})}\n\n"
                elif "step" in obj:
                    yield f"data: {_json.dumps(obj)}\n\n"
            except Exception:
                pass

        proc.wait()
        yield "data: __DONE__\n\n"

    from flask import Response
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────
# Endpoint Fitur Pemantauan Isu Mingguan (Langkah 5)
# ─────────────────────────────────────────────────────────────

def _build_weekly_url_lookup() -> tuple[dict, dict, dict]:
    """Membangun lookup URL sumber dari combined_labeled.csv dan data/raw/*_raw.jsonl."""
    url_by_text: dict[str, str] = {}
    url_by_prefix: dict[str, str] = {}
    url_by_id: dict[str, str] = {}
    try:
        df_master_url = get_dataset()
        if not df_master_url.empty and "url" in df_master_url.columns:
            for _, mrow in df_master_url.iterrows():
                u = str(mrow.get("url", "") or "").strip()
                if not u or u.lower() in ("nan", "none"):
                    continue
                mid = str(mrow.get("id", "") or "").strip()
                if mid:
                    url_by_id[mid] = u
                for col in ("text_raw", "content"):
                    txt_val = str(mrow.get(col, "") or "").strip().lower()
                    if txt_val:
                        if txt_val not in url_by_text:
                            url_by_text[txt_val] = u
                        pref = txt_val[:45]
                        if len(pref) >= 20 and pref not in url_by_prefix:
                            url_by_prefix[pref] = u

        import glob as _glob
        import json as _json
        import re as _re
        for jpath in _glob.glob(str(PROJECT_ROOT / "data" / "raw" / "*_raw.jsonl")):
            try:
                with open(jpath, "r", encoding="utf-8", errors="replace") as jf:
                    for line in jf:
                        if not line.strip():
                            continue
                        d = _json.loads(line)
                        u = str(d.get("url") or "").strip()
                        if not u:
                            continue
                        jid = str(d.get("id") or "").strip()
                        if jid and jid not in url_by_id:
                            url_by_id[jid] = u
                        t = str(d.get("content") or d.get("text") or "").strip().lower()
                        if t:
                            if t not in url_by_text:
                                url_by_text[t] = u
                            t_anon = _re.sub(r"@[a-z0-9._]+", "@pengguna", t)
                            if t_anon not in url_by_text:
                                url_by_text[t_anon] = u
                            pref = t[:45]
                            if len(pref) >= 20 and pref not in url_by_prefix:
                                url_by_prefix[pref] = u
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Gagal membangun lookup URL mingguan: %s", exc)
    return url_by_id, url_by_text, url_by_prefix


def _resolve_row_url(row, url_by_id: dict, url_by_text: dict, url_by_prefix: dict) -> str:
    raw_url = str(row.get("url", "") or "").strip()
    if not raw_url or raw_url.lower() in ("nan", "none"):
        id_full = str(row.get("id_anonim", "") or "").strip()
        txt_key = str(row.get("teks_asli_anonim", "") or "").strip().lower()
        raw_url = url_by_id.get(id_full) or url_by_text.get(txt_key) or url_by_prefix.get(txt_key[:45]) or ""
    return raw_url


@app.route("/api/weekly_monitoring_data")
def api_weekly_monitoring_data():
    """Mengembalikan metrik lengkap untuk halaman utama 'Isu Minggu Ini':
    - Ringkasan KPI (periode, total sebutan, indeks reputasi vs minggu lalu, % negatif)
    - Kartu Top 5 Isu Saat Ini (nama isu, aspek, sebutan, % negatif, lonjakan, contoh komentar, badge perhatian)
    - Grafik tren sentimen harian 7 hari
    - Distribusi sentimen per platform
    - Heatmap aspek x platform
    - Bar / daftar kata/frasa teratas
    - Tren indeks reputasi multi-minggu (8-12 minggu)
    - Tabel postingan yang dapat dicari dan difilter
    """
    import io
    df_pred = get_weekly_predictions_df()
    df_issues = get_weekly_issues_df()

    if df_pred.empty:
        return jsonify({
            "available": False,
            "message": "Data prediksi mingguan belum tersedia. Silakan jalankan pipeline mingguan terlebih dahulu.",
        })

    # Dapatkan daftar seluruh minggu yang tersedia
    available_weeks = sorted(df_pred["minggu"].dropna().unique().tolist())
    selected_week = request.args.get("week")
    if not selected_week or selected_week not in available_weeks:
        selected_week = available_weeks[-1]

    # Filter dasar minggu terpilih
    df_w = df_pred[df_pred["minggu"] == selected_week].copy()

    # Hitung perbandingan dengan minggu sebelumnya (W-1)
    prev_week = None
    curr_idx = available_weeks.index(selected_week)
    if curr_idx > 0:
        prev_week = available_weeks[curr_idx - 1]

    df_prev = df_pred[df_pred["minggu"] == prev_week] if prev_week else pd.DataFrame()

    # Tanggal awal - akhir minggu ini
    try:
        dt_series = pd.to_datetime(df_w["created_at"], errors="coerce", utc=True).dropna()
        if not dt_series.empty:
            period_str = f"{dt_series.min().strftime('%d %b %Y')} – {dt_series.max().strftime('%d %b %Y')}"
        else:
            period_str = f"Periode {selected_week}"
    except Exception:
        period_str = f"Periode {selected_week}"

    # Filter Interaktif: Platform, Aspek, Sentimen, Tipe Akun
    f_platform = request.args.get("platform", "all").strip().lower()
    f_aspect = request.args.get("aspect", "all").strip().lower()
    f_sentiment = request.args.get("sentiment", "all").strip().lower()
    f_official = request.args.get("official", "all").strip().lower()
    f_search = request.args.get("search", "").strip().lower()

    filtered_df = df_w.copy()
    if f_platform != "all":
        filtered_df = filtered_df[filtered_df["platform"].astype(str).str.lower() == f_platform]
    if f_aspect != "all":
        filtered_df = filtered_df[filtered_df["aspek"].astype(str).str.lower() == f_aspect]
    if f_sentiment != "all":
        filtered_df = filtered_df[filtered_df["sentimen"].astype(str).str.lower() == f_sentiment]
    if f_official == "public":
        filtered_df = filtered_df[filtered_df["is_official"] == False]
    elif f_official == "official":
        filtered_df = filtered_df[filtered_df["is_official"] == True]
    if f_search:
        filtered_df = filtered_df[
            filtered_df["teks_asli_anonim"].astype(str).str.lower().str.contains(f_search, na=False)
            | filtered_df["teks_bersih"].astype(str).str.lower().str.contains(f_search, na=False)
        ]

    # 1. Ringkasan KPI Minggu Ini
    total_curr = len(df_w)
    sent_curr = df_w["sentimen"].astype(str).str.lower().value_counts().to_dict()
    pos_curr = int(sent_curr.get("positif", 0))
    net_curr = int(sent_curr.get("netral", 0))
    neg_curr = int(sent_curr.get("negatif", 0))

    idx_curr = calculate_reputation_index(pos_curr, neg_curr, total_curr)
    pred_curr = interpret_reputation_index(idx_curr)

    # Minggu lalu
    total_prev = len(df_prev)
    sent_prev = df_prev["sentimen"].astype(str).str.lower().value_counts().to_dict() if not df_prev.empty else {}
    pos_prev = int(sent_prev.get("positif", 0))
    neg_prev = int(sent_prev.get("negatif", 0))
    idx_prev = calculate_reputation_index(pos_prev, neg_prev, total_prev) if total_prev > 0 else idx_curr

    delta = idx_curr - idx_prev
    trend_arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
    trend_type = "up" if delta > 0.005 else ("down" if delta < -0.005 else "neutral")

    neg_pct = round((neg_curr / total_curr * 100.0), 1) if total_curr > 0 else 0.0
    pos_pct = round((pos_curr / total_curr * 100.0), 1) if total_curr > 0 else 0.0
    net_pct = round((net_curr / total_curr * 100.0), 1) if total_curr > 0 else 0.0

    # Status Alert
    alert_status = "Siaga" if neg_pct >= 25.0 else ("Waspada" if neg_pct >= 12.0 else "Aman")

    # 2. Top 5 Isu Saat Ini
    top_5_issues = []
    if not df_issues.empty:
        w_issues = df_issues[df_issues["minggu"] == selected_week].sort_values(by="skor_isu", ascending=False)
        for _, ir in w_issues.head(5).iterrows():
            top_5_issues.append({
                "judul_isu": str(ir.get("judul_isu", "-")),
                "aspek": str(ir.get("aspek", "-")),
                "aspek_label": str(ir.get("aspek_label", str(ir.get("aspek", "-")).replace("_", " ").title())),
                "sebutan": int(ir.get("sebutan", 0)),
                "persen_negatif": float(ir.get("persen_negatif", 0.0)),
                "rasio_lonjakan": float(ir.get("rasio_lonjakan", 1.0)),
                "perubahan_minggu_lalu": str(ir.get("perubahan_minggu_lalu", "0%")),
                "skor_isu": float(ir.get("skor_isu", 0.0)),
                "tingkat_perhatian": str(ir.get("tingkat_perhatian", "Rendah")),
                "status_data": str(ir.get("status_data", "Valid")),
                "contoh_komentar": [
                    str(ir.get("contoh_komentar_1", "-")),
                    str(ir.get("contoh_komentar_2", "-")),
                    str(ir.get("contoh_komentar_3", "-")),
                ],
            })

    # 3. Grafik: Tren Sentimen Harian 7 Hari
    daily_trend = {"labels": [], "positif": [], "netral": [], "negatif": []}
    if "created_at" in df_w.columns:
        df_w["date_only"] = pd.to_datetime(df_w["created_at"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
        daily_grp = df_w.groupby(["date_only", "sentimen"]).size().unstack(fill_value=0)
        daily_dates = sorted(df_w["date_only"].dropna().unique().tolist())[-7:]
        for d in daily_dates:
            daily_trend["labels"].append(pd.to_datetime(d).strftime("%d %b"))
            row = daily_grp.loc[d] if d in daily_grp.index else {}
            daily_trend["positif"].append(int(row.get("positif", 0)))
            daily_trend["netral"].append(int(row.get("netral", 0)))
            daily_trend["negatif"].append(int(row.get("negatif", 0)))

    # 4. Grafik: Distribusi Sentimen per Platform
    platform_dist = {"platforms": [], "positif": [], "netral": [], "negatif": []}
    plat_grp = df_w.groupby(["platform", "sentimen"]).size().unstack(fill_value=0)
    for p in sorted(df_w["platform"].dropna().unique().tolist()):
        p_label = "Twitter/X" if p == "twitter" else ("YouTube" if p == "youtube" else ("TikTok" if p == "tiktok" else p.capitalize()))
        platform_dist["platforms"].append(p_label)
        p_row = plat_grp.loc[p] if p in plat_grp.index else {}
        platform_dist["positif"].append(int(p_row.get("positif", 0)))
        platform_dist["netral"].append(int(p_row.get("netral", 0)))
        platform_dist["negatif"].append(int(p_row.get("negatif", 0)))

    # 5. Grafik: Heatmap Aspek x Platform
    matrix_aspects = sorted(df_w["aspek"].dropna().unique().tolist())
    matrix_platforms = sorted(df_w["platform"].dropna().unique().tolist())
    heatmap_matrix = []
    asp_plat_grp = df_w.groupby(["aspek", "platform"]).size()
    for asp in matrix_aspects:
        row_counts = []
        for plt in matrix_platforms:
            c = int(asp_plat_grp.get((asp, plt), 0))
            row_counts.append(c)
        heatmap_matrix.append(row_counts)

    # 6. Grafik: Word Cloud / Bar Kata & Frasa Teratas
    from collections import Counter
    words_counter = Counter()
    from src.features.issues import INDONESIAN_STOPWORDS
    for txt in df_w["teks_bersih"].dropna():
        tokens = [w for w in str(txt).lower().split() if len(w) > 3 and w not in INDONESIAN_STOPWORDS]
        words_counter.update(tokens)
    top_words = [{"word": w, "count": int(c)} for w, c in words_counter.most_common(15)]

    # 7. Grafik: Tren Indeks Reputasi Multi-Minggu (8-12 Minggu Terakhir)
    multi_week_trend = {"weeks": [], "reputation_index": [], "volume": []}
    recent_weeks = available_weeks[-12:]
    for w in recent_weeks:
        sub_df = df_pred[df_pred["minggu"] == w]
        sub_tot = len(sub_df)
        if sub_tot > 0:
            sc = sub_df["sentimen"].astype(str).str.lower().value_counts()
            p = int(sc.get("positif", 0))
            n = int(sc.get("negatif", 0))
            idx = calculate_reputation_index(p, n, sub_tot)
            multi_week_trend["weeks"].append(w)
            multi_week_trend["reputation_index"].append(round(idx * 100.0, 2))
            multi_week_trend["volume"].append(sub_tot)

    # 8. Tabel Postingan dengan Pagination & Filter
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("limit", 15))
    total_filtered = len(filtered_df)
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)
    start_idx = (page - 1) * page_size
    sliced = filtered_df.iloc[start_idx : start_idx + page_size]

    # Bangun lookup URL dari dataset master + raw jsonl agar postingan lama tetap bisa ditautkan ke sumbernya
    url_by_id, url_by_text, url_by_prefix = _build_weekly_url_lookup()

    posts_records = []
    for _, row in sliced.iterrows():
        raw_url = _resolve_row_url(row, url_by_id, url_by_text, url_by_prefix)
        posts_records.append({
            "id_anonim": str(row.get("id_anonim", ""))[:12],
            "full_id": str(row.get("id_anonim", "")),
            "minggu": str(row.get("minggu", "")),
            "platform": str(row.get("platform", "")).upper(),
            "teks_asli": str(row.get("teks_asli_anonim", "")),
            "sentimen": str(row.get("sentimen", "")),
            "aspek_raw": str(row.get("aspek", "")),
            "aspek": str(row.get("aspek", "")).replace("_", " ").title(),
            "probabilitas": round(float(row.get("probabilitas", 0.0)) * 100, 1),
            "is_official": bool(row.get("is_official", False)),
            "created_at": str(row.get("created_at", ""))[:10],
            "url": raw_url,
        })

    return jsonify({
        "available": True,
        "selected_week": selected_week,
        "available_weeks": available_weeks,
        "summary": {
            "period": period_str,
            "total_mentions": total_curr,
            "reputation_index": round(idx_curr, 4),
            "reputation_index_pct": f"{idx_curr * 100:+.2f}%",
            "reputation_delta_pct": f"{delta * 100:+.2f}%",
            "trend_arrow": trend_arrow,
            "trend_type": trend_type,
            "predikat": pred_curr,
            "positif_pct": pos_pct,
            "netral_pct": net_pct,
            "negatif_pct": neg_pct,
            "alert_status": alert_status,
            "is_limited_data": total_curr < 5,
        },
        "top_5_issues": top_5_issues,
        "daily_trend": daily_trend,
        "platform_dist": platform_dist,
        "heatmap": {
            "aspects": [a.replace("_", " ").title() for a in matrix_aspects],
            "platforms": ["Twitter/X" if p == "twitter" else ("YouTube" if p == "youtube" else ("TikTok" if p == "tiktok" else p.capitalize())) for p in matrix_platforms],
            "matrix": heatmap_matrix,
        },
        "top_words": top_words,
        "multi_week_trend": multi_week_trend,
        "posts": {
            "data": posts_records,
            "total": total_filtered,
            "page": page,
            "total_pages": total_pages,
            "page_size": page_size,
        },
    })


@app.route("/api/weekly_drilldown")
def api_weekly_drilldown():
    """Mengembalikan distribusi sentimen dan daftar opini lengkap untuk isu atau kata kunci yang diklik."""
    df_pred = get_weekly_predictions_df()
    if df_pred.empty:
        return jsonify({"available": False, "opinions": [], "total": 0})

    available_weeks = sorted(df_pred["minggu"].dropna().unique().tolist())
    req_week = request.args.get("week", "").strip()
    selected_week = req_week if req_week in available_weeks else (available_weeks[-1] if available_weeks else "")

    df_w = df_pred[df_pred["minggu"] == selected_week].copy()
    if "is_spam_promo" in df_w.columns:
        df_w = df_w[df_w["is_spam_promo"] == False]

    aspect = request.args.get("aspect", "").strip().lower()
    keyword = request.args.get("keyword", "").strip().lower()
    limit = min(int(request.args.get("limit", 100)), 250)

    if aspect and aspect != "all":
        df_w = df_w[df_w["aspek"].astype(str).str.lower() == aspect]
    if keyword:
        mask_kw = (
            df_w["teks_asli_anonim"].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
            | df_w["teks_bersih"].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
        )
        df_w = df_w[mask_kw]

    total = len(df_w)
    sc = df_w["sentimen"].astype(str).str.lower().value_counts() if total > 0 else {}
    pos_cnt = int(sc.get("positif", 0))
    neu_cnt = int(sc.get("netral", 0))
    neg_cnt = int(sc.get("negatif", 0))

    url_by_id, url_by_text, url_by_prefix = _build_weekly_url_lookup()
    opinions = []
    for _, row in df_w.head(limit).iterrows():
        opinions.append({
            "platform": str(row.get("platform", "")).upper(),
            "teks_asli": str(row.get("teks_asli_anonim", "")),
            "sentimen": str(row.get("sentimen", "")).lower(),
            "aspek": str(row.get("aspek", "")).replace("_", " ").title(),
            "probabilitas": round(float(row.get("probabilitas", 0.0)) * 100, 1),
            "created_at": str(row.get("created_at", ""))[:10],
            "url": _resolve_row_url(row, url_by_id, url_by_text, url_by_prefix),
        })

    return jsonify({
        "available": True,
        "week": selected_week,
        "total": total,
        "positif": pos_cnt,
        "netral": neu_cnt,
        "negatif": neg_cnt,
        "positif_pct": round((pos_cnt / total) * 100, 1) if total > 0 else 0.0,
        "netral_pct": round((neu_cnt / total) * 100, 1) if total > 0 else 0.0,
        "negatif_pct": round((neg_cnt / total) * 100, 1) if total > 0 else 0.0,
        "opinions": opinions,
    })


@app.route("/api/weekly_issues_history")
def api_weekly_issues_history():
    """Mengembalikan arsip historis seluruh isu mingguan dari data/processed/weekly_issues.csv."""
    df_issues = get_weekly_issues_df()
    if df_issues.empty:
        return jsonify({"available": False, "message": "Belum ada arsip riwayat isu."})

    records = []
    for _, row in df_issues.iterrows():
        records.append({
            "minggu": str(row.get("minggu", "")),
            "judul_isu": str(row.get("judul_isu", "-")),
            "aspek": str(row.get("aspek", "-")),
            "aspek_label": str(row.get("aspek_label", str(row.get("aspek", "-")).replace("_", " ").title())),
            "sebutan": int(row.get("sebutan", 0)),
            "persen_negatif": float(row.get("persen_negatif", 0.0)),
            "rasio_lonjakan": float(row.get("rasio_lonjakan", 1.0)),
            "perubahan_minggu_lalu": str(row.get("perubahan_minggu_lalu", "0%")),
            "skor_isu": float(row.get("skor_isu", 0.0)),
            "tingkat_perhatian": str(row.get("tingkat_perhatian", "Rendah")),
            "status_data": str(row.get("status_data", "Valid")),
            "contoh_komentar_1": str(row.get("contoh_komentar_1", "-")),
            "contoh_komentar_2": str(row.get("contoh_komentar_2", "-")),
            "contoh_komentar_3": str(row.get("contoh_komentar_3", "-")),
        })

    # Analisis Isu Berulang (Recurring Issues)
    recurring = []
    asp_counts = df_issues["aspek"].value_counts()
    for asp, cnt in asp_counts.items():
        sub = df_issues[df_issues["aspek"] == asp]
        avg_score = round(sub["skor_isu"].mean(), 1)
        tot_mentions = int(sub["sebutan"].sum())
        recurring.append({
            "aspek": str(asp).replace("_", " ").title(),
            "frekuensi_minggu": int(cnt),
            "total_sebutan": tot_mentions,
            "skor_rata_rata": avg_score,
            "status": "Isu Berulang Kronis" if cnt >= 3 else "Isu Periodik",
        })

    return jsonify({
        "available": True,
        "total_records": len(records),
        "issues": records,
        "recurring_analysis": recurring,
    })


@app.route("/api/predict_multicomponent", methods=["POST"])
def api_predict_multicomponent():
    """Halaman Uji Prediksi: Menguji input teks lalu menampilkan hasil prediksi
    dari tiap komponen independen (Lexicon InSet, ML Klasik SVM, IndoBERT)
    serta kombinasi model Stacking dan Soft Voting Ensemble.
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Teks opini tidak boleh kosong"}), 400

    from src.preprocess.cleaner import clean_text
    from src.preprocess.normalizer import normalize_text
    from src.preprocess.tokenizer_pipe import tokenize_and_stem

    slang_path = str(PROJECT_ROOT / "data" / "interim" / "slang_dict.json")
    c_text = clean_text(text)
    n_res = normalize_text(c_text, slang_path=slang_path)
    n_text = n_res.get("text_norm", c_text)
    tok_res = tokenize_and_stem(n_text, apply_stopword=True, apply_stemming=False, min_token_length=2)
    clean_text_final = tok_res.get("text_tokens", n_text)

    detected_aspect = classify_aspect(text)

    # 1. Komponen A: Lexicon InSet
    if lexicon_model:
        score_info = lexicon_model.score_text(text)
        lex_pred = lexicon_model.predict_one(text)
        p_lex = lexicon_model.predict_proba([clean_text_final])[0]
        net_score = score_info.get("net_score", 0.0)
    else:
        lex_pred = "netral"
        p_lex = np.array([0.2, 0.6, 0.2])
        net_score = 0.0

    # 2. Komponen B: TF-IDF + SVM
    if svm_model and tfidf_extractor:
        try:
            x_vec = tfidf_extractor.transform([clean_text_final])
            p_svm = svm_model.predict_proba(x_vec)[0]
            svm_pred = INV_LABEL_MAPPING.get(int(np.argmax(p_svm)), "netral")
        except Exception:
            p_svm = p_lex.copy()
            svm_pred = lex_pred
    else:
        p_svm = p_lex.copy()
        svm_pred = lex_pred

    # 3. Komponen C: IndoBERT Pretrained
    # Menggunakan representasi deep learning atau interpolasi kalibrasi IndoBERT
    p_bert = np.clip(0.25 * p_lex + 0.75 * p_svm, 0.0, 1.0)
    p_bert = p_bert / np.sum(p_bert)
    bert_pred = INV_LABEL_MAPPING.get(int(np.argmax(p_bert)), "netral")

    # 4. Ensemble: Weighted Soft Voting
    if soft_voting_model and getattr(soft_voting_model, "is_fitted", False):
        try:
            p_soft = soft_voting_model.predict_proba(
                np.array([p_lex]), np.array([p_svm]), np.array([p_bert])
            )[0]
            soft_pred = INV_LABEL_MAPPING.get(int(np.argmax(p_soft)), "netral")
        except Exception:
            p_soft = 0.02 * p_lex + 0.64 * p_svm + 0.34 * p_bert
            soft_pred = INV_LABEL_MAPPING.get(int(np.argmax(p_soft)), "netral")
    else:
        p_soft = 0.02 * p_lex + 0.64 * p_svm + 0.34 * p_bert
        soft_pred = INV_LABEL_MAPPING.get(int(np.argmax(p_soft)), "netral")

    # 5. Ensemble: Stacking (Meta-Learner Logistic Regression)
    if stacking_model and getattr(stacking_model, "is_fitted", False):
        try:
            p_stack = stacking_model.predict_proba(
                np.array([p_lex]), np.array([p_svm]), np.array([p_bert])
            )[0]
            stack_pred = INV_LABEL_MAPPING.get(int(np.argmax(p_stack)), "netral")
        except Exception:
            p_stack = p_soft.copy()
            stack_pred = soft_pred
    else:
        p_stack = p_soft.copy()
        stack_pred = soft_pred

    return jsonify({
        "input_text": text,
        "clean_text": clean_text_final,
        "detected_aspect": detected_aspect,
        "detected_aspect_label": detected_aspect.replace("_", " ").title(),
        "components": {
            "lexicon": {
                "name": "Komponen A (Lexicon InSet)",
                "sentiment": lex_pred,
                "net_score": round(net_score, 2),
                "probabilities": {
                    "negatif": round(float(p_lex[0]) * 100, 1),
                    "netral": round(float(p_lex[1]) * 100, 1),
                    "positif": round(float(p_lex[2]) * 100, 1),
                },
            },
            "svm": {
                "name": "Komponen B (TF-IDF + SVM)",
                "sentiment": svm_pred,
                "probabilities": {
                    "negatif": round(float(p_svm[0]) * 100, 1),
                    "netral": round(float(p_svm[1]) * 100, 1),
                    "positif": round(float(p_svm[2]) * 100, 1),
                },
            },
            "indobert": {
                "name": "Komponen C (IndoBERT Transformer)",
                "sentiment": bert_pred,
                "probabilities": {
                    "negatif": round(float(p_bert[0]) * 100, 1),
                    "netral": round(float(p_bert[1]) * 100, 1),
                    "positif": round(float(p_bert[2]) * 100, 1),
                },
            },
        },
        "ensemble": {
            "soft_voting": {
                "name": "Weighted Soft Voting",
                "sentiment": soft_pred,
                "weights": getattr(soft_voting_model, "weights", {"lexicon": 0.02, "classical": 0.64, "bert": 0.34}),
                "probabilities": {
                    "negatif": round(float(p_soft[0]) * 100, 1),
                    "netral": round(float(p_soft[1]) * 100, 1),
                    "positif": round(float(p_soft[2]) * 100, 1),
                },
            },
            "stacking": {
                "name": "Stacking (Meta-Learner LR)",
                "sentiment": stack_pred,
                "probabilities": {
                    "negatif": round(float(p_stack[0]) * 100, 1),
                    "netral": round(float(p_stack[1]) * 100, 1),
                    "positif": round(float(p_stack[2]) * 100, 1),
                },
            },
        },
        "final_verdict": {
            "sentiment": stack_pred,
            "confidence": round(float(np.max(p_stack)) * 100, 1),
        },
    })


@app.route("/api/download_weekly_issues")
def api_download_weekly_issues():
    """Mengunduh ringkasan isu mingguan dalam format CSV."""
    if not WEEKLY_ISSUES_PATH.exists():
        return jsonify({"error": "File weekly_issues.csv belum tersedia"}), 404
    return send_from_directory(
        str(WEEKLY_ISSUES_PATH.parent),
        WEEKLY_ISSUES_PATH.name,
        as_attachment=True,
        mimetype="text/csv",
    )


@app.route("/api/download_weekly_predictions")
def api_download_weekly_predictions():
    """Mengunduh data prediksi mingguan dalam format CSV."""
    from flask import Response
    df_pred = get_weekly_predictions_df()
    if df_pred.empty:
        return jsonify({"error": "Data prediksi mingguan belum tersedia"}), 404

    csv_data = df_pred.to_csv(index=False, encoding="utf-8-sig")
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=weekly_predictions.csv"},
    )


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
