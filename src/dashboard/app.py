"""
src/dashboard/app.py
====================
Dashboard Interaktif Pemantauan Reputasi Kampus Undiksha (PRD §6 F12).

Fitur Utama:
1. Ringkasan Eksekutif & Indeks Reputasi (Positif, Netral, Negatif).
2. Analisis Sentimen Lintas 7 Aspek Reputasi Undiksha.
3. Galeri Visualisasi Publikasi (Word Cloud & Grafik Perbandingan Model).
4. Live Sentiment Predictor (Uji coba teks opini baru secara instan).
5. Data Explorer & Unduh Laporan CSV.

Penggunaan:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dashboard.components import get_sentiment_badge_html, render_metric_card
from src.evaluate.aspect import classify_aspect, compute_aspect_sentiment_breakdown
from src.evaluate.reputation_index import (
    calculate_reputation_index,
    compute_reputation_by_group,
    interpret_reputation_index,
)
from src.features.tfidf_vectorizer import TextFeatureExtractor
from src.models.lexicon import InSetLexiconModel
from src.utils import load_config

# ─────────────────────────────────────────────────────────────
# Konfigurasi Halaman Streamlit
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Reputasi Undiksha | Hybrid Ensemble",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk tampilan modern
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3799;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 25px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cache Data & Model Loader
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_labeled_data() -> pd.DataFrame:
    comb_path = Path("data/labeled/combined_labeled.csv")
    yt_path = Path("data/labeled/youtube_labeled.csv")
    if comb_path.exists():
        df = pd.read_csv(comb_path)
    elif yt_path.exists():
        df = pd.read_csv(yt_path)
    else:
        df = pd.DataFrame()
    return df


@st.cache_resource
def load_trained_models():
    models = {}
    lexicon_path = Path("data/interim/inset_lexicon.tsv")
    if lexicon_path.exists():
        models["lexicon"] = InSetLexiconModel(lexicon_path)

    svm_path = Path("models/svm.joblib")
    if svm_path.exists():
        models["svm"] = joblib.load(svm_path)

    return models


# ─────────────────────────────────────────────────────────────
# Header Aplikasi
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎓 Analisis Sentimen & Reputasi Undiksha</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Monitoring Opini Publik Media Sosial Menggunakan Hybrid Ensemble (Lexicon + ML + IndoBERT)</div>', unsafe_allow_html=True)

df_all = load_labeled_data()
models = load_trained_models()

if df_all.empty:
    st.error("Data berlabel tidak ditemukan di `data/labeled/youtube_labeled.csv`. Jalankan pipeline terlebih dahulu.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Sidebar Filter
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Logo_Undiksha.png/300px-Logo_Undiksha.png", width=90)
    st.header("🔍 Filter Data")

    platforms = ["Semua Platform"] + sorted(df_all["platform"].dropna().unique().tolist())
    selected_platform = st.selectbox("Platform Media Sosial", platforms)

    aspects = ["Semua Aspek"] + sorted(df_all["aspect"].dropna().unique().tolist())
    selected_aspect = st.selectbox("Aspek Reputasi", aspects)

    sentiments = ["Semua Sentimen", "positif", "netral", "negatif"]
    selected_sentiment = st.selectbox("Sentimen Opini", sentiments)

    official_filter = st.radio("Tipe Akun", ["Semua Akun", "Hanya Akun Publik", "Hanya Akun Resmi"])

# Terapkan Filter
filtered_df = df_all.copy()
if selected_platform != "Semua Platform":
    filtered_df = filtered_df[filtered_df["platform"] == selected_platform]
if selected_aspect != "Semua Aspek":
    filtered_df = filtered_df[filtered_df["aspect"] == selected_aspect]
if selected_sentiment != "Semua Sentimen":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]
if official_filter == "Hanya Akun Publik":
    filtered_df = filtered_df[filtered_df["is_official"] == False]
elif official_filter == "Hanya Akun Resmi":
    filtered_df = filtered_df[filtered_df["is_official"] == True]

# ─────────────────────────────────────────────────────────────
# Tab Navigasi Dashboard
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ringkasan & Indeks Reputasi",
    "🎯 Analisis Aspek & Word Cloud",
    "⚡ Live Sentiment Predictor",
    "📁 Data Explorer",
])

# ─── TAB 1: RINGKASAN & INDEKS REPUTASI ───
with tab1:
    total_data = len(filtered_df)
    n_pos = int((filtered_df["sentiment"] == "positif").sum())
    n_net = int((filtered_df["sentiment"] == "netral").sum())
    n_neg = int((filtered_df["sentiment"] == "negatif").sum())

    rep_index = calculate_reputation_index(n_pos, n_neg, total_data)
    predikat = interpret_reputation_index(rep_index)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Komentar", f"{total_data:,}")
    with col2:
        st.metric("Indeks Reputasi", f"{rep_index * 100:+.1f}%", predikat)
    with col3:
        st.metric("Sentimen Positif", f"{n_pos:,}", f"{n_pos / max(1, total_data) * 100:.1f}%")
    with col4:
        st.metric("Sentimen Netral", f"{n_net:,}", f"{n_net / max(1, total_data) * 100:.1f}%")
    with col5:
        st.metric("Sentimen Negatif", f"{n_neg:,}", f"{n_neg / max(1, total_data) * 100:.1f}%", delta_color="inverse")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("Distribusi Proporsi Sentimen")
        sentiment_counts = pd.Series({"Positif": n_pos, "Netral": n_net, "Negatif": n_neg})
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        )
        st.pyplot(fig)

    with col_chart2:
        st.subheader("Peringkat Indeks Reputasi per Aspek")
        rep_aspect_df = compute_reputation_by_group(filtered_df, group_col="aspect")
        st.dataframe(
            rep_aspect_df[["aspect", "Total", "Positif", "Netral", "Negatif", "Indeks_%", "Predikat"]],
            use_container_width=True,
            hide_index=True,
        )

# ─── TAB 2: ANALISIS ASPEK & WORD CLOUD ───
with tab2:
    st.subheader("Komposisi Sentimen Berdasarkan Aspek Layanan Undiksha")
    fig_aspect_path = Path("reports/figures/aspect_sentiment_breakdown.png")
    if fig_aspect_path.exists():
        st.image(str(fig_aspect_path), use_container_width=True)
    else:
        ct = compute_aspect_sentiment_breakdown(filtered_df)
        st.dataframe(ct, use_container_width=True)

    st.markdown("---")
    st.subheader("Analisis Kata Kunci Opini (Word Cloud)")
    col_wc1, col_wc2, col_wc3 = st.columns(3)

    with col_wc1:
        st.markdown("**Sentimen Positif (Apresiasi)**")
        p_pos = Path("reports/figures/wordcloud_positif.png")
        if p_pos.exists():
            st.image(str(p_pos), use_container_width=True)

    with col_wc2:
        st.markdown("**Sentimen Netral (Informasi)**")
        p_net = Path("reports/figures/wordcloud_netral.png")
        if p_net.exists():
            st.image(str(p_net), use_container_width=True)

    with col_wc3:
        st.markdown("**Sentimen Negatif (Keluhan & Kendala)**")
        p_neg = Path("reports/figures/wordcloud_negatif.png")
        if p_neg.exists():
            st.image(str(p_neg), use_container_width=True)

# ─── TAB 3: LIVE SENTIMENT PREDICTOR ───
with tab3:
    st.subheader("⚡ Prediksi Sentimen & Aspek Teks Opini Baru (Real-Time)")
    st.write("Masukkan kalimat opini media sosial untuk menguji klasifikasi secara instan:")

    input_text = st.text_area(
        "Ketik teks opini atau pertanyaan seputar Undiksha:",
        value="Kampus Undiksha sekarang semakin keren fasilitas lab FTK lengkap dan dosennya sangat suportif!",
        height=100,
    )

    if st.button("Analisis Sentimen", type="primary"):
        if input_text.strip():
            detected_aspect = classify_aspect(input_text)

            # Prediksi Lexicon
            lex_model = models.get("lexicon")
            if lex_model:
                scores = lex_model.score_text(input_text)
                lex_pred = lex_model.predict_one(input_text)
                lex_probs = lex_model.predict_proba([input_text])[0]
            else:
                lex_pred = "N/A"
                lex_probs = [0.33, 0.33, 0.34]

            # Prediksi Hasil
            st.markdown("### Hasil Analisis Opini")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"**Prediksi Sentimen**: {get_sentiment_badge_html(lex_pred)}", unsafe_allow_html=True)
                st.markdown(f"**Kategori Aspek Terdeteksi**: `{detected_aspect.upper()}`")
            with res_col2:
                st.markdown(f"**Skor Polaritas Bersih**: `{scores['net_score'] if lex_model else 0.0:+.2f}`")

            st.markdown("#### Estimasi Probabilitas Kelas (Softmax)")
            prob_df = pd.DataFrame({
                "Kelas Sentimen": ["Negatif", "Netral", "Positif"],
                "Probabilitas (%)": [lex_probs[0] * 100, lex_probs[1] * 100, lex_probs[2] * 100],
            })
            st.bar_chart(prob_df.set_index("Kelas Sentimen"), color="#3498db")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")

# ─── TAB 4: DATA EXPLORER ───
with tab4:
    st.subheader("Eksplorasi Data Komentar & Ekspor")
    st.write(f"Menampilkan **{len(filtered_df):,}** baris komentar berdasarkan filter aktif:")

    display_cols = ["id_anon", "platform", "text_raw", "sentiment", "aspect", "is_official"]
    available_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(filtered_df[available_cols], use_container_width=True, height=400)

    csv_data = filtered_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 Unduh Data Terfilter (.csv)",
        data=csv_data,
        file_name="undiksha_sentiment_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Dikembangkan untuk Program Magang 20 SKS UPT TIK Undiksha | Berbasis Python & Streamlit")
