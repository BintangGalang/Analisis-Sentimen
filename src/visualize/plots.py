"""
src/visualize/plots.py
======================
Modul visualisasi data dan hasil evaluasi untuk Laporan Magang (PRD §6 F11, F16).

Karakteristik Visualisasi Akademis:
- Resolusi tinggi (300 DPI, format PNG).
- Tipografi baku Bahasa Indonesia dengan label sumbu dan legenda jelas.
- Palet warna konsisten:
    * Positif : #2ecc71 (Hijau Zamrud)
    * Netral  : #95a5a6 (Abu-abu Slate)
    * Negatif : #e74c3c (Merah Crimson)

Grafik yang Dihasilkan:
1. sentiment_distribution.png     : Donut chart & Bar chart proporsi sentimen.
2. aspect_sentiment_breakdown.png : 100% Stacked bar chart komposisi sentimen per aspek.
3. reputation_index_aspect.png    : Horizontal bar chart Indeks Reputasi per aspek.
4. model_comparison_chart.png     : Grouped bar chart performa seluruh model (CV 5-Fold).
5. confusion_matrix_ensemble.png  : Heatmap Confusion Matrix normalisasi dari Ensemble.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config

logger = logging.getLogger(__name__)

# Warna tematik konsisten
COLOR_PALETTE = {
    "positif": "#2ecc71",
    "netral":  "#95a5a6",
    "negatif": "#e74c3c",
    "primary": "#3498db",
    "dark":    "#2c3e50",
}


def setup_plot_style() -> None:
    """Mengatur konfigurasi visual matplotlib agar seragam dan berstandar publikasi."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#bdc3c7"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#ecf0f1"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_sentiment_distribution(
    df: pd.DataFrame,
    output_path: str | Path,
    sentiment_col: str = "sentiment",
) -> Path:
    """Membuat visualisasi Donut Chart dan Bar Chart distribusi kelas sentimen."""
    setup_plot_style()
    counts = df[sentiment_col].value_counts()
    labels = ["positif", "netral", "negatif"]
    vals = [counts.get(lb, 0) for lb in labels]
    colors = [COLOR_PALETTE[lb] for lb in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Donut Chart
    wedges, texts, autotexts = ax1.pie(
        vals,
        labels=[lb.capitalize() for lb in labels],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_weight("bold")
    ax1.set_title("Proporsi Kelas Sentimen", fontsize=13, weight="bold", pad=10)

    # 2. Bar Chart
    bars = ax2.bar(
        [lb.capitalize() for lb in labels],
        vals,
        color=colors,
        edgecolor="#2c3e50",
        alpha=0.88,
        width=0.55,
    )
    for bar in bars:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + max(vals) * 0.02,
            f"{yval:,} ({yval / sum(vals) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    ax2.set_ylabel("Jumlah Komentar", fontsize=11)
    ax2.set_title("Distribusi Frekuensi Sentimen", fontsize=13, weight="bold", pad=10)
    ax2.set_ylim(0, max(vals) * 1.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Analisis Distribusi Sentimen Publik terhadap Undiksha", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Grafik disimpan: %s", out_file)
    return out_file


def plot_aspect_sentiment_breakdown(
    df: pd.DataFrame,
    output_path: str | Path,
    aspect_col: str = "aspect",
    sentiment_col: str = "sentiment",
) -> Path:
    """Membuat 100% Stacked Bar Chart komposisi sentimen untuk setiap aspek reputasi."""
    setup_plot_style()
    ct = pd.crosstab(df[aspect_col], df[sentiment_col], normalize="index") * 100

    # Urutkan aspek berdasarkan jumlah data total
    aspect_counts = df[aspect_col].value_counts()
    ordered_aspects = aspect_counts.index.tolist()
    ct = ct.reindex(ordered_aspects)

    for col in ["positif", "netral", "negatif"]:
        if col not in ct.columns:
            ct[col] = 0.0

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    bottom = np.zeros(len(ct))
    for sent, color in [("positif", COLOR_PALETTE["positif"]), ("netral", COLOR_PALETTE["netral"]), ("negatif", COLOR_PALETTE["negatif"])]:
        values = ct[sent].values
        ax.barh(
            ct.index,
            values,
            left=bottom,
            color=color,
            edgecolor="white",
            label=sent.capitalize(),
            alpha=0.9,
            height=0.6,
        )
        for i, val in enumerate(values):
            if val >= 5.0:  # Hanya tampilkan label jika proporsi cukup lebar
                ax.text(
                    bottom[i] + val / 2,
                    i,
                    f"{val:.1f}%",
                    va="center",
                    ha="center",
                    fontsize=9,
                    weight="bold",
                    color="white" if sent != "netral" else "#2c3e50",
                )
        bottom += values

    ax.set_xlabel("Persentase Komentar (%)", fontsize=11)
    ax.set_ylabel("Aspek Reputasi Undiksha", fontsize=11)
    ax.set_title("Komposisi Sentimen per Aspek Reputasi Undiksha", fontsize=13, weight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Grafik disimpan: %s", out_file)
    return out_file


def plot_reputation_index_aspect(
    df_aspect_reputation: pd.DataFrame | str | Path,
    output_path: str | Path,
) -> Path:
    """Membuat Horizontal Bar Chart nilai Indeks Reputasi per Aspek (-1.0 s.d. +1.0)."""
    setup_plot_style()
    if isinstance(df_aspect_reputation, (str, Path)):
        df_aspect_reputation = pd.read_csv(df_aspect_reputation)

    df_sorted = df_aspect_reputation.sort_values(by="Indeks_Reputasi", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    indices = df_sorted["Indeks_Reputasi"].values
    y_pos = np.arange(len(df_sorted))
    colors = [COLOR_PALETTE["positif"] if val >= 0 else COLOR_PALETTE["negatif"] for val in indices]

    bars = ax.barh(y_pos, indices, color=colors, edgecolor="#2c3e50", alpha=0.88, height=0.55)

    for i, bar in enumerate(bars):
        val = bar.get_width()
        x_text = val + (0.02 if val >= 0 else -0.06)
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f} ({df_sorted['Total'].iloc[i]} komentar)",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9.5,
            weight="bold",
        )

    ax.axvline(0, color="black", linewidth=1.0, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(a).replace("_", " ").title() for a in df_sorted["aspect"]])
    ax.set_xlabel("Nilai Indeks Reputasi: (Positif - Negatif) / Total", fontsize=11)
    ax.set_xlim(-0.3, max(0.6, max(indices) * 1.25))
    ax.set_title("Indeks Reputasi Berdasarkan Aspek Layanan & Fasilitas Undiksha", fontsize=13, weight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Grafik disimpan: %s", out_file)
    return out_file


def plot_model_comparison(
    model_comparison_csv: str | Path,
    output_path: str | Path,
) -> Path:
    """Membuat Grouped Bar Chart perbandingan Accuracy & F1-Macro seluruh model dari CSV."""
    setup_plot_style()
    df_comp = pd.read_csv(model_comparison_csv)

    models_clean = [
        str(m).replace("Komponen ", "").replace("Ensemble: ", "").replace(" (ML Klasik)", "")
        for m in df_comp["Model"]
    ]
    acc_vals = [float(str(a).split("%")[0]) for a in df_comp["Accuracy"]]
    f1_vals = [float(str(f).split("%")[0]) for f in df_comp["F1-Macro"]]

    x = np.arange(len(models_clean))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    b1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy (%)", color="#3498db", edgecolor="#2c3e50", alpha=0.88)
    b2 = ax.bar(x + width / 2, f1_vals, width, label="F1-Macro (%)", color="#2ecc71", edgecolor="#2c3e50", alpha=0.88)

    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", fontsize=9)
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", fontsize=9, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models_clean, rotation=12, ha="right", fontsize=10)
    ax.set_ylabel("Skor Performa (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Perbandingan Kinerja Komponen Tunggal vs Hybrid Ensemble (Stratified 5-Fold CV)", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Grafik disimpan: %s", out_file)
    return out_file


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    output_path: str | Path,
    model_name: str = "Hybrid Ensemble",
) -> Path:
    """Membuat Heatmap Confusion Matrix persentase normalisasi dan jumlah absolut."""
    setup_plot_style()
    from sklearn.metrics import confusion_matrix

    labels = ["negatif", "netral", "positif"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100

    annot_matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)")
        annot_matrix.append(row)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=[lb.capitalize() for lb in labels],
        yticklabels=[lb.capitalize() for lb in labels],
        ax=ax,
        linewidths=1.0,
        linecolor="white",
    )
    ax.set_xlabel("Prediksi Model", fontsize=11, weight="bold")
    ax.set_ylabel("Label Aktual (Ground Truth)", fontsize=11, weight="bold")
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, weight="bold", pad=12)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Grafik disimpan: %s", out_file)
    return out_file


def main() -> None:
    """CLI untuk menggenerasi seluruh grafik visualisasi publikasi."""
    parser = argparse.ArgumentParser(description="Generate grafik visualisasi publikasi.")
    parser.add_argument("--data", default="data/labeled/combined_labeled.csv")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--tables-dir", default="reports/tables")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    fig_dir = Path(args.figures_dir)
    tbl_dir = Path(args.tables_dir)

    print("Membuat grafik publikasi...")
    plot_sentiment_distribution(df, fig_dir / "sentiment_distribution.png")
    plot_aspect_sentiment_breakdown(df, fig_dir / "aspect_sentiment_breakdown.png")
    
    rep_csv = tbl_dir / "reputation_index_aspect.csv"
    if rep_csv.exists():
        plot_reputation_index_aspect(rep_csv, fig_dir / "reputation_index_aspect.png")

    comp_csv = tbl_dir / "model_comparison.csv"
    if comp_csv.exists():
        plot_model_comparison(comp_csv, fig_dir / "model_comparison_chart.png")

    print("Semua grafik berhasil disimpan di", fig_dir)


if __name__ == "__main__":
    main()
