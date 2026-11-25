"""
Modul Visualisasi Hasil Analisis Sentimen
Menghasilkan grafik beresolusi tinggi (300 DPI) untuk kebutuhan laporan magang, publikasi,
dan presentasi: Distribusi Sentimen, Perbandingan Platform, WordCloud, dan Confusion Matrix.
"""

import os
import re
from collections import Counter
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Mode non-GUI untuk kestabilan eksekusi skrip
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from rich.console import Console
from rich.table import Table

from ..utils.logger import logger


class SentimentVisualizer:
    """Kelas untuk membuat grafik dan visualisasi hasil analisis sentimen."""

    # Palet warna harmonis: Positif (Emerald Green), Netral (Amber Gold), Negatif (Crimson Red)
    COLOR_MAP = {
        "Positif": "#10b981",
        "Netral": "#f59e0b",
        "Negatif": "#ef4444"
    }

    # Stopwords tambahan khusus media sosial & kampus untuk WordCloud
    INDONESIAN_STOPWORDS = {
        "dan", "yang", "di", "ini", "itu", "untuk", "dari", "ke", "pada", "adalah",
        "dengan", "akan", "juga", "atau", "oleh", "saya", "aku", "kamu", "dia", "mereka",
        "kita", "kami", "anda", "ya", "iya", "dong", "sih", "deh", "kan", "nih", "tuh",
        "kok", "ada", "bisa", "sudah", "lagi", "mau", "hanya", "saja", "karena", "kalau",
        "jadi", "seperti", "lebih", "tapi", "namun", "tetapi", "undiksha", "kampus", "link",
        "https", "http", "com", "www", "post", "video", "reels", "foto", "yg", "dgn", "utk",
        "dr", "sy", "ak", "km", "bgt", "gak", "ga", "nggak", "tak", "bkn", "tdk", "aja"
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        vis_cfg = self.config.get("sentiment_analysis", {}).get("visualization", {})
        
        self.output_dir = vis_cfg.get("figures_dir", "reports/figures")
        self.dpi = vis_cfg.get("dpi", 300)
        os.makedirs(self.output_dir, exist_ok=True)
        self.console = Console()

        # Atur gaya visual default
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    def plot_distribution(self, df: pd.DataFrame) -> str:
        """
        Membuat grafik 2-panel: Donut Chart dan Bar Chart distribusi sentimen.
        """
        counts = df["sentiment"].value_counts()
        labels = [lbl for lbl in ["Positif", "Netral", "Negatif"] if lbl in counts]
        values = [counts[lbl] for lbl in labels]
        colors = [self.COLOR_MAP[lbl] for lbl in labels]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Panel 1: Donut Chart
        wedges, texts, autotexts = ax1.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
            textprops=dict(color="#1f2937", fontsize=11, weight="bold")
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(11)
            at.set_weight("bold")
        ax1.set_title("Proporsi Sentimen (%)", fontsize=13, weight="bold", pad=15)

        # Panel 2: Bar Chart
        bars = ax2.bar(labels, values, color=colors, width=0.55, edgecolor="#d1d5db", linewidth=1.2)
        ax2.set_title("Jumlah Data per Kelas Sentimen", fontsize=13, weight="bold", pad=15)
        ax2.set_ylabel("Jumlah Komentar / Post", fontsize=11)
        ax2.set_ylim(0, max(values) * 1.2 if values else 10)

        # Label angka di atas setiap bar
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=11, weight="bold"
            )

        plt.suptitle("Analisis Distribusi Sentimen Opini Publik Undiksha", fontsize=15, weight="bold", y=0.98)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "sentiment_distribution.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"[VISUALISASI] Grafik distribusi tersimpan di: {output_path}")
        return output_path

    def plot_platform_comparison(self, df: pd.DataFrame) -> str:
        """
        Membuat grafik perbandingan sentimen antar platform media sosial.
        """
        if "platform" not in df.columns:
            logger.warning("[VISUALISASI] Kolom 'platform' tidak ditemukan untuk perbandingan.")
            return ""

        # Grouping platform x sentiment
        cross = pd.crosstab(df["platform"], df["sentiment"])
        
        # Urutkan kolom sentimen
        ordered_cols = [c for c in ["Positif", "Netral", "Negatif"] if c in cross.columns]
        cross = cross[ordered_cols]
        colors = [self.COLOR_MAP[c] for c in ordered_cols]

        fig, ax = plt.subplots(figsize=(10, 6))
        cross.plot(kind="bar", stacked=False, color=colors, ax=ax, width=0.7, edgecolor="#d1d5db")

        ax.set_title("Perbandingan Sentimen Menurut Platform Media Sosial", fontsize=14, weight="bold", pad=15)
        ax.set_xlabel("Platform", fontsize=11, weight="bold")
        ax.set_ylabel("Jumlah Konten", fontsize=11, weight="bold")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Sentimen", title_fontsize="10", fontsize=10)

        # Beri label angka di atas setiap bar
        for p in ax.patches:
            val = int(p.get_height())
            if val > 0:
                ax.annotate(
                    f"{val}",
                    (p.get_x() + p.get_width() / 2., val),
                    ha="center", va="center",
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=9, weight="bold"
                )

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "sentiment_by_platform.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"[VISUALISASI] Grafik platform tersimpan di: {output_path}")
        return output_path

    def plot_wordclouds(self, df: pd.DataFrame) -> str:
        """
        Membuat 2 WordCloud berdampingan: Sentimen Positif vs Negatif.
        """
        text_col = "clean_content" if "clean_content" in df.columns else "content"

        pos_texts = df[df["sentiment"] == "Positif"][text_col].dropna().astype(str).tolist()
        neg_texts = df[df["sentiment"] == "Negatif"][text_col].dropna().astype(str).tolist()

        def clean_words(texts):
            words = []
            for t in texts:
                for w in re.findall(r"\b[a-zA-Z]{3,}\b", t.lower()):
                    if w not in self.INDONESIAN_STOPWORDS:
                        words.append(w)
            return " ".join(words)

        pos_str = clean_words(pos_texts) or "semangat bangga hebat prestasi sukses keren"
        neg_str = clean_words(neg_texts) or "kecewa lambat jelek sulit rugi parah"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Wordcloud Positif
        wc_pos = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap="Greens",
            max_words=60,
            collocations=False
        ).generate(pos_str)
        ax1.imshow(wc_pos, interpolation="bilinear")
        ax1.axis("off")
        ax1.set_title("Kata Kunci Dominan: Sentimen POSITIF", fontsize=13, weight="bold", pad=10, color="#10b981")

        # Wordcloud Negatif
        wc_neg = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap="Reds",
            max_words=60,
            collocations=False
        ).generate(neg_str)
        ax2.imshow(wc_neg, interpolation="bilinear")
        ax2.axis("off")
        ax2.set_title("Kata Kunci Dominan: Sentimen NEGATIF", fontsize=13, weight="bold", pad=10, color="#ef4444")

        plt.suptitle("Word Cloud Kata Sentimen Mahasiswa & Publik Undiksha", fontsize=15, weight="bold", y=0.98)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "sentiment_wordclouds.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"[VISUALISASI] WordCloud tersimpan di: {output_path}")
        return output_path

    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str]) -> str:
        """
        Membuat visualisasi heatmap Confusion Matrix evaluasi model Machine Learning.
        """
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"}
        )

        ax.set_title("Confusion Matrix Evaluasi Model Sentimen", fontsize=13, weight="bold", pad=12)
        ax.set_xlabel("Sentimen Prediksi (Predicted)", fontsize=11, weight="bold", labelpad=10)
        ax.set_ylabel("Sentimen Aktual (Ground Truth)", fontsize=11, weight="bold", labelpad=10)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"[VISUALISASI] Heatmap Confusion Matrix tersimpan di: {output_path}")
        return output_path

    def generate_all(
        self,
        df: pd.DataFrame,
        cm: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Menjalankan seluruh fungsi pembuatan visualisasi dan menampilkan ringkasan.
        """
        generated = {}
        generated["distribution"] = self.plot_distribution(df)
        generated["platform"] = self.plot_platform_comparison(df)
        generated["wordclouds"] = self.plot_wordclouds(df)

        if cm is not None and labels is not None:
            generated["confusion_matrix"] = self.plot_confusion_matrix(cm, labels)

        # Cetak tabel laporan file yang terbentuk
        table = Table(title="=== Hasil Pembuatan Visualisasi Sentimen ===", show_header=True, header_style="bold green")
        table.add_column("Grafik", justify="left")
        table.add_column("Lokasi File", justify="left")

        for key, path in generated.items():
            if path:
                table.add_row(key.replace("_", " ").title(), path)

        self.console.print(table)
        return generated
