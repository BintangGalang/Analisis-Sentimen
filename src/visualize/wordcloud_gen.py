"""
src/visualize/wordcloud_gen.py
==============================
Pembuat visualisasi Word Cloud per kelas sentimen (PRD §6 F11, F16).

Karakteristik:
- Filter stopword Bahasa Indonesia + kata pengisi umum.
- Colormap tematik per sentimen:
    * Positif: Greens / Summer
    * Netral : Blues / Bone
    * Negatif: Reds / Autumn
- Resolusi tinggi 300 DPI disimpan di reports/figures/.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config

logger = logging.getLogger(__name__)

# Stopwords tambahan untuk visualisasi kata kunci yang bersih
ADDITIONAL_STOPWORDS = {
    "yang", "di", "dan", "ke", "dari", "ini", "itu", "dengan", "untuk", "dalam",
    "juga", "sudah", "tidak", "saya", "kamu", "dia", "mereka", "kita", "akan",
    "bisa", "ada", "karena", "tetapi", "tapi", "namun", "lagi", "mau", "pada",
    "oleh", "saat", "setelah", "sebelum", "jadi", "pas", "ya", "iya", "aja",
    "saja", "nih", "tuh", "deh", "dong", "kan", "kok", "nya", "ku", "mu",
    "yg", "dgn", "utk", "dlm", "jg", "sdh", "tdk", "sy", "km", "bgt", "bgtu",
    "undiksha", "kampus", "universitas", "singaraja", "bali",  # Filter nama entitas agar kata opini dominan
}


def clean_text_for_wordcloud(texts: list[str]) -> str:
    """Membersihkan kumpulan teks dari karakter non-alfanumerik dan stopwords untuk input Word Cloud."""
    combined = " ".join([str(t) for t in texts if pd.notna(t)])
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", combined.lower())
    filtered = [w for w in tokens if w not in ADDITIONAL_STOPWORDS]
    return " ".join(filtered)


def generate_single_wordcloud(
    text: str,
    output_path: str | Path,
    colormap: str = "viridis",
    title: str = "Word Cloud",
    width: int = 1000,
    height: int = 600,
) -> Path:
    """Menghasilkan satu file gambar Word Cloud beresolusi tinggi."""
    if not text.strip():
        text = "tidak ada data kata"

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap=colormap,
        max_words=100,
        random_state=42,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=12)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Word Cloud disimpan: %s", out_file)
    return out_file


def generate_all_sentiment_wordclouds(
    df: pd.DataFrame,
    figures_dir: str | Path = "reports/figures",
    text_col: str = "text_clean",
    sentiment_col: str = "sentiment",
) -> dict[str, Path]:
    """Menghasilkan Word Cloud untuk setiap kelas sentimen (Positif, Netral, Negatif)."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "positif": {"cmap": "Greens", "title": "Word Cloud: Sentimen Positif (Apresiasi & Harapan)"},
        "netral":  {"cmap": "Blues",  "title": "Word Cloud: Sentimen Netral (Informasi & Pertanyaan)"},
        "negatif": {"cmap": "Reds",   "title": "Word Cloud: Sentimen Negatif (Keluhan & Kendala)"},
    }

    generated_paths: dict[str, Path] = {}

    for sent, cfg in configs.items():
        subset = df[df[sentiment_col] == sent]
        clean_corpus = clean_text_for_wordcloud(subset[text_col].tolist())
        out_path = fig_dir / f"wordcloud_{sent}.png"
        generate_single_wordcloud(
            clean_corpus,
            out_path,
            colormap=cfg["cmap"],
            title=cfg["title"],
        )
        generated_paths[sent] = out_path

    # Buat figure komposit (3 word cloud dalam 1 gambar untuk laporan)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    for idx, (sent, cfg) in enumerate(configs.items()):
        subset = df[df[sentiment_col] == sent]
        corpus = clean_text_for_wordcloud(subset[text_col].tolist())
        if not corpus.strip():
            corpus = "data kosong"
        wc = WordCloud(
            width=600,
            height=450,
            background_color="white",
            colormap=cfg["cmap"],
            max_words=80,
            random_state=42,
            collocations=False,
        ).generate(corpus)
        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(cfg["title"], fontsize=11, weight="bold", pad=8)

    plt.suptitle("Analisis Kata Kunci Opini Berdasarkan Sentimen Media Sosial", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()
    composite_path = fig_dir / "wordcloud_combined.png"
    plt.savefig(composite_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_paths["combined"] = composite_path
    logger.info("Komposit Word Cloud disimpan: %s", composite_path)

    return generated_paths


def main() -> None:
    """CLI untuk membuat word cloud seluruh sentimen."""
    parser = argparse.ArgumentParser(description="Buat Word Cloud per sentimen.")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    paths = generate_all_sentiment_wordclouds(df)
    print("\n" + "=" * 60)
    print("  WORD CLOUD BERHASIL DIBUAT (300 DPI)")
    print("=" * 60)
    for sent, path in paths.items():
        print(f"  • {sent.capitalize():<10}: {path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
