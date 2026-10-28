"""
src/label/annotation_tool.py
============================
Utilitas untuk pembuatan template anotasi sentimen dan aspek reputasi,
sampling data per platform, dan pembagian ke multi-annotator.

Penggunaan:
    python src/label/annotation_tool.py --platform youtube --sample-size 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)

# Kolom-kolom standar untuk template anotasi manual
ANNOTATION_COLUMNS = [
    "id_anon",
    "platform",
    "text_raw",
    "text_clean",
    "sentiment",    # Diisi oleh annotator: positif / netral / negatif
    "aspect",       # Diisi: akademik / biaya / fasilitas / layanan / kemahasiswaan / karier / citra_umum / lainnya
    "notes",        # Catatan khusus (misal: sarkasme, typo, code-mixing)
]


def create_annotation_sample(
    df_processed: pd.DataFrame,
    sample_size: int = 200,
    seed: int = 42,
    stratify_by_length: bool = True,
) -> pd.DataFrame:
    """Mengambil sampel representatif dari dataset yang telah diproses untuk dianotasi.

    Args:
        df_processed: DataFrame dari data/processed/ (sudah dibersihkan).
        sample_size: Jumlah baris sampel yang diinginkan.
        seed: Random seed untuk reproducibility.
        stratify_by_length: Jika True, sampel mencakup variasi panjang teks.

    Returns:
        DataFrame sampel dengan kolom-kolom standar anotasi.
    """
    if len(df_processed) == 0:
        raise ValueError("DataFrame input kosong.")

    # Filter out baris yang teks bersihnya kosong
    valid_df = df_processed[df_processed["text_clean"].str.strip().ne("")].copy()
    if len(valid_df) == 0:
        valid_df = df_processed.copy()

    actual_sample_size = min(sample_size, len(valid_df))

    if stratify_by_length and len(valid_df) >= actual_sample_size:
        # Buat kategori panjang teks untuk sampel berstrata
        valid_df["_char_len"] = valid_df["text_clean"].str.len()
        valid_df["_len_bin"] = pd.qcut(
            valid_df["_char_len"], q=min(3, len(valid_df)), labels=False, duplicates="drop"
        )
        # Gunakan include_groups=False untuk kompatibilitas pandas terbaru
        sample_df = (
            valid_df.groupby("_len_bin", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, int(actual_sample_size * len(g) / len(valid_df))),
                    random_state=seed,
                ),
                include_groups=False,
            )
            .head(actual_sample_size)
            .copy()
        )
        valid_df.drop(columns=["_char_len", "_len_bin"], inplace=True, errors="ignore")
    else:
        sample_df = valid_df.sample(n=actual_sample_size, random_state=seed).copy()

    # Siapkan kolom template
    for col in ANNOTATION_COLUMNS:
        if col not in sample_df.columns:
            sample_df[col] = ""

    return sample_df[ANNOTATION_COLUMNS].reset_index(drop=True)


def export_double_annotation_templates(
    sample_df: pd.DataFrame,
    output_dir: str | Path,
    platform: str,
    annotator_names: list[str] | None = None,
) -> tuple[Path, Path]:
    """Mengekspor dua file template anotasi independen untuk double-annotation (Annotator A & B).

    Args:
        sample_df: DataFrame sampel yang siap dianotasi.
        output_dir: Direktori tujuan ekspor (data/labeled/).
        platform: Nama platform media sosial.
        annotator_names: Daftar 2 nama annotator (default: ['annotator_a', 'annotator_b']).

    Returns:
        Tuple path file (path_a, path_b).
    """
    if annotator_names is None:
        annotator_names = ["annotator_a", "annotator_b"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kosongkan kolom sentimen dan aspek agar annotator mengisi dari nol
    template_df = sample_df.copy()
    template_df["sentiment"] = ""
    template_df["aspect"] = ""
    template_df["notes"] = ""

    path_a = out_dir / f"{platform}_annotation_{annotator_names[0]}.csv"
    path_b = out_dir / f"{platform}_annotation_{annotator_names[1]}.csv"

    template_df.to_csv(path_a, index=False, encoding="utf-8-sig")
    template_df.to_csv(path_b, index=False, encoding="utf-8-sig")

    logger.info("Template anotasi diekspor:")
    logger.info("  Annotator 1: %s (%d baris)", path_a, len(template_df))
    logger.info("  Annotator 2: %s (%d baris)", path_b, len(template_df))

    return path_a, path_b


def main() -> None:
    """Entry point CLI untuk sampling anotasi."""
    parser = argparse.ArgumentParser(description="Siapkan template anotasi untuk platform tertentu.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path config.yaml")
    parser.add_argument("--platform", default="youtube", help="Nama platform (youtube/instagram/twitter/tiktok)")
    parser.add_argument("--sample-size", type=int, default=200, help="Jumlah baris sampel untuk anotasi")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    processed_path = Path(cfg["paths"]["processed_dir"]) / f"{args.platform}_processed.csv"
    if not processed_path.exists():
        logger.error("File processed tidak ditemukan: %s", processed_path)
        return

    df_proc = pd.read_csv(processed_path)
    sample_df = create_annotation_sample(
        df_proc, sample_size=args.sample_size, seed=cfg.get("seed", 42)
    )

    out_dir = Path(cfg["paths"]["labeled_dir"])
    export_double_annotation_templates(sample_df, out_dir, args.platform)
    logger.info("Selesai membuat template anotasi.")


if __name__ == "__main__":
    main()
