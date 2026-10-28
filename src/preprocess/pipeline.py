"""
src/preprocess/pipeline.py
==========================
Orkestrasi pipeline preprocessing end-to-end.

Alur lengkap (sesuai project-rules):
    1. lowercase
    2. hapus URL
    3. hapus mention (@username)
    4. normalisasi hashtag (#topik → topik)
    5. emoji → teks
    6. normalisasi slang (kamus JSON)
    7. hapus noise (karakter berulang, whitespace)
    8. stopword Sastrawi
    9. stemming Sastrawi

Output per baris:
    - text_raw      : Teks asli (tidak pernah diubah)
    - text_clean    : Setelah tahap 1-7 (sebelum stopword/stem)
    - text_tokens   : Setelah stopword removal
    - text_stemmed  : Setelah stemming (input ke model TF-IDF/ML)
    - emoji_replaced: Flag True jika ada emoji
    - has_codemix   : Flag True jika ada kata Inggris
    - en_words      : Kata Inggris yang ditemukan (JSON list)

Penggunaan:
    python src/preprocess/pipeline.py
    python src/preprocess/pipeline.py --platform youtube
    python src/preprocess/pipeline.py --input data/raw/youtube_raw_20261025.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, set_seed, setup_logging
from src.preprocess.cleaner import clean_text
from src.preprocess.normalizer import normalize_text
from src.preprocess.tokenizer_pipe import tokenize_and_stem

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Pemrosesan Satu Baris
# ─────────────────────────────────────────────────────────────

def preprocess_row(
    text_raw: str,
    slang_path: str,
    apply_stopword: bool,
    apply_stemming: bool,
    min_token_length: int,
    oov_collector: list[str] | None = None,
) -> dict[str, Any]:
    """Menjalankan pipeline preprocessing pada satu teks.

    Args:
        text_raw         : Teks asli dari dataset.
        slang_path       : Path kamus slang.
        apply_stopword   : Apakah stopword removal diterapkan.
        apply_stemming   : Apakah stemming diterapkan.
        min_token_length : Panjang minimum token.
        oov_collector    : Pengumpul kata OOV (opsional).

    Returns:
        Dict kolom-kolom preprocessing.
    """
    # Jaga teks asli
    if not isinstance(text_raw, str) or not text_raw.strip():
        return {
            "text_raw":      text_raw,
            "text_clean":    "",
            "text_tokens":   "",
            "text_stemmed":  "",
            "emoji_replaced": False,
            "has_codemix":   False,
            "en_words":      "[]",
        }

    # --- Tahap 1-4 & 7: Cleaning ---
    text_cleaned = clean_text(text_raw)

    # --- Tahap 5-6: Normalisasi ---
    norm_result = normalize_text(
        text_cleaned,
        slang_path=slang_path,
        oov_collector=oov_collector,
    )
    text_norm     = norm_result["text_norm"]
    emoji_replaced = norm_result["emoji_replaced"]
    has_codemix   = norm_result["has_codemix"]
    en_words      = norm_result["en_words"]

    # --- Tahap 8-9: Tokenisasi, Stopword, Stemming ---
    tok_result = tokenize_and_stem(
        text_norm,
        apply_stopword=apply_stopword,
        apply_stemming=apply_stemming,
        min_token_length=min_token_length,
    )

    return {
        "text_raw":      text_raw,
        "text_clean":    text_norm,         # Teks bersih (pre-stem)
        "text_tokens":   tok_result["text_tokens"],
        "text_stemmed":  tok_result["text_stemmed"],
        "emoji_replaced": emoji_replaced,
        "has_codemix":   has_codemix,
        "en_words":      json.dumps(en_words, ensure_ascii=False),
    }


# ─────────────────────────────────────────────────────────────
# Pipeline untuk DataFrame
# ─────────────────────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
    text_col: str = "text",
) -> pd.DataFrame:
    """Menjalankan preprocessing pada seluruh DataFrame.

    Kolom ``text_raw``, ``text_clean``, ``text_tokens``, ``text_stemmed``
    ditambahkan ke DataFrame; kolom asli lainnya tetap dipertahankan.

    Args:
        df      : DataFrame input dengan kolom ``text_col``.
        config  : Dict konfigurasi dari load_config().
        text_col: Nama kolom teks input.

    Returns:
        DataFrame dengan kolom preprocessing tambahan.
    """
    slang_path        = config["paths"]["slang_dict"]
    apply_stopword    = config["preprocessing"]["remove_stopwords"]
    apply_stemming    = config["preprocessing"]["apply_stemming"]
    min_token_length  = config["preprocessing"]["min_token_length"]

    # Kolom text_raw = backup teks asli
    if "text_raw" not in df.columns:
        df = df.copy()
        df["text_raw"] = df[text_col].astype(str)

    oov_words: list[str] = []
    results: list[dict[str, Any]] = []

    logger.info("Memulai preprocessing: %d baris...", len(df))

    for text in tqdm(df["text_raw"], desc="Preprocessing", unit="baris"):
        row_result = preprocess_row(
            text_raw=text,
            slang_path=slang_path,
            apply_stopword=apply_stopword,
            apply_stemming=apply_stemming,
            min_token_length=min_token_length,
            oov_collector=oov_words,
        )
        results.append(row_result)

    # Gabungkan hasil ke DataFrame
    result_df = pd.DataFrame(results)
    new_cols = ["text_clean", "text_tokens", "text_stemmed",
                "emoji_replaced", "has_codemix", "en_words"]

    # Tambahkan kolom baru; jaga urutan kolom asli di depan
    for col in new_cols:
        df = df.copy()
        df[col] = result_df[col].values

    # Laporan OOV
    oov_unique = sorted(set(oov_words))
    logger.info(
        "Preprocessing selesai. OOV unik ditemukan: %d kata.", len(oov_unique)
    )
    if oov_unique:
        logger.debug("Contoh OOV (20 pertama): %s", oov_unique[:20])

    return df, oov_unique


# ─────────────────────────────────────────────────────────────
# Simpan Hasil
# ─────────────────────────────────────────────────────────────

def save_processed(
    df: pd.DataFrame,
    platform: str,
    processed_dir: str | Path,
) -> Path:
    """Menyimpan DataFrame hasil preprocessing ke data/processed/.

    Args:
        df           : DataFrame hasil preprocessing.
        platform     : Nama platform.
        processed_dir: Direktori output.

    Returns:
        Path file yang disimpan.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{platform}_processed.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Data processed disimpan: %s (%d baris)", out_path, len(df))
    return out_path


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point CLI untuk pipeline preprocessing."""
    parser = argparse.ArgumentParser(
        description="Jalankan pipeline preprocessing pada data mentah."
    )
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Nama platform (youtube/instagram/twitter/tiktok). "
             "Jika tidak diisi, proses semua file di data/raw/.",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path file CSV input spesifik (override --platform).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    set_seed(config["seed"])

    raw_dir       = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])

    # Kumpulkan file yang akan diproses
    if args.input:
        files_to_process = [Path(args.input)]
    elif args.platform:
        pattern = f"{args.platform}_raw_*.csv"
        files_to_process = sorted(raw_dir.glob(pattern))
        if not files_to_process:
            logger.error("Tidak ada file ditemukan: %s/%s", raw_dir, pattern)
            sys.exit(1)
        # Ambil file terbaru saja
        files_to_process = [files_to_process[-1]]
    else:
        # Proses semua platform
        files_to_process = sorted(raw_dir.glob("*_raw_*.csv"))

    if not files_to_process:
        logger.error(
            "Tidak ada file raw CSV di %s. "
            "Jalankan tahap pengumpulan data terlebih dahulu.", raw_dir
        )
        sys.exit(1)

    for raw_file in files_to_process:
        logger.info("Memproses: %s", raw_file)
        df = pd.read_csv(raw_file, dtype=str, encoding="utf-8-sig")
        logger.info("  Baris dibaca: %d", len(df))

        # Ekstrak nama platform dari nama file (mis. youtube_raw_20261025.csv)
        platform = raw_file.stem.split("_raw_")[0]

        df_processed, oov = preprocess_dataframe(df, config)
        save_processed(df_processed, platform, processed_dir)

        # Simpan OOV ke interim untuk analisis
        if oov:
            oov_path = Path(config["paths"]["interim_dir"]) / f"{platform}_oov.json"
            oov_path.parent.mkdir(parents=True, exist_ok=True)
            with oov_path.open("w", encoding="utf-8") as f:
                json.dump(sorted(oov), f, ensure_ascii=False, indent=2)
            logger.info("  OOV disimpan: %s (%d kata)", oov_path, len(oov))

    logger.info("Semua file selesai diproses.")


if __name__ == "__main__":
    main()
