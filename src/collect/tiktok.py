"""
src/collect/tiktok.py
=====================
Pengumpulan komentar TikTok REAL untuk analisis sentimen Undiksha.

Mendukung 2 metode:
1. **Manual / Export Import** (`--export-file`):
   Memuat data komentar TikTok dari file ekspor (CSV, JSON, JSONL, Excel).
   Direkomendasikan karena bot-protection Cloudflare/Captcha TikTok memblokir
   otomatisasi tanpa residential proxy.
2. **Scraper Otomatis** (`--collect`):
   Menggunakan TikTokApi + Playwright Chromium dengan msToken.

Penggunaan:
    # Memuat dari file export/dataset:
    python src/collect/tiktok.py --export-file data/raw/tiktok_export.csv

    # Mencoba scraper otomatis:
    python src/collect/tiktok.py --collect --ms-token YOUR_MS_TOKEN --videos 15
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.collect.utils_collect import (
    STANDARD_COLUMNS,
    build_standard_row,
    save_raw,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

PLATFORM = "tiktok"

# Hashtag TikTok terkait Undiksha
HASHTAGS = [
    "undiksha",
    "kampusundiksha",
    "undikshasingaraja",
    "universitaspendidikanganesha",
    "mahasiswaundiksha",
    "pmbundiksha",
]


def load_from_export(
    export_path: str | Path,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Memuat data komentar TikTok dari file ekspor CSV, JSON, JSONL, atau Excel.

    Args:
        export_path: Path ke file ekspor komentar TikTok.
        raw_dir: Direktori penyimpanan data mentah.

    Returns:
        DataFrame berformat STANDARD_COLUMNS.
    """
    export_path = Path(export_path)
    logger.info("Memuat data TikTok dari ekspor: %s", export_path)

    if not export_path.exists():
        raise FileNotFoundError(f"File ekspor tidak ditemukan: {export_path}")

    suffix = export_path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        raw_df = pd.read_csv(export_path, sep=sep, dtype=str, encoding="utf-8-sig")
    elif suffix == ".json":
        raw_df = pd.read_json(export_path, dtype=str)
    elif suffix == ".jsonl":
        raw_df = pd.read_json(export_path, lines=True, dtype=str)
    elif suffix in (".xlsx", ".xls"):
        raw_df = pd.read_excel(export_path, dtype=str)
    else:
        raise ValueError(f"Format {suffix} tidak didukung. Gunakan .csv, .json, .jsonl, atau .xlsx")

    logger.info("Total baris mentah dibaca: %d", len(raw_df))

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for i, rec in raw_df.iterrows():
        # Fleksibel deteksi teks komentar
        raw_text = (
            rec.get("text")
            or rec.get("content")
            or rec.get("comment")
            or rec.get("komentar")
            or rec.get("caption")
            or ""
        )
        text = str(raw_text).strip()
        if not text or text.lower() == "nan" or len(text) < 3:
            continue

        text = text.encode("utf-8", errors="replace").decode("utf-8")

        username = str(
            rec.get("username")
            or rec.get("author")
            or rec.get("user")
            or rec.get("nickname")
            or f"tiktok_user_{i}"
        )
        created_at = (
            rec.get("created_at")
            or rec.get("timestamp")
            or rec.get("date")
            or datetime.now(timezone.utc).isoformat()
        )
        raw_id = str(rec.get("id") or rec.get("cid") or rec.get("comment_id") or f"tiktok_{i}")
        url = str(rec.get("url") or rec.get("video_url") or "https://www.tiktok.com/@undiksha.bali")

        id_anon = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        if id_anon in seen_ids:
            continue
        seen_ids.add(id_anon)

        row = build_standard_row(
            raw_id=raw_id,
            platform=PLATFORM,
            text=text,
            created_at=created_at,
            username=username,
            url=url,
        )
        rows.append(row)

    if not rows:
        logger.warning("Tidak ada komentar valid yang diekstrak dari file.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id_anon", "text"]).reset_index(drop=True)
    logger.info("Total %d komentar unik valid diekstrak.", len(df))

    saved_path = save_raw(df, platform=PLATFORM, raw_dir=raw_dir)
    logger.info("Berhasil disimpan ke %s", saved_path)
    return df


async def collect_tiktok_comments_async(
    n_videos_per_tag: int = 10,
    n_comments_per_video: int = 30,
    output_dir: str | Path = "data/raw",
    ms_token: str | None = None,
) -> pd.DataFrame:
    """Mengumpulkan komentar TikTok real untuk hashtag terkait Undiksha via Playwright."""
    try:
        from TikTokApi import TikTokApi
    except ImportError:
        raise ImportError(
            "TikTokApi belum diinstall. Jalankan:\n"
            "  pip install TikTokApi playwright\n"
            "  playwright install chromium"
        )

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    async with TikTokApi() as api:
        await api.create_sessions(
            ms_tokens=[ms_token] if ms_token else [],
            num_sessions=1,
            sleep_after=3,
            headless=True,
        )

        for tag in HASHTAGS:
            logger.info("Mencari video TikTok untuk hashtag: #%s", tag)
            video_count = 0
            try:
                async for video in api.hashtag(name=tag).videos(count=n_videos_per_tag):
                    video_id = str(video.id)
                    if hasattr(video, "author") and video.author and hasattr(video.author, "username"):
                        video_url = f"https://www.tiktok.com/@{video.author.username}/video/{video_id}"
                    else:
                        video_url = f"https://www.tiktok.com/video/{video_id}"

                    logger.info("  Video %s — ambil komentar...", video_id)
                    comment_count = 0
                    try:
                        async for comment in api.video(id=video_id).comments(count=n_comments_per_video):
                            cid = str(comment.id)
                            if cid in seen_ids:
                                continue
                            seen_ids.add(cid)

                            text = comment.text if hasattr(comment, "text") else str(comment)
                            if not text or len(str(text).strip()) < 3:
                                continue

                            create_time = getattr(comment, "create_time", None)
                            if create_time:
                                created_at = datetime.fromtimestamp(
                                    int(create_time), tz=timezone.utc
                                ).isoformat()
                            else:
                                created_at = datetime.now(timezone.utc).isoformat()

                            row = build_standard_row(
                                raw_id=cid,
                                platform=PLATFORM,
                                text=str(text).strip(),
                                created_at=created_at,
                                username=getattr(comment.author, "username", "") if hasattr(comment, "author") else "",
                                url=video_url,
                            )
                            all_rows.append(row)
                            comment_count += 1

                    except Exception as e:
                        logger.warning("    Gagal ambil komentar video %s: %s", video_id, e)

                    logger.info("    -> %d komentar dari video %s", comment_count, video_id)
                    video_count += 1

            except Exception as e:
                logger.warning("  Error hashtag #%s: %s", tag, e)

            logger.info("  -> Total %d video diproses untuk #%s", video_count, tag)

    if not all_rows:
        logger.warning(
            "TikTok memblokir permintaan otomatis (anti-bot challenge). "
            "Gunakan opsi --export-file untuk memuat file ekspor komentar TikTok."
        )
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(all_rows, columns=STANDARD_COLUMNS)
    df = df.drop_duplicates(subset=["id_anon"])

    saved_path = save_raw(df, platform=PLATFORM, raw_dir=output_dir)
    logger.info("Total %d komentar unik disimpan ke %s", len(df), saved_path)
    return df


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Pengumpulan Komentar TikTok Real Undiksha.")
    parser.add_argument("--collect", action="store_true", help="Kumpulkan komentar TikTok secara otomatis.")
    parser.add_argument("--export-file", type=str, default=None, help="Muat data dari file ekspor (CSV/JSON/JSONL/Excel).")
    parser.add_argument("--videos", type=int, default=10, help="Jumlah video per hashtag (default: 10).")
    parser.add_argument("--comments", type=int, default=30, help="Jumlah komentar per video (default: 30).")
    parser.add_argument(
        "--ms-token",
        type=str,
        default=None,
        help="Nilai cookie msToken dari TikTok.",
    )
    parser.add_argument("--output-dir", default="data/raw", help="Direktori output.")
    args = parser.parse_args()

    if args.export_file:
        print("\n" + "=" * 55)
        print("  Memuat Komentar TikTok dari File Ekspor...")
        print("=" * 55)
        df = load_from_export(args.export_file, raw_dir=args.output_dir)
        print("\n" + "=" * 55)
        print(f"  HASIL IMPORT TIKTOK: {len(df)} komentar berhasil disimpan.")
        print("=" * 55)

    elif args.collect:
        print("\n" + "=" * 55)
        print("  Mulai pengumpulan komentar TikTok Real (Scraper)...")
        print("=" * 55)
        df = asyncio.run(
            collect_tiktok_comments_async(
                n_videos_per_tag=args.videos,
                n_comments_per_video=args.comments,
                output_dir=args.output_dir,
                ms_token=args.ms_token,
            )
        )
        print("\n" + "=" * 55)
        print("  HASIL PENGUMPULAN DATA TIKTOK")
        print("=" * 55)
        print(f"  Total komentar terkumpul : {len(df)}")
        if not df.empty:
            print("\n  Contoh komentar:")
            for i, r in df.head(5).iterrows():
                safe = str(r["text"]).encode("ascii", errors="replace").decode("ascii")
                print(f"  [{i+1}] {safe[:70]}")
        else:
            print("\n  [TIPS] TikTok memblokir scraping headless.")
            print("         Gunakan: python src/collect/tiktok.py --export-file <path_file>")
        print("=" * 55)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
