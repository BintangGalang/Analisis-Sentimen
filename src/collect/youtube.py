"""
src/collect/youtube.py
======================
Pengumpulan komentar YouTube terkait Undiksha menggunakan YouTube Data API v3.

Alur:
    1. Cari video dengan keyword Undiksha (search.list).
    2. Ambil komentar dari setiap video (commentThreads.list).
    3. Anonimkan ID pengguna, tandai akun resmi & spam.
    4. Simpan ke data/raw/youtube_raw_{tanggal}.csv.

Penggunaan:
    python src/collect/youtube.py
    python src/collect/youtube.py --max-videos 50 --max-comments 200

Catatan:
    - Memerlukan YOUTUBE_API_KEY di file .env
    - Kuota default API: 10.000 unit/hari. Satu pencarian = 100 unit.
      Satu permintaan commentThreads = 1 unit per halaman.
    - Komentar yang dinonaktifkan pemilik video akan dilewati otomatis.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

# Tambahkan root proyek ke sys.path agar src.* dapat diimpor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, set_seed, setup_logging
from src.collect.utils_collect import build_standard_row, save_raw, RateLimiter, STANDARD_COLUMNS

import pandas as pd

logger = logging.getLogger(__name__)

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────

PLATFORM = "youtube"
MAX_RESULTS_PER_PAGE = 50       # Batas API: maks 50 per halaman
DEFAULT_MAX_VIDEOS   = 30
DEFAULT_MAX_COMMENTS = 100      # Per video


# ─────────────────────────────────────────────────────────────
# Helper: Membangun klien YouTube
# ─────────────────────────────────────────────────────────────

def _build_youtube_client() -> Any:
    """Membuat klien YouTube Data API v3.

    Returns:
        Resource objek dari google-api-python-client.

    Raises:
        EnvironmentError: Jika YOUTUBE_API_KEY tidak ditemukan.
        ImportError: Jika google-api-python-client belum terinstall.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "YOUTUBE_API_KEY tidak ditemukan di environment. "
            "Salin .env.example ke .env dan isi nilainya."
        )

    try:
        from googleapiclient.discovery import build  # noqa: PLC0415
        return build("youtube", "v3", developerKey=api_key)
    except ImportError as exc:
        raise ImportError(
            "google-api-python-client belum terinstall. "
            "Jalankan: pip install google-api-python-client"
        ) from exc


# ─────────────────────────────────────────────────────────────
# Pencarian Video
# ─────────────────────────────────────────────────────────────

def search_videos(
    youtube: Any,
    query: str,
    max_results: int = DEFAULT_MAX_VIDEOS,
    language: str = "id",
) -> list[dict[str, Any]]:
    """Mencari video YouTube berdasarkan query.

    Args:
        youtube    : Klien YouTube API.
        query      : Kata kunci pencarian.
        max_results: Jumlah video maksimum yang diambil.
        language   : Kode bahasa untuk filter relevansi (default: 'id').

    Returns:
        List dict berisi videoId, title, channelTitle, publishedAt.
    """
    logger.info("Mencari video untuk query: '%s' (maks %d video).", query, max_results)
    videos: list[dict[str, Any]] = []
    next_page_token: str | None = None

    limiter = RateLimiter(min_interval=0.5)

    while len(videos) < max_results:
        batch = min(MAX_RESULTS_PER_PAGE, max_results - len(videos))
        params: dict[str, Any] = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": batch,
            "relevanceLanguage": language,
            "pageToken": next_page_token,
        }

        with limiter:
            response = youtube.search().list(**params).execute()

        for item in response.get("items", []):
            # Skip item non-video (channel/playlist) yang tidak punya videoId
            video_id_val = item.get("id", {}).get("videoId")
            if not video_id_val:
                continue
            snippet = item.get("snippet", {})
            videos.append({
                "video_id":      video_id_val,
                "title":         snippet.get("title", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at":  snippet.get("publishedAt"),
                "description":   snippet.get("description", ""),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    logger.info("Total video ditemukan: %d", len(videos))
    return videos


# ─────────────────────────────────────────────────────────────
# Pengambilan Komentar
# ─────────────────────────────────────────────────────────────

def fetch_comments(
    youtube: Any,
    video_id: str,
    video_url: str,
    channel_title: str,
    max_comments: int = DEFAULT_MAX_COMMENTS,
) -> Iterator[dict[str, Any]]:
    """Generator: mengambil komentar dari satu video YouTube.

    Menggunakan pagination otomatis. Berhenti saat komentar habis
    atau ``max_comments`` tercapai.

    Args:
        youtube       : Klien YouTube API.
        video_id      : ID video YouTube.
        video_url     : URL video (untuk kolom 'url').
        channel_title : Nama channel pemilik video.
        max_comments  : Batas komentar per video.

    Yields:
        Dict baris data dengan skema STANDARD_COLUMNS.
    """
    count = 0
    next_page_token: str | None = None
    limiter = RateLimiter(min_interval=0.3)

    while count < max_comments:
        batch = min(MAX_RESULTS_PER_PAGE, max_comments - count)
        params: dict[str, Any] = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": batch,
            "textFormat": "plainText",
            "pageToken": next_page_token,
        }

        try:
            with limiter:
                response = youtube.commentThreads().list(**params).execute()
        except Exception as exc:
            # Komentar dinonaktifkan atau video dihapus
            logger.warning("Tidak bisa ambil komentar video %s: %s", video_id, exc)
            return

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            raw_author_id = top.get("authorChannelId", {}).get("value", video_id + str(count))
            text = top.get("textDisplay", "")
            # Sanitasi: hapus karakter yang tidak bisa di-encode (fix Windows)
            text = text.encode("utf-8", errors="replace").decode("utf-8")
            if not text.strip():
                continue

            row = build_standard_row(
                raw_id=raw_author_id,
                platform=PLATFORM,
                text=text,
                created_at=top.get("publishedAt"),
                username=top.get("authorDisplayName", ""),
                url=video_url,
            )
            # Metadata tambahan YouTube (tidak ada di skema standar, disimpan terpisah)
            row["like_count"]    = top.get("likeCount", 0)
            row["video_id_anon"] = __import__("hashlib").sha256(video_id.encode()).hexdigest()[:16]

            yield row
            count += 1
            if count >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break


# ─────────────────────────────────────────────────────────────
# Pipeline Utama
# ─────────────────────────────────────────────────────────────

def collect_youtube(
    keywords: list[str],
    raw_dir: str | Path,
    max_videos: int = DEFAULT_MAX_VIDEOS,
    max_comments: int = DEFAULT_MAX_COMMENTS,
) -> pd.DataFrame:
    """Menjalankan pipeline pengumpulan data YouTube secara lengkap.

    Args:
        keywords    : Daftar kata kunci pencarian dari config.
        raw_dir     : Direktori output data mentah.
        max_videos  : Maks video per keyword.
        max_comments: Maks komentar per video.

    Returns:
        DataFrame dengan semua komentar yang dikumpulkan.
    """
    youtube = _build_youtube_client()
    all_rows: list[dict[str, Any]] = []

    for kw in keywords:
        videos = search_videos(youtube, kw, max_results=max_videos)

        for video in videos:
            video_url = f"https://www.youtube.com/watch?v={video['video_id']}"
            logger.info(
                "Mengambil komentar dari: %s (%s)", video["title"][:60], video["video_id"]
            )

            video_count = 0
            for row in fetch_comments(
                youtube,
                video_id=video["video_id"],
                video_url=video_url,
                channel_title=video["channel_title"],
                max_comments=max_comments,
            ):
                all_rows.append(row)
                video_count += 1
            logger.info(
                "  → %d komentar dikumpulkan dari video: %s",
                video_count, video["title"][:50],
            )

    if not all_rows:
        logger.warning("Tidak ada komentar yang terkumpul. Periksa keyword dan kuota API.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(all_rows)

    # Deduplikasi berdasarkan id_anon + text
    before = len(df)
    df = df.drop_duplicates(subset=["id_anon", "text"]).reset_index(drop=True)
    logger.info("Deduplikasi: %d → %d baris.", before, len(df))

    save_raw(df, PLATFORM, raw_dir)
    return df


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point CLI untuk pengumpulan data YouTube."""
    parser = argparse.ArgumentParser(description="Kumpulkan komentar YouTube tentang Undiksha.")
    parser.add_argument("--config",        default="configs/config.yaml", help="Path ke config.yaml")
    parser.add_argument("--max-videos",    type=int, default=None, help="Override maks video per keyword")
    parser.add_argument("--max-comments",  type=int, default=None, help="Override maks komentar per video")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    set_seed(config["seed"])

    max_videos   = args.max_videos   or DEFAULT_MAX_VIDEOS
    max_comments = args.max_comments or DEFAULT_MAX_COMMENTS

    df = collect_youtube(
        keywords=config["search_keywords"],
        raw_dir=config["paths"]["raw_dir"],
        max_videos=max_videos,
        max_comments=max_comments,
    )

    logger.info("Selesai. Total baris terkumpul: %d", len(df))
    print(df[["platform", "text", "is_official", "is_spam_flag"]].head(10).to_string())


if __name__ == "__main__":
    main()
