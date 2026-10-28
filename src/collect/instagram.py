"""
src/collect/instagram.py
========================
Pengumpulan data Instagram/Threads terkait Undiksha menggunakan
Threads API (Meta Graph API v1.0).

Strategi pengumpulan data (berurutan):
    1. **Keyword Search** (butuh App Review Meta): Mencari postingan publik
       yang menyebut Undiksha via `threads/search`. Memerlukan izin
       `threads_keyword_search` yang harus disetujui Meta terlebih dahulu.
    2. **Own Account Posts**: Ambil postingan dari akun yang terhubung token.
       Berguna jika token milik akun resmi Undiksha.
    3. **Load from Export** (fallback praktis): Muat data dari file CSV/JSON
       hasil ekspor manual. Direkomendasikan jika App Review belum selesai.

⚠️  Status Token Saat Ini (token akun pribadi):
    - Keyword search: ❌ Memerlukan App Review Meta
    - Own posts: ❌ Akun pribadi tidak ada postingan terkait Undiksha
    - Manual export: ✅ Gunakan `--export-file path/ke/export.csv`

Cara mendapatkan data Threads Undiksha (pilih salah satu):
    A. Minta token dari akun resmi Undiksha (jika ada akses).
    B. Ekspor manual: buka Threads.net → search 'Undiksha' →
       salin postingan ke CSV dengan kolom: id, text, timestamp, username.
    C. Daftar App Review Meta untuk izin threads_keyword_search.
    D. Gunakan dataset Threads publik dari Kaggle/Zenodo.

Penggunaan:
    # Muat dari file ekspor manual:
    python src/collect/instagram.py --export-file data/raw/threads_manual.csv

    # Jalankan search (jika sudah punya izin App Review):
    python src/collect/instagram.py --keyword "Undiksha" --max-posts 100
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, set_seed, setup_logging
from src.collect.utils_collect import (
    build_standard_row,
    save_raw,
    RateLimiter,
    STANDARD_COLUMNS,
)

import pandas as pd

logger = logging.getLogger(__name__)
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────

PLATFORM             = "instagram"
THREADS_BASE         = "https://graph.threads.net/v1.0"
DEFAULT_MAX_POSTS    = 100
DEFAULT_MAX_REPLIES  = 50

# Field standar yang diminta dari API
POST_FIELDS   = "id,text,timestamp,permalink,username,media_type"
REPLY_FIELDS  = "id,text,timestamp,username"


# ─────────────────────────────────────────────────────────────
# Helper: Konfigurasi Token
# ─────────────────────────────────────────────────────────────

def _get_credentials() -> tuple[str, str]:
    """Mengambil Threads Access Token dan User ID dari environment.

    Returns:
        Tuple (access_token, user_id).

    Raises:
        EnvironmentError: Jika token/user_id tidak ditemukan.
    """
    token   = os.environ.get("THREADS_ACCESS_TOKEN", "").strip()
    user_id = os.environ.get("THREADS_USER_ID", "").strip()

    if not token or not user_id:
        raise EnvironmentError(
            "THREADS_ACCESS_TOKEN atau THREADS_USER_ID tidak ditemukan di .env."
        )
    return token, user_id


def _api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Wrapper GET request ke Threads API dengan error handling.

    Args:
        endpoint: Path API relatif (tanpa leading slash), atau URL penuh.
        params  : Parameter query string.

    Returns:
        Response JSON sebagai dict.

    Raises:
        requests.HTTPError: Jika respons bukan 2xx.
    """
    url = endpoint if endpoint.startswith("http") else f"{THREADS_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)

    if not resp.ok:
        logger.warning("API error %s: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    return resp.json()


# ─────────────────────────────────────────────────────────────
# 1. Keyword Search (konten publik menyebut Undiksha)
# ─────────────────────────────────────────────────────────────

def search_threads_keyword(
    access_token: str,
    query: str,
    max_posts: int = DEFAULT_MAX_POSTS,
) -> list[dict[str, Any]]:
    """Mencari postingan publik Threads yang cocok dengan kata kunci.

    Menggunakan endpoint: GET /threads/search?q={query}
    Tidak memerlukan token dari akun resmi — token pribadi sudah cukup.

    Args:
        access_token: Token akses Threads.
        query       : Kata kunci pencarian (mis. "Undiksha").
        max_posts   : Jumlah maksimum postingan yang dikembalikan.

    Returns:
        List dict postingan dengan field: id, text, timestamp, username, permalink.
    """
    logger.info("Mencari Threads publik dengan keyword: '%s' (maks %d).", query, max_posts)
    posts: list[dict[str, Any]] = []

    params: dict[str, Any] = {
        "q":            query,
        "fields":       POST_FIELDS,
        "access_token": access_token,
        "limit":        min(max_posts, 100),
    }

    limiter = RateLimiter(min_interval=0.5)

    try:
        with limiter:
            data = _api_get("threads/search", params)
    except requests.HTTPError as exc:
        logger.error(
            "Keyword search gagal untuk '%s': %s\n"
            "\n"
            "📌 Threads Keyword Search memerlukan App Review Meta:\n"
            "   • Izin yang diperlukan: threads_keyword_search\n"
            "   • Daftarkan di: https://developers.facebook.com/\n"
            "     → Your App → App Review → Permissions\n"
            "\n"
            "🔄 Alternatif sementara (pilih salah satu):\n"
            "   A. Ekspor manual: buka https://www.threads.net/search?q=Undiksha\n"
            "      → salin ke CSV dengan kolom: id,text,timestamp,username\n"
            "      → jalankan: python src/collect/instagram.py --export-file data/raw/threads_manual.csv\n"
            "   B. Gunakan token dari akun resmi Undiksha (jika ada akses)\n"
            "   C. Dataset publik Threads dari Kaggle/HuggingFace",
            query, exc,
        )
        return posts

    for item in data.get("data", []):
        posts.append(item)
        if len(posts) >= max_posts:
            break

    # Pagination jika ada
    next_url = data.get("paging", {}).get("next")
    while next_url and len(posts) < max_posts:
        try:
            with limiter:
                data = _api_get(next_url, {})
        except requests.HTTPError:
            break
        for item in data.get("data", []):
            posts.append(item)
            if len(posts) >= max_posts:
                break
        next_url = data.get("paging", {}).get("next")

    logger.info("Postingan ditemukan untuk '%s': %d", query, len(posts))
    return posts


# ─────────────────────────────────────────────────────────────
# 2. Postingan Akun Sendiri (Own Posts)
# ─────────────────────────────────────────────────────────────

def fetch_own_posts(
    access_token: str,
    user_id: str,
    max_posts: int = 50,
) -> list[dict[str, Any]]:
    """Mengambil postingan dari akun yang terhubung ke token.

    Berguna jika token milik akun resmi Undiksha.
    Jika akun tidak punya postingan, mengembalikan list kosong (bukan error).

    Args:
        access_token: Token akses.
        user_id     : Threads User ID.
        max_posts   : Jumlah maksimum postingan.

    Returns:
        List dict postingan.
    """
    logger.info("Mengambil postingan own account (user_id: %s).", user_id)
    params: dict[str, Any] = {
        "fields":       POST_FIELDS,
        "access_token": access_token,
        "limit":        min(max_posts, 100),
    }
    limiter = RateLimiter(min_interval=0.5)

    try:
        with limiter:
            data = _api_get(f"{user_id}/threads", params)
    except requests.HTTPError as exc:
        logger.warning("Tidak bisa ambil own posts: %s", exc)
        return []

    posts = data.get("data", [])
    logger.info("Own posts ditemukan: %d", len(posts))
    return posts


# ─────────────────────────────────────────────────────────────
# 3. Replies dari Satu Postingan
# ─────────────────────────────────────────────────────────────

def fetch_replies(
    access_token: str,
    post_id: str,
    post_url: str,
    max_replies: int = DEFAULT_MAX_REPLIES,
) -> list[dict[str, Any]]:
    """Mengambil balasan dari satu postingan Threads.

    Args:
        access_token: Token akses.
        post_id     : ID postingan.
        post_url    : URL postingan (untuk kolom 'url').
        max_replies : Jumlah maksimum balasan.

    Returns:
        List dict baris data dengan skema STANDARD_COLUMNS.
    """
    rows: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "fields":       REPLY_FIELDS,
        "access_token": access_token,
        "limit":        min(max_replies, 100),
    }
    limiter = RateLimiter(min_interval=0.3)

    try:
        with limiter:
            data = _api_get(f"{post_id}/replies", params)
    except requests.HTTPError as exc:
        logger.debug("Tidak bisa ambil replies post %s: %s", post_id, exc)
        return rows

    for item in data.get("data", []):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        row = build_standard_row(
            raw_id=item.get("id", post_id),
            platform=PLATFORM,
            text=text,
            created_at=item.get("timestamp"),
            username=item.get("username", ""),
            url=post_url,
        )
        rows.append(row)
        if len(rows) >= max_replies:
            break

    return rows


# ─────────────────────────────────────────────────────────────
# 4. Konversi Post → Standard Row
# ─────────────────────────────────────────────────────────────

def _post_to_row(post: dict[str, Any]) -> dict[str, Any] | None:
    """Mengonversi satu postingan API ke format STANDARD_COLUMNS.

    Args:
        post: Dict postingan dari Threads API.

    Returns:
        Dict baris standar, atau None jika teks kosong.
    """
    text = (post.get("text") or "").strip()
    # Sanitasi encoding
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    if not text:
        return None

    return build_standard_row(
        raw_id=post.get("id", ""),
        platform=PLATFORM,
        text=text,
        created_at=post.get("timestamp"),
        username=post.get("username", ""),
        url=post.get("permalink", ""),
    )


# ─────────────────────────────────────────────────────────────
# 5. Pipeline Utama
# ─────────────────────────────────────────────────────────────

def collect_instagram(
    keywords: list[str],
    raw_dir: str | Path,
    max_posts: int = DEFAULT_MAX_POSTS,
    max_replies: int = DEFAULT_MAX_REPLIES,
) -> pd.DataFrame:
    """Menjalankan pipeline pengumpulan data Instagram/Threads secara lengkap.

    Urutan strategi:
        1. Keyword search untuk setiap kata kunci Undiksha → opini publik.
        2. Own account posts (jika ada) → postingan resmi.
        3. Replies dari setiap postingan yang ditemukan.

    Args:
        keywords    : Daftar kata kunci dari configs/config.yaml.
        raw_dir     : Direktori output data mentah.
        max_posts   : Maks postingan per keyword.
        max_replies : Maks replies per postingan.

    Returns:
        DataFrame hasil pengumpulan.
    """
    access_token, user_id = _get_credentials()
    all_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()  # Deduplikasi real-time

    # ── Strategi 1: Keyword Search ────────────────────────────
    # Gunakan keyword utama agar tidak terlalu banyak memanggil API
    primary_keywords = [kw for kw in keywords if "undiksha" in kw.lower()][:5]

    for kw in primary_keywords:
        posts = search_threads_keyword(access_token, kw, max_posts=max_posts)

        for post in posts:
            post_id  = post.get("id", "")
            post_url = post.get("permalink", "")

            # Tambah postingan itu sendiri
            row = _post_to_row(post)
            if row and post_id not in seen_ids:
                all_rows.append(row)
                seen_ids.add(post_id)

            # Ambil replies (opini orang lain terhadap postingan)
            if post_id:
                replies = fetch_replies(access_token, post_id, post_url, max_replies)
                for r in replies:
                    rid = r["id_anon"]
                    if rid not in seen_ids:
                        all_rows.append(r)
                        seen_ids.add(rid)

        logger.info("Keyword '%s': total akumulasi %d baris.", kw, len(all_rows))

    # ── Strategi 2: Own Account Posts ────────────────────────
    own_posts = fetch_own_posts(access_token, user_id, max_posts=50)
    for post in own_posts:
        post_id = post.get("id", "")
        row = _post_to_row(post)
        if row and post_id not in seen_ids:
            all_rows.append(row)
            seen_ids.add(post_id)

    # ── Hasil ────────────────────────────────────────────────
    if not all_rows:
        logger.warning(
            "Tidak ada data dari Instagram/Threads.\n"
            "Kemungkinan penyebab:\n"
            "  1. Keyword search belum didukung oleh versi token Anda\n"
            "     (perlu izin 'threads_basic' + 'threads_content_publish').\n"
            "  2. Tidak ada postingan publik yang cocok keyword.\n"
            "  3. Token sudah expired — generate ulang di developers.facebook.com."
        )
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(all_rows)
    logger.info("Total baris terkumpul dari Threads: %d", len(df))
    save_raw(df, PLATFORM, raw_dir)
    return df


# ─────────────────────────────────────────────────────────────
# 6. Load dari Ekspor Manual (Fallback)
# ─────────────────────────────────────────────────────────────

def load_from_export(
    export_path: str | Path,
    raw_dir: str | Path,
) -> pd.DataFrame:
    """Memuat data dari file CSV/JSON hasil ekspor manual Threads.

    Kolom minimum yang dibutuhkan: ``text``.
    Kolom opsional yang akan digunakan jika ada:
    ``id``, ``timestamp``, ``username``, ``url``.

    Ini adalah fallback praktis saat Threads Search API belum tersedia.

    Args:
        export_path: Path ke file CSV atau JSON ekspor manual.
        raw_dir    : Direktori untuk menyimpan hasil yang sudah dinormalisasi.

    Returns:
        DataFrame dengan skema STANDARD_COLUMNS.

    Contoh format CSV minimal::

        id,text,timestamp,username
        1,"Undiksha itu bagus sekali","2024-01-15T10:00:00Z","mahasiswa1"
        2,"UKT mahal banget","2024-01-16T09:00:00Z","user_xyz"
    """
    export_path = Path(export_path)
    logger.info("Memuat data dari ekspor manual: %s", export_path)

    if not export_path.exists():
        raise FileNotFoundError(f"File ekspor tidak ditemukan: {export_path}")

    # Baca CSV, JSON, atau JSONL
    if export_path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if export_path.suffix.lower() == ".tsv" else ","
        raw_df = pd.read_csv(export_path, sep=sep, dtype=str, encoding="utf-8-sig")
    elif export_path.suffix.lower() == ".json":
        raw_df = pd.read_json(export_path, dtype=str)
    elif export_path.suffix.lower() == ".jsonl":
        raw_df = pd.read_json(export_path, lines=True, dtype=str)
    else:
        raise ValueError(f"Format tidak didukung: {export_path.suffix}. Gunakan .csv, .json, atau .jsonl")

    logger.info("Baris mentah dibaca: %d", len(raw_df))

    rows: list[dict[str, Any]] = []
    for i, rec in raw_df.iterrows():
        # Fleksibel mengecek field teks: 'text' atau 'content'
        raw_text = rec.get("text") or rec.get("content") or ""
        text = str(raw_text).strip()
        if not text or text == "nan":
            continue
        text = text.encode("utf-8", errors="replace").decode("utf-8")

        username = str(rec.get("username") or rec.get("author") or "")
        created_at = rec.get("timestamp") or rec.get("created_at") or rec.get("scraped_at")
        raw_id = str(rec.get("id") or f"ig_{i}")

        row = build_standard_row(
            raw_id=raw_id,
            platform=PLATFORM,
            text=text,
            created_at=created_at,
            username=username,
            url=str(rec.get("url", rec.get("permalink", ""))),
        )
        rows.append(row)

    if not rows:
        logger.warning("Tidak ada baris valid dari file ekspor.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id_anon", "text"]).reset_index(drop=True)
    logger.info("Baris valid setelah normalisasi: %d", len(df))

    save_raw(df, PLATFORM, raw_dir)
    return df


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point CLI untuk pengumpulan data Instagram/Threads."""
    parser = argparse.ArgumentParser(
        description="Kumpulkan data Instagram/Threads tentang Undiksha.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh penggunaan:\n"
            "  # Load dari ekspor manual (DIREKOMENDASIKAN saat ini):\n"
            "  python src/collect/instagram.py --export-file data/raw/threads_manual.csv\n\n"
            "  # Search via API (butuh App Review Meta):\n"
            "  python src/collect/instagram.py --keyword Undiksha --max-posts 100\n"
        ),
    )
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--max-posts",   type=int, default=None)
    parser.add_argument("--max-replies", type=int, default=None)
    parser.add_argument(
        "--keyword", type=str, default=None,
        help="Override satu keyword (default: semua dari config)."
    )
    parser.add_argument(
        "--export-file", type=str, default=None,
        help="Path ke file CSV/JSON ekspor manual Threads. "
             "Gunakan ini jika Threads Search API belum tersedia.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    set_seed(config["seed"])

    # Mode: load dari ekspor manual
    if args.export_file:
        df = load_from_export(
            export_path=args.export_file,
            raw_dir=config["paths"]["raw_dir"],
        )
    else:
        # Mode: API search
        keywords = [args.keyword] if args.keyword else config["search_keywords"]
        df = collect_instagram(
            keywords=keywords,
            raw_dir=config["paths"]["raw_dir"],
            max_posts=args.max_posts   or DEFAULT_MAX_POSTS,
            max_replies=args.max_replies or DEFAULT_MAX_REPLIES,
        )

    logger.info("Selesai. Total baris: %d", len(df))
    if not df.empty:
        summary_str = df[["platform", "is_official", "is_spam_flag"]].head(10).to_string()
        print(summary_str)


if __name__ == "__main__":
    main()

