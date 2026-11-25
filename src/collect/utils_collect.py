"""
src/collect/utils_collect.py
============================
Utilitas bersama untuk semua modul pengumpulan data.

Berisi:
    - anonymize_id      : Hash SHA-256 username/user-ID agar tidak bisa dilacak.
    - is_official_account: Deteksi akun resmi Undiksha dari daftar putih.
    - flag_spam_promo   : Heuristik sederhana untuk postingan promosi/pengumuman PMB.
    - RateLimiter       : Context-manager untuk menjaga jarak antar request API.
    - save_raw          : Simpan DataFrame ke data/raw/ dengan nama standar.
    - STANDARD_COLUMNS  : Skema kolom seragam antar platform.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Skema Kolom Standar
# ─────────────────────────────────────────────────────────────

STANDARD_COLUMNS: list[str] = [
    "id_anon",        # SHA-256 dari ID asli
    "platform",       # twitter | youtube | instagram | tiktok
    "text",           # Teks konten (caption/komentar/tweet)
    "created_at",     # Datetime posting (UTC)
    "is_official",    # True jika akun resmi Undiksha
    "is_spam_flag",   # True jika terdeteksi promosi/spam
    "url",            # URL postingan (opsional)
    "collected_at",   # Waktu pengambilan data (UTC)
]

# ─────────────────────────────────────────────────────────────
# Daftar Akun Resmi Undiksha (perbarui sesuai temuan)
# ─────────────────────────────────────────────────────────────

OFFICIAL_ACCOUNTS: set[str] = {
    # YouTube
    "undiksha", "undiksha official", "universitas pendidikan ganesha",
    # Instagram / Threads
    "undiksha.ac.id", "undiksha_official", "humas.undiksha",
    # Twitter/X
    "@undiksha", "undiksha",
    # TikTok
    "undiksha.official",
}

# Kata kunci yang mengindikasikan postingan bukan opini (promosi, pengumuman)
_SPAM_KEYWORDS: list[str] = [
    "penerimaan mahasiswa baru", "pmb", "pendaftaran dibuka",
    "daftar sekarang", "klik link", "promo", "diskon",
    "follow", "subscribe", "like dan subscribe",
]


# ─────────────────────────────────────────────────────────────
# Fungsi Utilitas
# ─────────────────────────────────────────────────────────────

def anonymize_id(raw_id: str | int) -> str:
    """Menghasilkan hash SHA-256 dari ID/username asli.

    Proses ini satu arah — ID asli tidak bisa dipulihkan dari hash.
    Simpan mapping raw_id → id_anon di tempat aman terpisah jika perlu.

    Args:
        raw_id: ID atau username asli dari platform.

    Returns:
        String hex SHA-256 (64 karakter).
    """
    return hashlib.sha256(str(raw_id).encode("utf-8")).hexdigest()


def is_official_account(username: str) -> bool:
    """Mengembalikan True jika username termasuk akun resmi Undiksha.

    Args:
        username: Username atau nama channel dari platform.

    Returns:
        True jika akun resmi, False jika akun publik.
    """
    return username.strip().lower() in OFFICIAL_ACCOUNTS


def flag_spam_promo(text: str) -> bool:
    """Heuristik sederhana untuk menandai postingan promosi/pengumuman.

    Postingan yang terdeteksi spam/promo diberi flag True dan tidak
    dihitung sebagai opini publik dalam analisis sentimen.

    Args:
        text: Teks postingan yang sudah dilowercased.

    Returns:
        True jika terindikasi spam/promosi.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in _SPAM_KEYWORDS)


def build_standard_row(
    raw_id: str | int,
    platform: str,
    text: str,
    created_at: datetime | str | None,
    username: str = "",
    url: str = "",
) -> dict[str, Any]:
    """Membangun satu baris data dengan skema standar.

    Args:
        raw_id     : ID asli postingan atau komentar.
        platform   : Nama platform (youtube, instagram, dll).
        text       : Teks konten.
        created_at : Waktu posting.
        username   : Username/channel pemilik konten.
        url        : URL postingan (opsional).

    Returns:
        Dict dengan key sesuai STANDARD_COLUMNS.
    """
    return {
        "id_anon":      anonymize_id(raw_id),
        "platform":     platform,
        "text":         text.strip(),
        "created_at":   created_at,
        "is_official":  is_official_account(username),
        "is_spam_flag": flag_spam_promo(text),
        "url":          url,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def save_raw(df: pd.DataFrame, platform: str, raw_dir: str | Path) -> Path:
    """Menyimpan DataFrame ke direktori data/raw/ dalam format JSONL standar (seragam dengan seluruh platform).

    Format nama file: ``{platform}_raw.jsonl`` (append atau tulis).

    Args:
        df       : DataFrame dengan skema STANDARD_COLUMNS.
        platform : Nama platform (youtube, twitter, dll).
        raw_dir  : Path direktori output data/raw.

    Returns:
        Path file JSONL yang disimpan.
    """
    import json
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = raw_dir / f"{platform}_raw.jsonl"
    existing_ids = set()
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if "id" in obj:
                            existing_ids.add(str(obj["id"]))
        except Exception:
            pass

    new_count = 0
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for _, r in df.iterrows():
            content = str(r.get("text", r.get("content", ""))).strip()
            if not content or content == "nan":
                continue
            row_id = str(r.get("id_anon", r.get("id", ""))).strip()
            if not row_id or row_id == "nan":
                row_id = hashlib.sha256(content.encode()).hexdigest()[:16]

            if row_id in existing_ids:
                continue

            existing_ids.add(row_id)
            new_count += 1
            data_type = "comment" if ("video_id_anon" in r or "parent_id" in r or platform == "youtube") else "post"
            rec = {
                "id": row_id,
                "platform": platform,
                "data_type": data_type,
                "parent_id": str(r.get("video_id_anon", r.get("parent_id", ""))) if pd.notna(r.get("video_id_anon", r.get("parent_id"))) else None,
                "author": str(r.get("author", row_id[:16])),
                "content": content,
                "clean_content": None,
                "created_at": str(r.get("created_at", "")) if pd.notna(r.get("created_at")) else None,
                "url": str(r.get("url", "")) if pd.notna(r.get("url")) else None,
                "engagement": {
                    "likes": int(float(r.get("like_count", r.get("likes", 0)) or 0)),
                    "comments_count": 0,
                    "shares": 0,
                    "views": 0,
                },
                "hashtag_query": "#undiksha",
                "scraped_at": str(r.get("collected_at", r.get("scraped_at", ""))) if pd.notna(r.get("collected_at", r.get("scraped_at"))) else None,
                "metadata": {
                    "is_official": bool(r.get("is_official", False)),
                    "is_spam_flag": bool(r.get("is_spam_flag", False)),
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Data %s disimpan dalam format JSONL: %s (+%d baris baru)", platform, jsonl_path, new_count)
    return jsonl_path


# ─────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Context-manager sederhana untuk menjaga jarak antar request API.

    Contoh::

        limiter = RateLimiter(min_interval=1.0)
        for item in items:
            with limiter:
                response = api.call(item)
    """

    def __init__(self, min_interval: float = 1.0) -> None:
        """
        Args:
            min_interval: Waktu minimum (detik) antara dua request.
        """
        self.min_interval = min_interval
        self._last_call: float = 0.0

    def __enter__(self) -> "RateLimiter":
        elapsed = time.monotonic() - self._last_call
        wait = self.min_interval - elapsed
        if wait > 0:
            logger.debug("Rate limit: menunggu %.2f detik.", wait)
            time.sleep(wait)
        return self

    def __exit__(self, *args: Any) -> None:
        self._last_call = time.monotonic()
