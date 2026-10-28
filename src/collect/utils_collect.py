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
    """Menyimpan DataFrame ke direktori data/raw/ dalam format CSV.

    Format nama file: ``{platform}_raw_{YYYYMMDD}.csv``

    Args:
        df       : DataFrame dengan skema STANDARD_COLUMNS.
        platform : Nama platform.
        raw_dir  : Path direktori output.

    Returns:
        Path file yang disimpan.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = raw_dir / f"{platform}_raw_{date_str}.csv"

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Data %s disimpan: %s (%d baris)", platform, out_path, len(df))
    return out_path


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
