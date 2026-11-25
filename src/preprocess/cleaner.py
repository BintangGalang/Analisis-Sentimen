"""
src/preprocess/cleaner.py
=========================
Tahap pembersihan teks awal: lowercase, hapus URL/mention,
normalisasi hashtag, dan hapus noise karakter.

Urutan dalam pipeline (sesuai project-rules):
    1. lowercase
    2. hapus URL
    3. hapus mention (@username)
    4. normalisasi hashtag (#topik → topik)
    5. hapus karakter noise (angka opsional, karakter berulang)

Catatan:
    - Setiap fungsi adalah pure function (tidak ada side-effect).
    - Teks asli (text_raw) SELALU disimpan berdampingan dengan text_clean.
    - Karakter Unicode (emoji, dll.) TIDAK dihapus di tahap ini;
      ditangani di normalizer.py.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Pola Regex (dikompilasi sekali untuk efisiensi)
# ─────────────────────────────────────────────────────────────

_RE_URL      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION  = re.compile(r"@\w+")
_RE_HASHTAG  = re.compile(r"#(\w+)")
_RE_REPEATED = re.compile(r"(.)\1{2,}")      # karakter berulang >2x → 2x
_RE_MULTI_WS = re.compile(r"\s+")            # spasi berulang → satu spasi
_RE_NON_ALPHA_EDGE = re.compile(r"^[^\w]+|[^\w]+$")  # noise di awal/akhir token

# Pola regex deteksi iklan komersial non-opini
COMMERCIAL_SPAM_PATTERNS = [
    r'\bdi\s*jual\s+(tanah|rumah|ruko|kavling|bangunan|villa|kos|kost)\b',
    r'\btanah\s+(kavling|premium|murah|dijual|siap\s+bangun)\b',
    r'\b(shm|imb)\b',
    r'\bsertifikat\s+hak\s+milik\b',
    r'\bluas\s*:\s*\d+.*are\b',
    r'\bharga\s*:\s*.*per\s*are\b',
    r'\brumah\s+\d+\s+kamar\s+siap\s+huni\b',
    r'\bterima\s+(kost|kos|kontrakan|sewa\s+kamar)\b',
    r'\b(kost|kos)\s+exclusive\b',
    r'\[\s*kost\s+dekat\s+kampus',
    r'\bready\s+(\d+\s+)?(kamar|kos|kost)\b',
    r'\b\d+k\s*/\s*(bln|bulan)\b',
    r'\bkamar\s+mandi\s+(luar|dalam)\b',
    r'\bdi\s*sewakan\s+(kamar|kos|kost|rumah|kontrakan)\b',
    r'\bdikontrakan\b',
    r'\bover\s+sewa\b',
    r'\bbagi\s+yang\s+berminat\s+bisa\s+langsung\s+dm\b',
    r'\bminat\s+dm\s+ya\b',
    r'\bminat\s*\?\?\s*(dm|wa)\b',
    r'\bjasa\s+joki\b',
    r'\bjoki\s+(tugas|skripsi|makalah|krs|laporan|ujian|ospek)\b',
    r'\bopen\s+joki\b',
    r'\breadystock\b',
    r'\bhubungi\s+wa\s*:?\s*\d+',
    r'\bwa\s*:\s*08\d+',
    r'\bjasa\s+profesional\s+photo(box|booth)\b',
]
_RE_COMMERCIAL_SPAM = re.compile('|'.join(COMMERCIAL_SPAM_PATTERNS), re.IGNORECASE)


# ─────────────────────────────────────────────────────────────
# Fungsi Pembersihan Individual
# ─────────────────────────────────────────────────────────────

def remove_boilerplate(text: str) -> str:
    """Menghapus redaksi boilerplate UI media sosial (badge penulis, tombol lihat selengkapnya, tag arsip, timestamp edit)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = text
    cleaned = re.sub(r'\[\s*ARSIP[^\]]*\]', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(\s*ARSIP[^\)]*\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*(Pembuat|Penulis|Penggemar\s+Berat|Top\s+Fan|Kontributor\s+Utama)\s*[\r\n]+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*(Pembuat|Penulis|Penggemar\s+Berat|Top\s+Fan|Kontributor\s+Utama)\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*(Pembuat|Penulis)\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Telah\s+disunting(\s*•\s*\d+\s*(ming|hr|bln|thn|hari|minggu|bulan|tahun|m|w|d|h|s))?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*\d+\s*(ming|hr|bln|thn|hari|minggu|bulan|tahun)\s*[\r\n]+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'•\s*\d+\s*(ming|hr|bln|thn|hari|minggu|bulan|tahun|m|w|d|h|s)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(\.{2,}|…)?\s*(Lihat|Baca)\s+(selengkapnya|lebih\s+banyak)\b.*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(\.{2,}|…)?\s*See\s+more\b.*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(\.{2,}|…)?\s*Lihat\s+terjemahan\b.*', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip(' .\t\r\n-–—…')


def is_commercial_spam(text: str) -> bool:
    """Mendeteksi apakah teks merupakan iklan komersial non-opini kampus."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_RE_COMMERCIAL_SPAM.search(text))

def to_lowercase(text: str) -> str:
    """Mengubah seluruh teks menjadi huruf kecil.

    Args:
        text: Teks input.

    Returns:
        Teks lowercase.

    Contoh::

        >>> to_lowercase("Undiksha BAGUS")
        'undiksha bagus'
    """
    return text.lower()


def remove_urls(text: str) -> str:
    """Menghapus URL (http/https/www) dari teks.

    Args:
        text: Teks input.

    Returns:
        Teks tanpa URL, diganti spasi.

    Contoh::

        >>> remove_urls("kunjungi https://undiksha.ac.id untuk info")
        'kunjungi  untuk info'
    """
    return _RE_URL.sub(" ", text)


def remove_mentions(text: str) -> str:
    """Menghapus mention (@username) dari teks.

    Username asli sudah dianonimkan sebelumnya di tahap collect;
    mention di teks juga dihapus agar tidak membiaskan analisis.

    Args:
        text: Teks input.

    Returns:
        Teks tanpa mention.

    Contoh::

        >>> remove_mentions("@undiksha keren banget!")
        ' keren banget!'
    """
    return _RE_MENTION.sub(" ", text)


def normalize_hashtags(text: str) -> str:
    """Mengubah hashtag (#topik) menjadi kata biasa (topik).

    Tanda # dihapus agar kata kunci dari hashtag tetap dianalisis
    sebagai konten, bukan metadata.

    Args:
        text: Teks input.

    Returns:
        Teks dengan hashtag diubah jadi kata.

    Contoh::

        >>> normalize_hashtags("#undiksha #singaraja keren")
        'undiksha singaraja keren'
    """
    return _RE_HASHTAG.sub(r"\1", text)


def reduce_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """Mereduksi karakter yang berulang lebih dari ``max_repeat`` kali.

    Berguna untuk kasus seperti "kereeeeen" → "kereen" atau
    "wkwkwkwk" tetap utuh karena 'w' dan 'k' bergantian.

    Args:
        text      : Teks input.
        max_repeat: Jumlah maksimal pengulangan yang diizinkan.

    Returns:
        Teks dengan karakter berulang dikurangi.

    Contoh::

        >>> reduce_repeated_chars("bagussss")
        'baguss'
    """
    pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
    return pattern.sub(r"\1" * max_repeat, text)


def normalize_whitespace(text: str) -> str:
    """Menghapus spasi berlebih dan strip whitespace di tepi.

    Args:
        text: Teks input.

    Returns:
        Teks dengan spasi normal.
    """
    return _RE_MULTI_WS.sub(" ", text).strip()


# ─────────────────────────────────────────────────────────────
# Pipeline Pembersihan Lengkap
# ─────────────────────────────────────────────────────────────

def clean_text(
    text: str,
    lowercase:          bool = True,
    remove_url:         bool = True,
    remove_mention:     bool = True,
    norm_hashtag:       bool = True,
    reduce_repeated:    bool = True,
    remove_bp:          bool = True,
) -> str:
    """Menjalankan semua tahap pembersihan dasar secara berurutan.

    Ini adalah fungsi entry point untuk tahap 1 pipeline preprocessing.
    Teks asli TIDAK dimodifikasi — fungsi ini mengembalikan salinan bersih.

    Args:
        text            : Teks input mentah.
        lowercase       : Apakah lowercase diterapkan.
        remove_url      : Apakah URL dihapus.
        remove_mention  : Apakah mention dihapus.
        norm_hashtag    : Apakah hashtag dinormalisasi.
        reduce_repeated : Apakah karakter berulang dikurangi.
        remove_bp       : Apakah boilerplate medsos dihapus.

    Returns:
        Teks yang sudah dibersihkan.

    Contoh::

        >>> clean_text("@user Undiksha KEREEEEEN!! #undiksha https://t.co/xxx")
        'undiksha kereeen!! undiksha'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    result = text

    if remove_bp:
        result = remove_boilerplate(result)
    if lowercase:
        result = to_lowercase(result)
    if remove_url:
        result = remove_urls(result)
    if remove_mention:
        result = remove_mentions(result)
    if norm_hashtag:
        result = normalize_hashtags(result)
    if reduce_repeated:
        result = reduce_repeated_chars(result)

    result = normalize_whitespace(result)
    return result
