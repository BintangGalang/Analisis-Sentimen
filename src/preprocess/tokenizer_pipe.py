"""
src/preprocess/tokenizer_pipe.py
================================
Tahap tokenisasi, stopword removal, dan stemming Sastrawi.

Urutan dalam pipeline (sesuai project-rules):
    7. Hapus stopword (Sastrawi StopWordRemover)
    8. Stemming (Sastrawi Stemmer)

Catatan:
    - Tokenisasi menggunakan split sederhana setelah pembersihan.
    - Stopword removal menggunakan daftar Sastrawi + custom domain.
    - Stemming Sastrawi dipanggil sekali per token (bukan per teks)
      untuk efisiensi; hasil di-cache dengan lru_cache.
    - Teks stemmed dan teks bersih (sebelum stemming) keduanya disimpan.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Stopword Tambahan Domain Akademik / Sosmed
# ─────────────────────────────────────────────────────────────

CUSTOM_STOPWORDS: set[str] = {
    # Interjeksi sosmed yang tidak membawa makna
    "yuk", "yukk", "ya", "yaa", "deh", "dong", "lah", "kah",
    "tuh", "nih", "nah", "wah", "weh", "woi", "hei", "hey",
    "aduh", "duh", "ih", "ah", "oh", "eh",
    # Kata filler
    "gitu", "gini", "begitu", "begini", "kayak", "kaya",
    # Pengisi komentar tanpa sentimen
    "komentar", "video", "konten", "channel", "playlist",
    # Kata terlalu umum untuk domain Undiksha
    "universitas", "kampus",  # → terlalu umum, tapi bisa dipertahankan jika perlu
}


# ─────────────────────────────────────────────────────────────
# Inisialisasi Sastrawi (lazy, hanya jika digunakan)
# ─────────────────────────────────────────────────────────────

def _get_sastrawi_tools() -> tuple[Any, set[str]]:
    """Menginisialisasi Sastrawi Stemmer dan StopWordRemover.

    Returns:
        Tuple (stemmer, set_stopwords).

    Raises:
        ImportError: Jika PySastrawi tidak terinstall.
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory       # noqa: PLC0415
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import (     # noqa: PLC0415
            StopWordRemoverFactory,
        )

        stemmer = StemmerFactory().create_stemmer()
        stop_factory = StopWordRemoverFactory()
        stopwords: set[str] = set(stop_factory.get_stop_words())
        stopwords.update(CUSTOM_STOPWORDS)

        return stemmer, stopwords

    except ImportError as exc:
        raise ImportError(
            "PySastrawi tidak terinstall. Jalankan: pip install PySastrawi"
        ) from exc


# Cache agar Sastrawi tidak diinisialisasi berulang kali
_sastrawi_cache: tuple[Any, set[str]] | None = None


def get_sastrawi() -> tuple[Any, set[str]]:
    """Mengembalikan instance Sastrawi (singleton pattern).

    Returns:
        Tuple (stemmer, stopwords_set).
    """
    global _sastrawi_cache
    if _sastrawi_cache is None:
        _sastrawi_cache = _get_sastrawi_tools()
        logger.info(
            "Sastrawi diinisialisasi. Jumlah stopword: %d", len(_sastrawi_cache[1])
        )
    return _sastrawi_cache


# ─────────────────────────────────────────────────────────────
# Tokenisasi
# ─────────────────────────────────────────────────────────────

def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Memecah teks menjadi list token (kata).

    Menggunakan pemisahan berdasarkan spasi dan tanda baca,
    dengan filter panjang minimum.

    Args:
        text      : Teks input (sudah dinormalisasi).
        min_length: Panjang minimum token yang disimpan.

    Returns:
        List token.

    Contoh::

        >>> tokenize("kuliah di undiksha sangat menyenangkan")
        ['kuliah', 'di', 'undiksha', 'sangat', 'menyenangkan']
    """
    # Hapus tanda baca yang tersisa, kecuali tanda negatif/angka
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if len(t) >= min_length]


# ─────────────────────────────────────────────────────────────
# Stopword Removal
# ─────────────────────────────────────────────────────────────

def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Menghapus stopword dari list token.

    Args:
        tokens   : List token hasil tokenisasi.
        stopwords: Set kata-kata yang akan dihapus.

    Returns:
        List token tanpa stopword.

    Contoh::

        >>> remove_stopwords(['saya', 'kuliah', 'di', 'undiksha'], stopwords)
        ['kuliah', 'undiksha']
    """
    return [t for t in tokens if t not in stopwords]


# ─────────────────────────────────────────────────────────────
# Stemming
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=50_000)
def stem_word_cached(word: str, stemmer: Any) -> str:
    """Melakukan stemming satu kata (dengan cache untuk efisiensi).

    Sastrawi stemmer cukup lambat untuk volume besar;
    caching per-kata signifikan meningkatkan performa.

    Args:
        word   : Kata yang akan di-stem.
        stemmer: Instance Sastrawi stemmer.

    Returns:
        Kata hasil stemming.

    Contoh::

        >>> stem_word_cached("mahasiswanya", stemmer)
        'mahasiswa'
    """
    return stemmer.stem(word)


def stem_tokens(tokens: list[str], stemmer: Any) -> list[str]:
    """Melakukan stemming pada seluruh list token.

    Args:
        tokens : List token.
        stemmer: Instance Sastrawi stemmer.

    Returns:
        List token hasil stemming.
    """
    return [stem_word_cached(t, stemmer) for t in tokens]


# ─────────────────────────────────────────────────────────────
# Pipeline Tokenisasi Lengkap
# ─────────────────────────────────────────────────────────────

def tokenize_and_stem(
    text: str,
    apply_stopword: bool = True,
    apply_stemming: bool = True,
    min_token_length: int = 2,
) -> dict[str, Any]:
    """Menjalankan tokenisasi, stopword removal, dan stemming.

    Args:
        text             : Teks input (hasil normalize_text).
        apply_stopword   : Apakah stopword removal diterapkan.
        apply_stemming   : Apakah stemming Sastrawi diterapkan.
        min_token_length : Panjang minimum token.

    Returns:
        Dict berisi:
            - ``tokens_clean`` : Token setelah stopword removal (sebelum stem).
            - ``tokens_stemmed``: Token setelah stemming.
            - ``text_tokens``   : Teks dari tokens_clean (spasi-pisah).
            - ``text_stemmed``  : Teks dari tokens_stemmed (spasi-pisah).

    Contoh::

        >>> result = tokenize_and_stem("saya kuliah di universitas undiksha")
        >>> result['tokens_stemmed']
        ['kuliah', 'universitas', 'undiksha']
    """
    stemmer, stopwords = get_sastrawi()

    # 1. Tokenisasi
    tokens = tokenize(text, min_length=min_token_length)

    # 2. Stopword removal
    tokens_clean = remove_stopwords(tokens, stopwords) if apply_stopword else tokens

    # 3. Stemming
    tokens_stemmed = stem_tokens(tokens_clean, stemmer) if apply_stemming else tokens_clean

    return {
        "tokens_clean":  tokens_clean,
        "tokens_stemmed": tokens_stemmed,
        "text_tokens":   " ".join(tokens_clean),
        "text_stemmed":  " ".join(tokens_stemmed),
    }
