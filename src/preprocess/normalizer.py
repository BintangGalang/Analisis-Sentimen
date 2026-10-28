"""
src/preprocess/normalizer.py
============================
Tahap normalisasi teks: emoji → teks, normalisasi slang,
dan deteksi code-mixing (Indonesia-Inggris).

Urutan dalam pipeline (sesuai project-rules):
    4. emoji → teks (menggunakan library `emoji`)
    5. normalisasi slang (kamus JSON)
    6. deteksi code-mixing (flag, bukan hapus)

Catatan:
    - Code-mixing (kata Inggris) TIDAK dihapus, hanya diberi flag.
      Ini penting karena kata Inggris bisa membawa sentimen.
    - Kata yang tidak dikenal (OOV) dicatat, bukan dihapus diam-diam.
    - Kamus slang dapat diperbarui tanpa mengubah kode.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Konstanta & Pola
# ─────────────────────────────────────────────────────────────

# Pola kasar untuk mendeteksi kata Inggris (heuristik):
# Kata yang HANYA terdiri dari huruf ASCII dan tidak ada di kamus
# Indonesia sederhana dianggap potensi code-mix.
_RE_LATIN_WORD = re.compile(r"\b[a-zA-Z]+\b")

# Kata Indonesia sangat umum (tidak dianggap code-mix)
_ID_COMMON_WORDS: set[str] = {
    "yang", "di", "dan", "ke", "dari", "ini", "itu", "dengan",
    "untuk", "dalam", "juga", "sudah", "tidak", "saya", "kamu",
    "dia", "kami", "kita", "mereka", "ada", "akan", "bisa",
    "pada", "atau", "karena", "jika", "saat", "lebih", "sangat",
    "saja", "tapi", "namun", "sama", "harus", "mau", "sini",
    "sana", "mana", "bagaimana", "mengapa", "kapan", "siapa",
    "apa", "bukan", "belum", "masih", "terus", "lagi", "sudah",
    "tentang", "terima", "kasih", "selamat", "pagi", "siang",
    "malam", "hari", "tahun", "bulan", "minggu", "waktu",
    "tempat", "kali", "orang", "teman", "keluarga", "kerja",
    "belajar", "sekolah", "kampus", "universitas", "mahasiswa",
    "dosen", "kuliah", "nilai", "ujian", "tugas", "semester",
    "undiksha", "singaraja", "denpasar", "bali", "ganesha",
    # Tambahan kata umum Indonesia
    "sangat", "sekali", "paling", "cukup", "agak", "sedikit",
    "menyenangkan", "menarik", "bagus", "baik", "buruk", "jelek",
    "besar", "kecil", "banyak", "sedikit", "lama", "baru", "lain",
    "semua", "setiap", "beberapa", "karena", "supaya", "agar",
    "memang", "tetap", "sudah", "pernah", "sedang", "tengah",
    "antara", "sampai", "sejak", "ketika", "sehingga", "meski",
    "walaupun", "meskipun", "bahwa", "agar", "supaya", "kalau",
    "seperti", "seakan", "seolah", "hampir", "hanya", "cuma",
    "perlu", "ingin", "mau", "dapat", "boleh", "wajib", "harus",
    "jadi", "menjadi", "adalah", "merupakan", "termasuk",
    "serta", "maupun", "ataupun", "bahkan", "selain", "kecuali",
    "selalu", "sering", "kadang", "jarang", "pernah", "tidak",
    "belum", "sudah", "lagi", "lalu", "kemudian", "akhirnya",
    "pertama", "kedua", "ketiga", "terakhir", "sebelum", "sesudah",
    "depan", "belakang", "atas", "bawah", "kiri", "kanan",
    "sini", "sana", "situ", "mana", "sebelah", "antara",
}


# ─────────────────────────────────────────────────────────────
# Loader Kamus Slang (cached)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_slang_dict(slang_path: str = "data/interim/slang_dict.json") -> dict[str, str]:
    """Memuat kamus slang dari file JSON (cached, hanya dibaca sekali).

    Args:
        slang_path: Path ke file kamus slang JSON.

    Returns:
        Dict mapping slang → kata baku.

    Raises:
        FileNotFoundError: Jika file kamus tidak ditemukan.
    """
    path = Path(slang_path)
    if not path.exists():
        logger.warning(
            "Kamus slang tidak ditemukan: %s. Normalisasi slang dilewati.", path
        )
        return {}

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    # Hapus kunci metadata
    slang_dict = {k: v for k, v in raw.items() if not k.startswith("_")}
    logger.info("Kamus slang dimuat: %d entri dari %s", len(slang_dict), path)
    return slang_dict


# ─────────────────────────────────────────────────────────────
# Emoji → Teks
# ─────────────────────────────────────────────────────────────

def emoji_to_text(text: str, language: str = "id") -> tuple[str, bool]:
    """Mengonversi emoji menjadi representasi teks.

    Menggunakan library `emoji` dengan deskripsi bahasa Inggris
    karena deskripsi bahasa Indonesia belum tersedia di library.
    Hasilnya tetap bermakna untuk model.

    Args:
        text    : Teks input (sudah lowercase).
        language: Kode bahasa untuk deskripsi emoji ('en' atau 'id').

    Returns:
        Tuple (teks_dengan_emoji_diganti, ada_emoji: bool).

    Contoh::

        >>> emoji_to_text("Undiksha keren 🔥")
        ('Undiksha keren  fire ', True)
    """
    try:
        import emoji as emoji_lib  # noqa: PLC0415

        has_emoji = emoji_lib.emoji_count(text) > 0
        # Demojize: :fire: dsb. → lalu bersihkan tanda titik dua
        demojized = emoji_lib.demojize(text, language="en")
        # Ubah :kata_kunci: → kata kunci (dengan spasi)
        cleaned = re.sub(r":([a-z_]+):", lambda m: f" {m.group(1).replace('_', ' ')} ", demojized)
        return cleaned, has_emoji

    except ImportError:
        logger.warning("Library 'emoji' tidak terinstall. Emoji tidak dikonversi.")
        return text, False


# ─────────────────────────────────────────────────────────────
# Normalisasi Slang
# ─────────────────────────────────────────────────────────────

def normalize_slang(
    text: str,
    slang_dict: dict[str, str] | None = None,
    slang_path: str = "data/interim/slang_dict.json",
    oov_collector: list[str] | None = None,
) -> str:
    """Mengganti kata-kata slang dengan bentuk baku sesuai kamus.

    Normalisasi dilakukan per-token (kata). Token yang tidak ada
    di kamus dibiarkan apa adanya; jika ``oov_collector`` diberikan,
    token tersebut dicatat untuk analisis lebih lanjut.

    Args:
        text         : Teks input (sudah lowercase dan dibersihkan).
        slang_dict   : Dict slang (jika None, dimuat dari slang_path).
        slang_path   : Path kamus slang (digunakan jika slang_dict=None).
        oov_collector: List untuk mengumpulkan kata OOV (out-of-vocabulary).

    Returns:
        Teks dengan slang sudah dinormalisasi.

    Contoh::

        >>> normalize_slang("gak bisa kuliah krn sakit")
        'tidak bisa kuliah karena sakit'
    """
    if slang_dict is None:
        slang_dict = load_slang_dict(slang_path)

    if not slang_dict:
        return text

    tokens = text.split()
    normalized: list[str] = []

    for token in tokens:
        if token in slang_dict:
            normalized.append(slang_dict[token])
        else:
            normalized.append(token)
            # Catat kata tidak dikenal (bukan angka, bukan emoji desc)
            if oov_collector is not None and re.match(r"^[a-z]+$", token):
                if token not in _ID_COMMON_WORDS and len(token) > 2:
                    oov_collector.append(token)

    return " ".join(normalized)


# ─────────────────────────────────────────────────────────────
# Deteksi Code-Mixing
# ─────────────────────────────────────────────────────────────

def detect_codemix(text: str) -> tuple[bool, list[str]]:
    """Mendeteksi keberadaan kata bahasa Inggris (code-mixing ID-EN).

    Heuristik: kata Latin yang bukan merupakan kata Indonesia umum
    dan tidak ada di kamus slang dianggap sebagai kata Inggris.

    PENTING: Kata Inggris TIDAK dihapus — hanya ditandai.
    Code-mixing adalah fenomena linguistik yang membawa sentimen.

    Args:
        text: Teks input (sudah dinormalisasi slang).

    Returns:
        Tuple (has_codemix: bool, english_words: list[str]).

    Contoh::

        >>> detect_codemix("kampusnya nice banget recommend deh")
        (True, ['nice', 'recommend'])
    """
    words = _RE_LATIN_WORD.findall(text.lower())
    english_candidates: list[str] = []

    for word in words:
        if (
            word not in _ID_COMMON_WORDS
            and len(word) > 2
            and not word.isdigit()
        ):
            english_candidates.append(word)

    return bool(english_candidates), english_candidates


# ─────────────────────────────────────────────────────────────
# Pipeline Normalisasi Lengkap
# ─────────────────────────────────────────────────────────────

def normalize_text(
    text: str,
    slang_path: str = "data/interim/slang_dict.json",
    oov_collector: list[str] | None = None,
) -> dict[str, Any]:
    """Menjalankan semua tahap normalisasi secara berurutan.

    Tahap:
        1. Emoji → teks
        2. Normalisasi slang
        3. Deteksi code-mixing

    Args:
        text         : Teks input (hasil dari cleaner.clean_text).
        slang_path   : Path kamus slang.
        oov_collector: List untuk kumpul kata OOV (opsional).

    Returns:
        Dict berisi:
            - ``text_norm``    : Teks yang sudah dinormalisasi.
            - ``emoji_replaced``: True jika ada emoji yang dikonversi.
            - ``has_codemix``  : True jika ada kata Inggris terdeteksi.
            - ``en_words``     : List kata Inggris yang ditemukan.
    """
    slang_dict = load_slang_dict(slang_path)

    # 1. Emoji → teks
    text_no_emoji, has_emoji = emoji_to_text(text)

    # 2. Normalisasi slang
    text_norm = normalize_slang(
        text_no_emoji,
        slang_dict=slang_dict,
        oov_collector=oov_collector,
    )

    # Bersihkan spasi setelah normalisasi
    text_norm = re.sub(r"\s+", " ", text_norm).strip()

    # 3. Deteksi code-mixing
    has_codemix, en_words = detect_codemix(text_norm)

    return {
        "text_norm":      text_norm,
        "emoji_replaced": has_emoji,
        "has_codemix":    has_codemix,
        "en_words":       en_words,
    }
