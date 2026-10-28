"""
src/evaluate/aspect.py
======================
Klasifikasi aspek reputasi kampus Undiksha (PRD §3a, PRD §6 F13).

Kategori aspek (sesuai PRD §3a & configs/config.yaml):
1. akademik              : Dosen, kurikulum, akreditasi, ranking, matkul, skripsi
2. penerimaan_biaya      : SNBP, SNBT, mandiri, UKT, beasiswa, KIP-K, pendaftaran
3. fasilitas             : Kampus, gedung, lab, wifi, ruang kelas, toilet, perpus
4. layanan_administrasi  : SIAKAD, birokrasi, admin, pelayanan, surat, KRS
5. kemahasiswaan         : PKKMB, BEM, UKM, organisasi, lomba, prestasi, KKN
6. lulusan_karier        : Wisuda, alumni, lulusan, dunia kerja, tracer study
7. isu_citra             : Berita, viral, reputasi, kebijakan, almamater
8. lainnya               : Komentar umum tanpa aspek spesifik
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config

logger = logging.getLogger(__name__)

# Definisi kata kunci aspek komprehensif
DEFAULT_ASPECT_KEYWORDS: dict[str, list[str]] = {
    "akademik": [
        "dosen", "kuliah", "matkul", "mata kuliah", "kurikulum", "akreditasi",
        "ranking", "skripsi", "tugas", "ujian", "uas", "uts", "nilai", "ipk",
        "prodi", "fakultas", "ftk", "fbs", "fmipa", "fok", "fip", "fh", "fe",
        "kedokteran", "belajar", "praktikum", "bimbingan", "profesor",
    ],
    "penerimaan_biaya": [
        "ukt", "biaya", "bayar", "snbp", "snbt", "mandiri", "beasiswa", "kip",
        "kip-k", "cicilan", "uang", "mahal", "gratis", "pendaftaran", "daftar",
        "gelombang", "lolos", "diterima", "seleksi", "pmb", "jalur",
    ],
    "fasilitas": [
        "kampus", "gedung", "lab", "laboratorium", "ruang", "kelas", "ac",
        "toilet", "wc", "parkir", "wifi", "internet", "jaringan", "asrama",
        "kos", "kost", "perpustakaan", "perpus", "taman", "lapangan",
    ],
    "layanan_administrasi": [
        "siakad", "web", "website", "server", "admin", "birokrasi", "surat",
        "ijazah", "legalisir", "pelayanan", "staf", "pegawai", "sistem",
        "krs", "war krs", "registrasi", "layanan",
    ],
    "kemahasiswaan": [
        "pkkmb", "bem", "ormawa", "ukm", "lomba", "prestasi", "juara", "kkn",
        "mbkm", "magang", "organisasi", "kegiatan", "acara", "dies natalis",
        "seni", "tari", "musik", "lagu", "padus",
    ],
    "lulusan_karier": [
        "wisuda", "wisudawan", "lulus", "lulusan", "alumni", "sarjana", "kerja",
        "karier", "karir", "tracer", "ijazah", "toga", "magang kerja",
    ],
    "isu_citra": [
        "singaraja", "bali", "undiksha", "kampusku", "almamater", "berita",
        "viral", "citra", "reputasi", "lingkungan", "masyarakat", "kebijakan",
    ],
}


def classify_aspect(
    text: str,
    keywords_dict: dict[str, list[str]] | None = None,
) -> str:
    """Mengklasifikasikan satu teks ke dalam satu aspek reputasi paling dominan.

    Args:
        text: Teks input.
        keywords_dict: Kamus kata kunci aspek (opsional, default DEFAULT_ASPECT_KEYWORDS).

    Returns:
        Nama aspek (string).
    """
    if not isinstance(text, str) or not text.strip():
        return "lainnya"

    kws_map = keywords_dict or DEFAULT_ASPECT_KEYWORDS
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for aspect, kws in kws_map.items():
        match_count = 0
        for kw in kws:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text_lower):
                match_count += 1
        if match_count > 0:
            scores[aspect] = match_count

    if not scores:
        return "isu_citra" if "undiksha" in text_lower else "lainnya"

    return max(scores, key=scores.get)


def classify_aspects_multi(
    text: str,
    keywords_dict: dict[str, list[str]] | None = None,
) -> list[str]:
    """Mengidentifikasi semua aspek reputasi yang disebutkan dalam satu teks (multi-aspek).

    Args:
        text: Teks input.
        keywords_dict: Kamus kata kunci aspek.

    Returns:
        Daftar nama aspek yang cocok.
    """
    if not isinstance(text, str) or not text.strip():
        return ["lainnya"]

    kws_map = keywords_dict or DEFAULT_ASPECT_KEYWORDS
    text_lower = text.lower()
    matched_aspects = []

    for aspect, kws in kws_map.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                matched_aspects.append(aspect)
                break

    return matched_aspects if matched_aspects else ["lainnya"]


def compute_aspect_sentiment_breakdown(
    df: pd.DataFrame,
    aspect_col: str = "aspect",
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Menghitung tabulasi silang (cross-tabulation) distribusi sentimen per aspek reputasi.

    Args:
        df: DataFrame berlabel.
        aspect_col: Nama kolom aspek.
        sentiment_col: Nama kolom sentimen.

    Returns:
        DataFrame ringkasan jumlah dan persentase per sentimen per aspek.
    """
    ct = pd.crosstab(df[aspect_col], df[sentiment_col], margins=True, margins_name="Total")

    # Pastikan ketiga kolom sentimen ada
    for s in ["positif", "netral", "negatif"]:
        if s not in ct.columns:
            ct[s] = 0

    ordered_cols = ["positif", "netral", "negatif", "Total"]
    ct = ct[ordered_cols].copy()

    # Hitung persentase sentimen per aspek
    ct["%_positif"] = (ct["positif"] / ct["Total"] * 100).round(2)
    ct["%_netral"] = (ct["netral"] / ct["Total"] * 100).round(2)
    ct["%_negatif"] = (ct["negatif"] / ct["Total"] * 100).round(2)

    return ct


def main() -> None:
    """CLI untuk klasifikasi dan ringkasan sentimen per aspek."""
    parser = argparse.ArgumentParser(description="Klasifikasi aspek reputasi pada data berlabel.")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "aspect" not in df.columns:
        df["aspect"] = df["text_clean"].apply(classify_aspect)

    breakdown = compute_aspect_sentiment_breakdown(df)
    print("\n" + "=" * 70)
    print("  DISTRIBUSI SENTIMEN PER ASPEK REPUTASI UNDIKSHA")
    print("=" * 70)
    print(breakdown[["positif", "netral", "negatif", "Total", "%_positif", "%_netral", "%_negatif"]])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
