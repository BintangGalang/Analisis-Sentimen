"""
src/evaluate/reputation_index.py
================================
Penghitungan Indeks Reputasi Kampus Undiksha (PRD §3a, PRD §6 F14).

Formulasi:
    Indeks Reputasi = (Jumlah Positif - Jumlah Negatif) / Total Komentar
    Rentang Nilai   : -1.00 (Sangat Buruk) s.d. +1.00 (Sangat Baik)
    Ambang Batas    :
        - Indeks >= +0.50 : Reputasi Sangat Positif
        - +0.20 s.d. +0.49: Reputasi Positif
        - -0.19 s.d. +0.19: Reputasi Netral / Moderat
        - -0.49 s.d. -0.20: Reputasi Negatif (Perlu Perhatian)
        - Indeks <= -0.50 : Reputasi Sangat Kritis

Dimensi Analisis:
1. Keseluruhan (Overall)
2. Per Aspek Reputasi (PRD §3a)
3. Per Platform Media Sosial
4. Per Periode Waktu (Bulanan)
5. Akun Resmi vs Akun Publik (Filter bias F15)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config

logger = logging.getLogger(__name__)


def interpret_reputation_index(index_value: float) -> str:
    """Menginterpretasikan nilai indeks reputasi ke dalam predikat kualitatif."""
    if index_value >= 0.50:
        return "Sangat Positif (Reputasi Unggul)"
    elif index_value >= 0.20:
        return "Positif (Apresiasi Baik)"
    elif index_value >= -0.19:
        return "Netral / Moderat"
    elif index_value >= -0.49:
        return "Negatif (Perlu Evaluasi)"
    else:
        return "Sangat Kritis (Sentimen Negatif Dominan)"


def calculate_reputation_index(
    n_pos: int,
    n_neg: int,
    total: int,
) -> float:
    """Menghitung nilai indeks reputasi: (pos - neg) / total."""
    if total <= 0:
        return 0.0
    return round(float((n_pos - n_neg) / total), 4)


def compute_reputation_by_group(
    df: pd.DataFrame,
    group_col: str,
    sentiment_col: str = "sentiment",
) -> pd.DataFrame:
    """Menghitung indeks reputasi per kelompok tertentu (aspek, platform, waktu, dll.).

    Args:
        df: DataFrame berlabel.
        group_col: Kolom pengelompokan.
        sentiment_col: Kolom sentimen.

    Returns:
        DataFrame ringkasan indeks reputasi per kelompok.
    """
    groups = df.groupby(group_col)
    rows = []

    for name, grp in groups:
        total = len(grp)
        pos = int((grp[sentiment_col] == "positif").sum())
        neg = int((grp[sentiment_col] == "negatif").sum())
        net = int((grp[sentiment_col] == "netral").sum())

        idx_val = calculate_reputation_index(pos, neg, total)

        rows.append({
            group_col: name,
            "Total": total,
            "Positif": pos,
            "Netral": net,
            "Negatif": neg,
            "Indeks_Reputasi": idx_val,
            "Indeks_%": f"{idx_val * 100:+.2f}%",
            "Predikat": interpret_reputation_index(idx_val),
        })

    out_df = pd.DataFrame(rows).sort_values(by="Indeks_Reputasi", ascending=False).reset_index(drop=True)
    return out_df


def compute_comprehensive_reputation_report(
    df: pd.DataFrame,
    aspect_col: str = "aspect",
    sentiment_col: str = "sentiment",
    date_col: str = "created_at",
    official_col: str = "is_official",
    tables_dir: str | Path = "reports/tables",
) -> dict[str, pd.DataFrame]:
    """Menghasilkan laporan lengkap indeks reputasi lintas dimensi dan menyimpannya ke CSV.

    Args:
        df: DataFrame berlabel.
        aspect_col: Nama kolom aspek.
        sentiment_col: Nama kolom sentimen.
        date_col: Nama kolom waktu/tanggal.
        official_col: Nama kolom flag akun resmi.
        tables_dir: Direktori penyimpanan tabel laporan.

    Returns:
        Dict berisi DataFrame per aspek, per waktu, dan per jenis akun.
    """
    total = len(df)
    pos = int((df[sentiment_col] == "positif").sum())
    neg = int((df[sentiment_col] == "negatif").sum())
    net = int((df[sentiment_col] == "netral").sum())
    overall_idx = calculate_reputation_index(pos, neg, total)

    df_overall = pd.DataFrame([{
        "Kategori": "Keseluruhan Kampus Undiksha",
        "Total": total,
        "Positif": pos,
        "Netral": net,
        "Negatif": neg,
        "Indeks_Reputasi": overall_idx,
        "Indeks_%": f"{overall_idx * 100:+.2f}%",
        "Predikat": interpret_reputation_index(overall_idx),
    }])

    # 1. Per Aspek
    df_aspect = compute_reputation_by_group(df, group_col=aspect_col, sentiment_col=sentiment_col)

    # 2. Akun Resmi vs Publik (PRD §3a & F15)
    if official_col in df.columns:
        df["_tipe_akun"] = df[official_col].map({True: "Akun Resmi Undiksha", False: "Akun Publik / Mahasiswa"})
        df_official = compute_reputation_by_group(df, group_col="_tipe_akun", sentiment_col=sentiment_col)
    else:
        df_official = pd.DataFrame()

    # 3. Per Tren Waktu (Bulanan jika kolom tanggal valid)
    df_timeline = pd.DataFrame()
    if date_col in df.columns and df[date_col].notna().any():
        try:
            df["_periode"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").astype(str)
            df_valid_dates = df[df["_periode"].ne("NaT")]
            if len(df_valid_dates) > 0:
                df_timeline = compute_reputation_by_group(df_valid_dates, group_col="_periode", sentiment_col=sentiment_col)
                df_timeline = df_timeline.sort_values(by="_periode").reset_index(drop=True)
        except Exception as e:
            logger.warning("Gagal mem-parsing kolom tanggal untuk tren reputasi: %s", e)

    # Ekspor tabel ke reports/tables/
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_aspect.to_csv(out_dir / "reputation_index_aspect.csv", index=False, encoding="utf-8-sig")
    if not df_timeline.empty:
        df_timeline.to_csv(out_dir / "reputation_index_timeline.csv", index=False, encoding="utf-8-sig")

    logger.info("Laporan Indeks Reputasi disimpan di %s", out_dir)

    return {
        "overall": df_overall,
        "aspect": df_aspect,
        "official": df_official,
        "timeline": df_timeline,
    }


def main() -> None:
    """CLI untuk menghitung indeks reputasi Undiksha."""
    parser = argparse.ArgumentParser(description="Hitung Indeks Reputasi Undiksha.")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    report = compute_comprehensive_reputation_report(df)

    print("\n" + "=" * 75)
    print("  INDEKS REPUTASI KESELURUHAN (UNDIKSHA)")
    print("=" * 75)
    print(report["overall"].to_string(index=False))

    print("\n" + "=" * 75)
    print("  INDEKS REPUTASI PER ASPEK")
    print("=" * 75)
    print(report["aspect"][["aspect", "Total", "Positif", "Negatif", "Indeks_%", "Predikat"]].to_string(index=False))

    if not report["official"].empty:
        print("\n" + "=" * 75)
        print("  INDEKS REPUTASI: AKUN RESMI VS AKUN PUBLIK")
        print("=" * 75)
        print(report["official"][["_tipe_akun", "Total", "Positif", "Negatif", "Indeks_%", "Predikat"]].to_string(index=False))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
