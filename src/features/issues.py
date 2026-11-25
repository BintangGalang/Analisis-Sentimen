"""
src/features/issues.py
======================
Modul Deteksi Isu dan Penilaian Lonjakan Mingguan (PRD Langkah 4).

Fitur:
1. Agregasi volume sebutan per aspek dan ekstraksi frasa kunci (unigram + bigram TF-IDF / c-TF-IDF).
2. Perhitungan proporsi sentimen negatif per topik dan aspek.
3. Deteksi lonjakan isu: Membandingkan 7 hari ini dengan rata-rata 4 minggu sebelumnya (growth ratio).
4. Perhitungan Skor Isu (Issue Score) terbobot:
   Skor = (w_vol * V_norm + w_neg * P_neg + w_growth * Growth_norm) * 100
5. Ambang batas min_mentions untuk menghindari kesimpulan keliru pada sampel kecil (Peringatan 'Data Terbatas').
6. Ekstraksi Top 5 Isu teratas dengan 3 contoh komentar representatif (anonim).
7. Penyimpanan hasil terstruktur ke data/processed/weekly_issues.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config

logger = logging.getLogger("issue_detector")
logger.setLevel(logging.INFO)

if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(sh)


INDONESIAN_STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "ke", "dari", "untuk", "pada", "adalah",
    "dengan", "dalam", "bisa", "ada", "tidak", "juga", "akan", "sudah", "saya",
    "kamu", "dia", "mereka", "kita", "kami", "sangat", "lebih", "karena", "agar",
    "undiksha", "singaraja", "bali", "kampus", "universitas", "deh", "sih", "dong",
    "ya", "nih", "kok", "kan", "tuh", "kah", "lah", "pun", "nya", "banget", "bgt",
    "aja", "saja", "kalo", "kalau", "tp", "tapi", "yg", "dgn", "utk", "dr", "krn",
    "sy", "gw", "lu", "lo", "gua", "min", "admin", "halo", "kak", "kakak", "ga", "gak",
}


class WeeklyIssueDetector:
    """Detektor isu media sosial mingguan kampus Undiksha."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config = load_config(config_path)
        wm_cfg = self.config.get("weekly_monitoring", {})

        self.min_mentions = int(wm_cfg.get("min_mentions", 5))
        self.weights = wm_cfg.get("weights", {"volume": 0.30, "sentiment": 0.45, "growth": 0.25})
        self.thresholds = wm_cfg.get("thresholds", {"high_alert": 65.0, "medium_alert": 40.0})
        self.predictions_path = PROJECT_ROOT / wm_cfg.get("predictions_file", "data/processed/weekly_predictions.parquet")
        self.issues_output_path = PROJECT_ROOT / wm_cfg.get("issues_file", "data/processed/weekly_issues.csv")

    def _extract_top_phrases_per_aspect(
        self,
        df_aspect: pd.DataFrame,
        top_k: int = 3,
    ) -> list[str]:
        """Ekstraksi frasa kunci unigram dan bigram representatif menggunakan TF-IDF.
        Memprioritaskan teks dari opini negatif agar langsung merefleksikan inti keluhan/masalah publik.
        """
        # Prioritaskan opini negatif jika tersedia untuk mengungkap akar isu
        neg_df = df_aspect[df_aspect["sentimen"].astype(str).str.lower() == "negatif"]
        source_df = neg_df if (not neg_df.empty and len(neg_df) >= 2) else df_aspect

        texts = [
            str(t) for t in source_df["teks_bersih"]
            if pd.notna(t) and len(str(t).strip()) > 3
        ]
        if not texts:
            return ["isu umum"]

        try:
            vec = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=list(INDONESIAN_STOPWORDS),
                max_features=50,
                min_df=1,
            )
            tfidf_mat = vec.fit_transform(texts)
            feature_names = vec.get_feature_names_out()
            scores = np.asarray(tfidf_mat.sum(axis=0)).ravel()
            top_indices = scores.argsort()[::-1][:top_k]
            top_phrases = [feature_names[i] for i in top_indices if len(feature_names[i].split()) > 1 or len(feature_names[i]) > 3]
            if not top_phrases:
                top_phrases = [feature_names[i] for i in top_indices[:top_k]]
            return top_phrases if top_phrases else ["topik terkini"]
        except Exception:
            return ["isu kampus"]

    def calculate_growth_ratio(
        self,
        vol_curr: int,
        past_volumes: list[int],
    ) -> float:
        """Menghitung rasio lonjakan dibanding rata-rata 4 minggu sebelumnya dengan Laplace smoothing.

        Formula:
            Growth Ratio = (vol_curr + 1) / (mean(past_4_weeks) + 1)
        """
        if not past_volumes:
            # Baseline awal jika belum ada data historis sebelumnya
            return 1.0
        avg_past = float(np.mean(past_volumes))
        return (vol_curr + 1.0) / (avg_past + 1.0)

    def calculate_issue_score(
        self,
        vol: int,
        max_vol: int,
        neg_ratio: float,
        growth_ratio: float,
    ) -> float:
        """Menghitung Skor Isu (0 - 100) berdasarkan bobot PRD.

        Formula:
            Skor = (w_vol * min(1.0, vol/max_vol) +
                    w_neg * neg_ratio +
                    w_growth * min(1.0, growth_ratio/5.0)) * 100
        """
        w_vol = self.weights.get("volume", 0.30)
        w_neg = self.weights.get("sentiment", 0.45)
        w_growth = self.weights.get("growth", 0.25)

        norm_vol = min(1.0, (vol / max(1, max_vol)))
        norm_growth = min(1.0, (growth_ratio / 5.0))
        norm_neg = min(1.0, max(0.0, neg_ratio))

        raw_score = (w_vol * norm_vol + w_neg * norm_neg + w_growth * norm_growth) * 100.0
        return round(float(raw_score), 2)

    def detect_issues_for_week(
        self,
        df_all: pd.DataFrame,
        target_week: str | None = None,
    ) -> pd.DataFrame:
        """Menganalisis isu mingguan pada target_week."""
        if df_all.empty:
            logger.warning("Dataset prediksi kosong.")
            return pd.DataFrame()

        # Ambil daftar minggu terurut
        all_weeks = sorted(df_all["minggu"].dropna().unique().tolist())
        if not all_weeks:
            return pd.DataFrame()

        target_week = target_week or all_weeks[-1]
        logger.info("Menganalisis isu mingguan untuk periode: %s", target_week)

        # Pisahkan data minggu target dan historis sebelumnya
        df_target = df_all[df_all["minggu"] == target_week]
        target_idx = all_weeks.index(target_week)
        past_weeks = all_weeks[max(0, target_idx - 4):target_idx]

        # Filter postingan promosi / iklan agar tidak mengotori isu publik
        if "is_spam_promo" in df_target.columns:
            df_target = df_target[~df_target["is_spam_promo"]]

        total_mentions = len(df_target)
        is_limited_data = total_mentions < self.min_mentions

        if is_limited_data:
            logger.warning(
                "Jumlah sebutan minggu %s hanya %d (di bawah ambang min_mentions: %d). Menandai 'Data Terbatas'.",
                target_week, total_mentions, self.min_mentions
            )

        aspect_groups = df_target.groupby("aspek")
        max_aspect_vol = aspect_groups.size().max() if not aspect_groups.size().empty else 1

        issue_rows = []

        for aspect_name, group in aspect_groups:
            aspect_name_str = str(aspect_name)
            vol_curr = len(group)

            # Jika data di bawah ambang dan tidak ada lonjakan signifikan, lewati aspek minor
            if vol_curr < 2 and is_limited_data:
                continue

            # Hitung proporsi sentimen
            sent_counts = group["sentimen"].astype(str).str.lower().value_counts()
            neg_count = int(sent_counts.get("negatif", 0))
            pos_count = int(sent_counts.get("positif", 0))
            net_count = int(sent_counts.get("netral", 0))
            neg_ratio = neg_count / vol_curr if vol_curr > 0 else 0.0

            # Hitung volume masa lalu untuk aspek ini
            past_vols = []
            for pw in past_weeks:
                df_pw = df_all[(df_all["minggu"] == pw) & (df_all["aspek"] == aspect_name)]
                past_vols.append(len(df_pw))

            growth_ratio = self.calculate_growth_ratio(vol_curr, past_vols)

            # Perubahan persentase dari minggu kemarin
            prev_week_vol = past_vols[-1] if past_vols else 0
            if prev_week_vol > 0:
                change_pct = round(((vol_curr - prev_week_vol) / prev_week_vol) * 100.0, 1)
                change_str = f"{change_pct:+g}%"
            elif prev_week_vol == 0 and vol_curr > 0:
                change_str = "+100% (Baru)"
            else:
                change_str = "0%"

            # Skor Isu
            score = self.calculate_issue_score(
                vol=vol_curr,
                max_vol=max_aspect_vol,
                neg_ratio=neg_ratio,
                growth_ratio=growth_ratio,
            )

            # Tingkat perhatian
            if score >= self.thresholds.get("high_alert", 65.0):
                alert_level = "Tinggi"
            elif score >= self.thresholds.get("medium_alert", 40.0):
                alert_level = "Sedang"
            else:
                alert_level = "Rendah"

            # Ekstraksi kata/frasa kunci isu
            top_phrases = self._extract_top_phrases_per_aspect(group, top_k=2)
            clean_aspect_label = aspect_name_str.replace("_", " ").title()
            judul_isu = f"Isu {clean_aspect_label}: " + ", ".join(top_phrases[:2]).title()

            # Pilih 3 contoh komentar representatif (utamakan sentimen negatif / keluhan nyata)
            neg_samples = group[group["sentimen"].astype(str).str.lower() == "negatif"]["teks_asli_anonim"].tolist()
            other_samples = group["teks_asli_anonim"].tolist()
            combined_samples = (neg_samples + other_samples)[:3]
            while len(combined_samples) < 3:
                combined_samples.append("-")

            is_priority_neg = (neg_ratio >= 0.20) or (neg_count >= 2) or (alert_level == "Tinggi")

            issue_rows.append({
                "minggu": target_week,
                "judul_isu": judul_isu,
                "aspek": aspect_name_str,
                "aspek_label": clean_aspect_label,
                "sebutan": vol_curr,
                "negatif_count": neg_count,
                "persen_negatif": round(neg_ratio * 100.0, 1),
                "persen_positif": round((pos_count / vol_curr) * 100.0, 1),
                "persen_netral": round((net_count / vol_curr) * 100.0, 1),
                "rasio_lonjakan": round(growth_ratio, 2),
                "perubahan_minggu_lalu": change_str,
                "skor_isu": score,
                "tingkat_perhatian": alert_level,
                "is_prioritas_negatif": is_priority_neg,
                "status_data": "Data Terbatas" if is_limited_data else "Valid",
                "contoh_komentar_1": str(combined_samples[0])[:250],
                "contoh_komentar_2": str(combined_samples[1])[:250],
                "contoh_komentar_3": str(combined_samples[2])[:250],
            })

        df_issues = pd.DataFrame(issue_rows)
        if not df_issues.empty:
            # Urutkan berdasarkan skor isu tertinggi dan persentase sentimen negatif (utamakan isu kritis)
            df_issues = df_issues.sort_values(
                by=["skor_isu", "persen_negatif", "sebutan"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

        return df_issues

    def detect_and_save_all(
        self,
        df_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Mendeteksi isu untuk seluruh minggu yang tersedia dan memperbarui weekly_issues.csv."""
        if df_predictions is None:
            if not self.predictions_path.exists():
                logger.error("File prediksi tidak ditemukan: %s", self.predictions_path)
                return pd.DataFrame()
            df_predictions = pd.read_parquet(self.predictions_path)

        all_weeks = sorted(df_predictions["minggu"].dropna().unique().tolist())
        logger.info("Mengevaluasi isu untuk %d minggu: %s", len(all_weeks), all_weeks)

        all_issue_dfs = []
        for w in all_weeks:
            w_issues = self.detect_issues_for_week(df_predictions, target_week=w)
            if not w_issues.empty:
                all_issue_dfs.append(w_issues)

        if not all_issue_dfs:
            logger.warning("Tidak ada isu yang berhasil dihasilkan.")
            return pd.DataFrame()

        df_all_issues = pd.concat(all_issue_dfs, ignore_index=True)

        # Simpan ke CSV
        self.issues_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_all_issues.to_csv(self.issues_output_path, index=False, encoding="utf-8-sig")
        logger.info("Hasil deteksi isu berhasil disimpan ke: %s (%d baris)", self.issues_output_path, len(df_all_issues))

        return df_all_issues


def main():
    parser = argparse.ArgumentParser(description="Deteksi Isu Mingguan & Skor Lonjakan Undiksha")
    parser.add_argument("--week", type=str, default=None, help="Spesifikasikan minggu ISO (misal: 2026-W47)")
    args = parser.parse_args()

    detector = WeeklyIssueDetector()
    df_res = detector.detect_and_save_all()

    if not df_res.empty:
        print("\n=== TOP 5 ISU MINGGU INI ===")
        latest_week = df_res["minggu"].iloc[-1]
        top5 = df_res[df_res["minggu"] == latest_week].head(5)
        for idx, row in top5.iterrows():
            print(f"[{row['tingkat_perhatian'].upper()}] Skor: {row['skor_isu']} | {row['judul_isu']}")
            print(f"   Sebutan: {row['sebutan']} | Negatif: {row['persen_negatif']}% | Lonjakan: {row['perubahan_minggu_lalu']}")
            print(f"   Contoh: \"{row['contoh_komentar_1'][:120]}...\"\n")
    else:
        print("Belum ada data isu.")


if __name__ == "__main__":
    main()

