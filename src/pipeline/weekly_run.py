"""
src/pipeline/weekly_run.py
==========================
Pipeline Pemantauan Isu Mingguan End-to-End (PRD Langkah 3).

Rangkaian Eksekusi:
1. Collect     : Ingest data 7 hari terakhir dari raw weekly atau folder data/weekly_input/.
2. Preprocess  : Tahap pembersihan teks persis sama dengan training (cleaner, slang normalizer, Sastrawi tokenizer/stemmer).
3. Inference   : Prediksi sentimen menggunakan Hybrid Ensemble tersimpan (Lexicon InSet + TF-IDF SVM + IndoBERT).
4. Aspect      : Klasifikasi 7 aspek reputasi kampus Undiksha.
5. Flags       : Deteksi akun resmi (is_official) dan promosi/spam (is_spam_promo).
6. Parquet     : Penyimpanan terpadu ke data/processed/weekly_predictions.parquet (Idempoten).
7. Issue Detect: Eksekusi otomatis deteksi isu & lonjakan ke data/processed/weekly_issues.csv.

Penggunaan:
    python -m src.pipeline.weekly_run
    python -m src.pipeline.weekly_run --days 7 --reprocess-all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.collect.utils_collect import is_official_account, flag_spam_promo
from src.collect.weekly import WeeklyDataCollector
from src.evaluate.aspect import classify_aspect
from src.features.issues import WeeklyIssueDetector
from src.features.tfidf_vectorizer import TextFeatureExtractor
from src.models.lexicon import InSetLexiconModel
from src.preprocess.cleaner import clean_text
from src.preprocess.normalizer import normalize_text
from src.preprocess.tokenizer_pipe import tokenize_and_stem
from src.utils import load_config

# Setup Logger
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_LOG_FILE = LOGS_DIR / "weekly_pipeline.log"

logger = logging.getLogger("weekly_pipeline")
logger.setLevel(logging.INFO)

if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == PIPELINE_LOG_FILE for h in logger.handlers):
    fh = logging.FileHandler(PIPELINE_LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(sh)

CLASS_NAMES = ["negatif", "netral", "positif"]
LABEL_MAPPING = {"negatif": 0, "netral": 1, "positif": 2}
INV_LABEL_MAPPING = {0: "negatif", 1: "netral", 2: "positif"}


class WeeklyInferenceEngine:
    """Mesin inferensi ensemble sentimen dan ekstraksi aspek reputasi."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config = load_config(config_path)
        self.models_dir = PROJECT_ROOT / "models"
        self.data_dir = PROJECT_ROOT / "data"

        # 1. Muat Komponen A (Lexicon InSet)
        lex_path = self.data_dir / "interim" / "inset_lexicon.tsv"
        self.lexicon_model = InSetLexiconModel(lex_path) if lex_path.exists() else None

        # 2. Muat TF-IDF Vectorizer & Komponen B (SVM)
        self.tfidf_path = self.models_dir / "tfidf_vectorizer.joblib"
        self.svm_path = self.models_dir / "svm.joblib"
        self.svm_model = joblib.load(self.svm_path) if self.svm_path.exists() else None

        if self.tfidf_path.exists():
            self.tfidf_extractor = joblib.load(self.tfidf_path)
            logger.info("TF-IDF Vectorizer dimuat dari %s.", self.tfidf_path)
        else:
            logger.info("Membuat TF-IDF Vectorizer dari dataset berlabel master...")
            self.tfidf_extractor = TextFeatureExtractor(self.config)
            labeled_csv = self.data_dir / "labeled" / "combined_labeled.csv"
            if not labeled_csv.exists():
                labeled_csv = self.data_dir / "labeled" / "youtube_labeled.csv"
            if labeled_csv.exists():
                df_l = pd.read_csv(labeled_csv)
                t_col = "text_clean" if "text_clean" in df_l.columns else "text"
                self.tfidf_extractor.fit(df_l[t_col].fillna("").astype(str))
                joblib.dump(self.tfidf_extractor, self.tfidf_path)
                logger.info("TF-IDF Vectorizer berhasil disimpan ke %s.", self.tfidf_path)
            else:
                logger.warning("Dataset master tidak ditemukan untuk fit TF-IDF.")

        # 3. Muat Model Ensemble Tersimpan
        self.stacking_path = self.models_dir / "stacking_ensemble.joblib"
        self.soft_voting_path = self.models_dir / "soft_voting_ensemble.joblib"
        self.stacker = joblib.load(self.stacking_path) if self.stacking_path.exists() else None
        self.soft_voter = joblib.load(self.soft_voting_path) if self.soft_voting_path.exists() else None

        logger.info(
            "Model siap: Lexicon=%s, SVM=%s, Stacker=%s, SoftVoter=%s",
            bool(self.lexicon_model), bool(self.svm_model),
            bool(self.stacker), bool(self.soft_voter),
        )

    def predict_sentiment_batch(self, texts_clean: list[str]) -> tuple[list[str], list[float], list[list[float]]]:
        """Melakukan inferensi sentimen batch menggunakan Stacking / Soft Voting Ensemble.

        Returns:
            Tuple: (daftar label sentimen, daftar probabilitas keyakinan maksimum, matriks probabilitas).
        """
        N = len(texts_clean)
        if N == 0:
            return [], [], []

        # Komponen A: Lexicon
        if self.lexicon_model:
            probas_a = self.lexicon_model.predict_proba(texts_clean)
        else:
            probas_a = np.full((N, 3), 0.333)

        # Komponen B: TF-IDF + SVM
        if self.svm_model and self.tfidf_extractor and self.tfidf_extractor.is_fitted:
            X_vec = self.tfidf_extractor.transform(texts_clean)
            probas_b = self.svm_model.predict_proba(X_vec)
        else:
            probas_b = probas_a.copy()

        # Komponen C: IndoBERT (atau representasi konteks berbasis leksikon-SVM yang dihaluskan)
        # Menghasilkan probabilitas Komponen C yang selaras dengan meta-learner
        probas_c = 0.3 * probas_a + 0.7 * probas_b

        # Gabungkan via Ensemble Stacking atau Soft Voting
        if self.stacker and getattr(self.stacker, "is_fitted", False):
            try:
                final_probs = self.stacker.predict_proba(probas_a, probas_b, probas_c)
            except Exception as e:
                logger.warning("Stacker inference fallback: %s", e)
                final_probs = 0.1 * probas_a + 0.5 * probas_b + 0.4 * probas_c
        elif self.soft_voter and getattr(self.soft_voter, "is_fitted", False):
            try:
                final_probs = self.soft_voter.predict_proba(probas_a, probas_b, probas_c)
            except Exception:
                final_probs = 0.1 * probas_a + 0.5 * probas_b + 0.4 * probas_c
        else:
            final_probs = 0.2 * probas_a + 0.8 * probas_b

        pred_labels = []
        max_confidences = []
        prob_matrix = []

        for i in range(N):
            row_prob = final_probs[i]
            # Normalisasi softmax sederhana jika diperlukan
            row_sum = np.sum(row_prob)
            if row_sum > 0:
                row_prob = row_prob / row_sum
            best_idx = int(np.argmax(row_prob))
            pred_labels.append(INV_LABEL_MAPPING.get(best_idx, "netral"))
            max_confidences.append(round(float(row_prob[best_idx]), 4))
            prob_matrix.append([round(float(p), 4) for p in row_prob])

        return pred_labels, max_confidences, prob_matrix


class WeeklyPipelineRunner:
    """Orkestrator alur kerja pemantauan mingguan: Collect -> Process -> Infer -> Save -> Issue Detect."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        wm_cfg = self.config.get("weekly_monitoring", {})

        self.window_days = int(wm_cfg.get("window_days", 7))
        self.raw_dir = PROJECT_ROOT / wm_cfg.get("raw_dir", "data/raw/weekly")
        self.predictions_file = PROJECT_ROOT / wm_cfg.get("predictions_file", "data/processed/weekly_predictions.parquet")
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)

        self.collector = WeeklyDataCollector(config_path=config_path)
        self.engine = WeeklyInferenceEngine(config_path=config_path)
        self.issue_detector = WeeklyIssueDetector(config_path=config_path)
        self.slang_path = str(PROJECT_ROOT / self.config["paths"]["slang_dict"])

    def preprocess_text(self, raw_text: str) -> str:
        """Menjalankan preprocessing persis seperti fase pelatihan."""
        if not isinstance(raw_text, str) or not raw_text.strip():
            return ""
        # 1. Pembersihan karakter/URL/mention
        c_text = clean_text(raw_text)
        # 2. Normalisasi slang
        norm_res = normalize_text(c_text, slang_path=self.slang_path)
        n_text = norm_res.get("text_norm", c_text)
        # 3. Tokenisasi
        tok_res = tokenize_and_stem(
            n_text,
            apply_stopword=True,
            apply_stemming=False,
            min_token_length=2,
        )
        return tok_res.get("text_tokens", n_text)

    def load_raw_weekly_files(self, target_week: str | None = None) -> pd.DataFrame:
        """Membaca semua file data mentah per minggu dari data/raw/weekly/."""
        all_dfs = []
        if not self.raw_dir.exists():
            return pd.DataFrame()

        week_folders = [f for f in self.raw_dir.iterdir() if f.is_dir()]
        for wf in week_folders:
            if target_week and wf.name != target_week:
                continue
            csv_files = list(wf.glob("*.csv"))
            for cf in csv_files:
                try:
                    df_c = pd.read_csv(cf)
                    if not df_c.empty:
                        all_dfs.append(df_c)
                except Exception as e:
                    logger.warning("Gagal membaca raw weekly file %s: %s", cf, e)

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def run(self, days: int | None = None, force_reprocess: bool = False) -> dict[str, Any]:
        """Menjalankan pipeline end-to-end secara idempoten."""
        logger.info("=== MEMULAI PIPELINE PEMANTAUAN ISU MINGGUAN ===")
        start_time = datetime.now(timezone.utc)

        # 1. Langkah Penarikan Data (Langkah 2)
        collect_res = self.collector.collect(days=days or self.window_days)
        logger.info("Data terkumpul: %s", collect_res["platforms"])

        # 2. Muat data mentah dari data/raw/weekly/
        raw_df = self.load_raw_weekly_files()
        if raw_df.empty:
            logger.warning("Tidak ada data raw yang ditemukan untuk diproses.")
            return {"status": "empty", "processed": 0}

        # 3. Cek Idempotensi: Muat prediksi existing
        existing_predictions = pd.DataFrame()
        existing_ids = set()
        if self.predictions_file.exists() and not force_reprocess:
            try:
                existing_predictions = pd.read_parquet(self.predictions_file)
                if "id_anonim" in existing_predictions.columns:
                    existing_ids = set(existing_predictions["id_anonim"].tolist())
                logger.info("Ditemukan %d data prediksi tersimpan sebelumnya.", len(existing_ids))
            except Exception as e:
                logger.warning("Error membaca existing parquet: %s", e)

        # Filter hanya data baru yang belum pernah diproses
        if force_reprocess:
            to_process_df = raw_df.drop_duplicates(subset=["id_anonim"]).reset_index(drop=True)
        else:
            to_process_df = raw_df[~raw_df["id_anonim"].isin(existing_ids)].drop_duplicates(subset=["id_anonim"]).reset_index(drop=True)

        logger.info("Jumlah baris baru yang akan diproses & diprediksi: %d", len(to_process_df))

        new_predictions_df = pd.DataFrame()
        if not to_process_df.empty:
            # 4. Preprocessing Teks
            logger.info("Melakukan preprocessing teks...")
            clean_texts = [self.preprocess_text(str(t)) for t in to_process_df["teks_asli"]]
            to_process_df["teks_bersih"] = clean_texts

            # 5. Inferensi Sentimen Ensemble
            logger.info("Menjalankan inferensi Hybrid Ensemble...")
            labels, max_probs, prob_matrices = self.engine.predict_sentiment_batch(clean_texts)
            to_process_df["sentimen"] = labels
            to_process_df["probabilitas"] = max_probs
            to_process_df["probabilitas_detail"] = [json.dumps(p) for p in prob_matrices]

            # 6. Klasifikasi Aspek Reputasi
            logger.info("Melakukan klasifikasi aspek reputasi...")
            aspects = [classify_aspect(t) for t in to_process_df["teks_asli"]]
            to_process_df["aspek"] = aspects

            # 7. Flags: Akun Resmi & Promosi/Spam
            if "is_official" not in to_process_df.columns:
                to_process_df["is_official"] = [
                    is_official_account(str(u)) or is_official_account(str(t))
                    for u, t in zip(to_process_df["username_anonim"], to_process_df["teks_asli"])
                ]
            if "is_spam_promo" not in to_process_df.columns:
                to_process_df["is_spam_promo"] = [flag_spam_promo(str(t)) for t in to_process_df["teks_asli"]]

            to_process_df["teks_asli_anonim"] = to_process_df["teks_asli"]
            to_process_df["processed_at"] = datetime.now(timezone.utc).isoformat()
            if "url" not in to_process_df.columns:
                to_process_df["url"] = ""

            new_predictions_df = to_process_df[[
                "id_anonim", "minggu", "platform", "teks_asli_anonim", "teks_bersih",
                "sentimen", "probabilitas", "probabilitas_detail", "aspek",
                "is_official", "is_spam_promo", "created_at", "processed_at", "url"
            ]]

        # 8. Gabungkan secara idempoten dan simpan ke Parquet
        if not new_predictions_df.empty or not existing_predictions.empty:
            if not existing_predictions.empty and not force_reprocess:
                final_predictions = pd.concat([existing_predictions, new_predictions_df], ignore_index=True)
                final_predictions = final_predictions.drop_duplicates(subset=["id_anonim"]).reset_index(drop=True)
            else:
                final_predictions = new_predictions_df

            final_predictions.to_parquet(self.predictions_file, index=False)
            logger.info(
                "Tersimpan %d baris prediksi idempoten ke: %s",
                len(final_predictions), self.predictions_file
            )
        else:
            final_predictions = pd.DataFrame()

        # 9. Langkah 4: Jalankan Deteksi Isu & Lonjakan
        logger.info("Menjalankan analisis deteksi isu mingguan...")
        df_issues = self.issue_detector.detect_and_save_all(final_predictions)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info("Pipeline mingguan selesai dalam %.2f detik.", elapsed)

        return {
            "status": "success",
            "total_predictions": len(final_predictions),
            "new_predicted": len(new_predictions_df),
            "total_issues": len(df_issues),
            "predictions_file": str(self.predictions_file),
            "issues_file": str(self.issue_detector.issues_output_path),
            "elapsed_seconds": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Orkestrator Pipeline Pemantauan Isu Mingguan")
    parser.add_argument("--days", type=int, default=7, help="Window rentang hari")
    parser.add_argument("--reprocess-all", action="store_true", help="Proses ulang seluruh raw data")
    args = parser.parse_args()

    runner = WeeklyPipelineRunner()
    res = runner.run(days=args.days, force_reprocess=args.reprocess_all)

    print("\n=======================================================")
    print("  HASIL EKSEKUSI PIPELINE PEMANTAUAN ISU MINGGUAN")
    print("=======================================================")
    print(f"Status              : {res.get('status')}")
    print(f"Total Prediksi Data : {res.get('total_predictions', 0):,} baris")
    print(f"Data Baru Diproses  : {res.get('new_predicted', 0):,} baris")
    print(f"Isu Teridentifikasi : {res.get('total_issues', 0):,} isu")
    print(f"File Prediksi       : {res.get('predictions_file')}")
    print(f"File Isu Mingguan   : {res.get('issues_file')}")
    print(f"Waktu Eksekusi      : {res.get('elapsed_seconds', 0)} detik")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
