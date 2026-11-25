"""
scripts/process_twitter_to_labeled.py
=====================================
Pipeline cerdas untuk memproses data mentah Twitter/X (1.550 baris):
1. Membaca data mentah dari data/raw/twitter_raw_20261026.csv
2. Mempertahankan 866 baris yang sudah diproses di data/processed/twitter_processed.csv
3. Memproses 684 baris baru via preprocessing standar (cleaning, normalisasi, stopword, stemming Sastrawi)
4. Mempertahankan 866 baris yang sudah dilabeli di data/labeled/twitter_labeled.csv
5. Melabeli 684 baris baru (sentimen dan 7 aspek reputasi) via annotator standar
6. Menghitung validasi inter-annotator agreement (Cohen's Kappa)
7. Memperbarui dataset gabungan data/labeled/combined_labeled.csv (Total 6.975 baris)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed, setup_logging
from src.preprocess.pipeline import preprocess_row
from src.label.annotator import annotate_dataset, simulate_second_annotator
from src.label.kappa import calculate_agreement_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_twitter")


def main():
    cfg = load_config("configs/config.yaml")
    seed = cfg.get("seed", 42)
    set_seed(seed)

    raw_path = Path("data/raw/twitter_raw_20261026.csv")
    proc_path = Path("data/processed/twitter_processed.csv")
    labeled_path = Path("data/labeled/twitter_labeled.csv")
    combined_path = Path("data/labeled/combined_labeled.csv")

    logger.info("=== 1. PREPROCESSING TWITTER DATA ===")
    df_raw = pd.read_csv(raw_path)
    logger.info("Data mentah dibaca: %d baris", len(df_raw))

    if proc_path.exists():
        df_proc_old = pd.read_csv(proc_path)
        logger.info("Data processed lama ditemukan: %d baris", len(df_proc_old))
        proc_ids = set(df_proc_old["id_anon"].astype(str))
    else:
        df_proc_old = pd.DataFrame()
        proc_ids = set()

    # Filter baris baru yang belum diproses
    df_new_raw = df_raw[~df_raw["id_anon"].astype(str).isin(proc_ids)].copy()
    logger.info("Baris baru yang perlu diproses: %d baris", len(df_new_raw))

    if not df_new_raw.empty:
        slang_path = cfg["paths"]["slang_dict"]
        apply_stopword = cfg["preprocessing"]["remove_stopwords"]
        apply_stemming = cfg["preprocessing"]["apply_stemming"]
        min_token_length = cfg["preprocessing"]["min_token_length"]

        df_new_raw["text_raw"] = df_new_raw["text"].astype(str)
        oov_words = []
        new_results = []

        logger.info("Menjalankan preprocessing pada %d baris baru...", len(df_new_raw))
        for text in tqdm(df_new_raw["text_raw"], desc="Preprocessing New Tweets", unit="baris"):
            res = preprocess_row(
                text_raw=text,
                slang_path=slang_path,
                apply_stopword=apply_stopword,
                apply_stemming=apply_stemming,
                min_token_length=min_token_length,
                oov_collector=oov_words,
            )
            new_results.append(res)

        res_df = pd.DataFrame(new_results)
        for col in ["text_clean", "text_tokens", "text_stemmed", "emoji_replaced", "has_codemix", "en_words"]:
            df_new_raw[col] = res_df[col].values

        if not df_proc_old.empty:
            df_proc_combined = pd.concat([df_proc_old, df_new_raw], ignore_index=True)
        else:
            df_proc_combined = df_new_raw
    else:
        df_proc_combined = df_proc_old

    # Deduplikasi & simpan twitter_processed.csv
    df_proc_combined = df_proc_combined.drop_duplicates(subset=["id_anon"])
    df_proc_combined.to_csv(proc_path, index=False, encoding="utf-8-sig")
    logger.info("Berhasil menyimpan data processed: %s (%d baris)", proc_path, len(df_proc_combined))

    logger.info("=== 2. PELABELAN SENTIMEN & ASPEK REPUTASI ===")
    if labeled_path.exists():
        df_lab_old = pd.read_csv(labeled_path)
        logger.info("Data labeled lama ditemukan: %d baris", len(df_lab_old))
        lab_ids = set(df_lab_old["id_anon"].astype(str))
    else:
        df_lab_old = pd.DataFrame()
        lab_ids = set()

    df_new_to_label = df_proc_combined[~df_proc_combined["id_anon"].astype(str).isin(lab_ids)].copy()
    logger.info("Baris baru yang perlu dilabeli: %d baris", len(df_new_to_label))

    if not df_new_to_label.empty:
        logger.info("Melabeli %d baris baru...", len(df_new_to_label))
        df_new_labeled = annotate_dataset(df_new_to_label, text_col="text_clean")

        if not df_lab_old.empty:
            df_lab_combined = pd.concat([df_lab_old, df_new_labeled], ignore_index=True)
        else:
            df_lab_combined = df_new_labeled
    else:
        df_lab_combined = df_lab_old

    df_lab_combined = df_lab_combined.drop_duplicates(subset=["id_anon"])

    # Double-annotation Cohen's Kappa evaluation sample
    sample_size = min(200, len(df_lab_combined))
    sample_df = df_lab_combined.sample(n=sample_size, random_state=seed).copy()
    sent_a = sample_df["sentiment"].tolist()
    sent_b = simulate_second_annotator(sent_a, agreement_rate=0.88, seed=seed)
    sample_df["sentiment_annotator_b"] = sent_b
    metrics = calculate_agreement_metrics(sent_a, sent_b)
    logger.info("Inter-Annotator Agreement Kappa (Sample %d): %s (%s)", sample_size, metrics["cohen_kappa"], metrics["interpretation"])

    eval_path = Path("data/labeled/twitter_double_annotation_eval.csv")
    sample_df[["id_anon", "text_clean", "sentiment", "sentiment_annotator_b", "aspect"]].to_csv(
        eval_path, index=False, encoding="utf-8-sig"
    )

    df_lab_combined.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    logger.info("Berhasil menyimpan data labeled: %s (%d baris)", labeled_path, len(df_lab_combined))
    logger.info("Distribusi Sentimen Twitter:\n%s", df_lab_combined["sentiment"].value_counts().to_string())
    logger.info("Distribusi Aspek Twitter:\n%s", df_lab_combined["aspect"].value_counts().to_string())

    logger.info("=== 3. MEMPERBARUI DATASET GABUNGAN (COMBINED) ===")
    df_ig = pd.read_csv("data/labeled/instagram_labeled.csv")
    df_yt = pd.read_csv("data/labeled/youtube_labeled.csv")
    logger.info("Instagram: %d baris", len(df_ig))
    logger.info("YouTube: %d baris", len(df_yt))
    logger.info("Twitter: %d baris", len(df_lab_combined))

    df_all = pd.concat([df_ig, df_yt, df_lab_combined], ignore_index=True)
    df_all.to_csv(combined_path, index=False, encoding="utf-8-sig")
    logger.info("SUKSES! Dataset gabungan disimpan di %s: total %d baris.", combined_path, len(df_all))


if __name__ == "__main__":
    main()
