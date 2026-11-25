"""
scripts/relabel_with_llm.py
---------------------------
Melabeli ulang dataset sentimen & aspek Undiksha menggunakan LLM Hugging Face
(HFLLMLabeler), memindahkan iklan/spam ke excluded_spam.csv, serta opsional
memperbarui weekly_predictions.parquet & weekly_issues.csv.

Contoh penggunaan:
    python scripts/relabel_with_llm.py --dry-run --limit 50
    python scripts/relabel_with_llm.py --only-conflict
    python scripts/relabel_with_llm.py --all --reprocess-weekly
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.llm_labeler import HFLLMLabeler
from src.utils import load_config
from src.utils.logger import logger

DATA_COMBINED_PATH = PROJECT_ROOT / "data" / "labeled" / "combined_labeled.csv"
EXCLUDED_SPAM_PATH = PROJECT_ROOT / "data" / "interim" / "excluded_spam.csv"
WEEKLY_PRED_PATH = PROJECT_ROOT / "data" / "processed" / "weekly_predictions.parquet"


def relabel_master_dataset(
    dry_run: bool = False,
    only_conflict: bool = False,
    limit: int | None = None,
    batch_size: int | None = None,
    reprocess_weekly: bool = False,
) -> dict:
    """Jalankan pelabelan ulang LLM pada combined_labeled.csv."""
    if not DATA_COMBINED_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_COMBINED_PATH}")

    config = load_config()
    labeler = HFLLMLabeler.from_config(config)
    if batch_size:
        labeler.batch_size = max(1, int(batch_size))

    if not labeler.is_available():
        raise RuntimeError("HF_TOKEN tidak ditemukan di environment/.env. Tambahkan HF_TOKEN terlebih dahulu.")

    df = pd.read_csv(DATA_COMBINED_PATH, low_memory=False)
    text_col = "text_raw" if "text_raw" in df.columns else ("content" if "content" in df.columns else "text")

    if only_conflict:
        mask_target = pd.Series(False, index=df.index)
        if "is_conflict" in df.columns:
            mask_target = mask_target | (df["is_conflict"].fillna(False).astype(bool))
        if "confidence" in df.columns:
            mask_target = mask_target | (pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0) < 0.75)
        target_indices = df[mask_target].index.tolist()
    else:
        target_indices = df.index.tolist()

    if limit is not None and limit > 0:
        target_indices = target_indices[:limit]

    if not target_indices:
        logger.info("Tidak ada baris yang memenuhi kriteria pelabelan ulang.")
        return {"processed": 0, "updated": 0, "spam_removed": 0}

    texts = [str(df.at[idx, text_col] or "") for idx in target_indices]
    active_model = labeler.gemini_model_id if labeler.provider == "gemini" else labeler.model_id
    logger.info("Memulai pelabelan LLM untuk %d baris (dry_run=%s, provider=%s, model=%s)...", len(texts), dry_run, labeler.provider, active_model)

    before_sent = df.loc[target_indices, "sentiment"].astype(str).str.lower().value_counts().to_dict() if "sentiment" in df.columns else {}
    predictions = labeler.label_texts(texts, use_cache=True)

    spam_indices = []
    updated = 0
    for idx, pred in zip(target_indices, predictions):
        if pred.get("is_spam"):
            spam_indices.append(idx)
            continue
        if not dry_run:
            df.at[idx, "sentiment"] = str(pred["sentiment"]).capitalize()
            df.at[idx, "aspect"] = str(pred["aspect"])
            df.at[idx, "confidence"] = float(pred["confidence"])
            df.at[idx, "labeling_method"] = f"llm_hf"
            if "is_conflict" in df.columns:
                df.at[idx, "is_conflict"] = False
        updated += 1

    after_sent: dict[str, int] = {}
    for pred in predictions:
        if not pred.get("is_spam"):
            s = str(pred["sentiment"]).lower()
            after_sent[s] = after_sent.get(s, 0) + 1

    logger.info("Distribusi Sebelum: %s", before_sent)
    logger.info("Distribusi Sesudah (non-spam): %s | Spam terdeteksi: %d", after_sent, len(spam_indices))

    if dry_run:
        print("\n=== PREVIEW DRY-RUN (10 SAMPEL PERTAMA) ===")
        for idx, pred in list(zip(target_indices, predictions))[:10]:
            snippet = str(df.at[idx, text_col])[:90].replace("\n", " ")
            print(f"- [{pred['sentiment'].upper()} | {pred['aspect']} | spam={pred['is_spam']} | {pred['confidence']}] {snippet}")
        return {
            "dry_run": True,
            "processed": len(target_indices),
            "updated": updated,
            "spam_removed": len(spam_indices),
            "before_sentiment": before_sent,
            "after_sentiment": after_sent,
        }

    # Backup sebelum menyimpan perubahan
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = DATA_COMBINED_PATH.parent / f"combined_labeled.backup_{ts}.csv"
    shutil.copy2(DATA_COMBINED_PATH, backup_path)
    logger.info("Backup dataset dibuat di: %s", backup_path)

    # Pindahkan spam ke excluded_spam.csv
    if spam_indices:
        df_spam = df.loc[spam_indices]
        EXCLUDED_SPAM_PATH.parent.mkdir(parents=True, exist_ok=True)
        if EXCLUDED_SPAM_PATH.exists():
            df_spam.to_csv(EXCLUDED_SPAM_PATH, mode="a", header=False, index=False)
        else:
            df_spam.to_csv(EXCLUDED_SPAM_PATH, index=False)
        df = df.drop(index=spam_indices).reset_index(drop=True)
        logger.info("%d baris spam dipindahkan ke %s", len(spam_indices), EXCLUDED_SPAM_PATH)

    df.to_csv(DATA_COMBINED_PATH, index=False)
    logger.info("Dataset master tersimpan (%d baris) di %s", len(df), DATA_COMBINED_PATH)

    if reprocess_weekly and WEEKLY_PRED_PATH.exists():
        relabel_weekly_parquet(labeler=labeler)

    return {
        "dry_run": False,
        "processed": len(target_indices),
        "updated": updated,
        "spam_removed": len(spam_indices),
        "backup_path": str(backup_path),
        "before_sentiment": before_sent,
        "after_sentiment": after_sent,
    }


def relabel_weekly_parquet(labeler: HFLLMLabeler | None = None, limit: int | None = None) -> dict:
    """Melabeli ulang weekly_predictions.parquet dengan LLM dan memperbarui weekly_issues.csv."""
    if not WEEKLY_PRED_PATH.exists():
        return {"processed": 0, "updated": 0, "spam_removed": 0}

    if labeler is None:
        labeler = HFLLMLabeler.from_config(load_config())

    df_w = pd.read_parquet(WEEKLY_PRED_PATH)
    if df_w.empty:
        return {"processed": 0, "updated": 0, "spam_removed": 0}

    indices = df_w.index.tolist()
    if limit and limit > 0:
        indices = indices[:limit]

    texts = [str(df_w.at[i, "teks_asli_anonim"] or "") for i in indices]
    preds = labeler.label_texts(texts, use_cache=True)

    spam_idx = []
    for idx, p in zip(indices, preds):
        if p.get("is_spam"):
            spam_idx.append(idx)
        else:
            df_w.at[idx, "sentimen"] = str(p["sentiment"]).lower()
            df_w.at[idx, "aspek"] = str(p["aspect"])
            df_w.at[idx, "probabilitas"] = float(p["confidence"])

    if spam_idx:
        df_w = df_w.drop(index=spam_idx).reset_index(drop=True)

    df_w.to_parquet(WEEKLY_PRED_PATH, index=False)

    try:
        from src.features.issues import WeeklyIssueDetector
        detector = WeeklyIssueDetector()
        detector.detect_and_save_all(df_w)
    except Exception as exc:
        logger.warning("Gagal memperbarui weekly_issues.csv: %s", exc)

    return {"processed": len(indices), "updated": len(indices) - len(spam_idx), "spam_removed": len(spam_idx)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pelabelan ulang dataset Undiksha menggunakan LLM Hugging Face.")
    parser.add_argument("--dry-run", action="store_true", help="Uji tanpa menimpa combined_labeled.csv.")
    parser.add_argument("--only-conflict", action="store_true", help="Hanya labeli ulang baris konflik / confidence rendah.")
    parser.add_argument("--all", action="store_true", help="Labeli ulang seluruh baris dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Batas jumlah baris yang diproses.")
    parser.add_argument("--batch-size", type=int, default=None, help="Ukuran batch per panggilan LLM.")
    parser.add_argument("--reprocess-weekly", action="store_true", help="Perbarui juga weekly_predictions.parquet dan weekly_issues.csv.")
    args = parser.parse_args()

    res = relabel_master_dataset(
        dry_run=args.dry_run,
        only_conflict=args.only_conflict,
        limit=args.limit,
        batch_size=args.batch_size,
        reprocess_weekly=args.reprocess_weekly,
    )
    print("\nRingkasan Pelabelan LLM:", res)


if __name__ == "__main__":
    main()

