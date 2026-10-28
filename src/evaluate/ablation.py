"""
src/evaluate/ablation.py
========================
Ablation Study untuk Hybrid Ensemble (PRD §6 F10).

Mengukur kontribusi marginal masing-masing komponen:
1. Lexicon Only (Komponen A)
2. ML Klasik Only (Komponen B: SVM)
3. IndoBERT Only (Komponen C)
4. Kombinasi 2 Model:
   - Lexicon + ML Klasik (A + B)
   - Lexicon + IndoBERT (A + C)
   - ML Klasik + IndoBERT (B + C)
5. Full Hybrid Ensemble (A + B + C)

Menghasilkan tabel ringkasan ablation study di reports/tables/ablation_study.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluate.metrics import calculate_comprehensive_metrics
from src.models.ensemble import (
    StackingEnsemble,
    WeightedSoftVotingEnsemble,
    run_stratified_kfold_evaluation,
)
from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)


def evaluate_two_model_soft_voting(
    prob_1: np.ndarray,
    prob_2: np.ndarray,
    y_true: np.ndarray,
    step: float = 0.05,
) -> tuple[np.ndarray, dict[str, float], float]:
    """Mengoptimasi bobot untuk kombinasi dua komponen dan mengembalikan metrik evaluasi."""
    best_f1 = -1.0
    best_w = 0.5
    for w in np.arange(0.0, 1.0 + step / 2, step):
        p_comb = w * prob_1 + (1.0 - w) * prob_2
        preds = np.argmax(p_comb, axis=1)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w = w

    p_final = best_w * prob_1 + (1.0 - best_w) * prob_2
    preds_final = np.argmax(p_final, axis=1)
    acc = accuracy_score(y_true, preds_final)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, preds_final, average="macro", zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p_m), 4),
        "recall_macro": round(float(r_m), 4),
        "f1_macro": round(float(f1_m), 4),
    }
    return p_final, metrics, round(float(best_w), 2)


def run_ablation_study(
    oof_bundle: dict[str, Any],
    tables_dir: str | Path = "reports/tables",
) -> pd.DataFrame:
    """Menjalankan ablation study lengkap pada OOF probabilitas seluruh konfigurasi komponen.

    Args:
        oof_bundle: Dict OOF probabilitas dari run_stratified_kfold_evaluation.
        tables_dir: Direktori tujuan ekspor tabel laporan.

    Returns:
        DataFrame ringkasan hasil ablation study.
    """
    oof_a = oof_bundle["oof_a"]
    oof_b = oof_bundle["oof_b"]
    oof_c = oof_bundle["oof_c"]
    y_true = oof_bundle["y_true"]
    oof_soft_vote = oof_bundle["oof_soft_vote"]
    oof_stacking = oof_bundle["oof_stacking"]

    # 1. Single Models
    m_a = calculate_comprehensive_metrics(y_true, np.argmax(oof_a, axis=1))
    m_b = calculate_comprehensive_metrics(y_true, np.argmax(oof_b, axis=1))
    m_c = calculate_comprehensive_metrics(y_true, np.argmax(oof_c, axis=1))

    # 2. Pairwise Combinations
    _, m_ab, w_ab = evaluate_two_model_soft_voting(oof_a, oof_b, y_true)
    _, m_ac, w_ac = evaluate_two_model_soft_voting(oof_a, oof_c, y_true)
    _, m_bc, w_bc = evaluate_two_model_soft_voting(oof_b, oof_c, y_true)

    # 3. Full Ensemble
    m_soft = calculate_comprehensive_metrics(y_true, np.argmax(oof_soft_vote, axis=1))
    m_stack = calculate_comprehensive_metrics(y_true, np.argmax(oof_stacking, axis=1))

    baseline_f1 = m_b["f1_macro"]  # Best single component (SVM)

    rows = [
        {
            "Konfigurasi Model": "1. Lexicon InSet Saja (A)",
            "Komponen": "A",
            "Accuracy": f"{m_a['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_a['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_a['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_a['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": f"{(m_a['f1_macro'] - baseline_f1)*100:+.2f}%",
        },
        {
            "Konfigurasi Model": "2. ML Klasik SVM Saja (B)",
            "Komponen": "B",
            "Accuracy": f"{m_b['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_b['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_b['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_b['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": "0.00% (Baseline)",
        },
        {
            "Konfigurasi Model": "3. IndoBERT Saja (C)",
            "Komponen": "C",
            "Accuracy": f"{m_c['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_c['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_c['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_c['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": f"{(m_c['f1_macro'] - baseline_f1)*100:+.2f}%",
        },
        {
            "Konfigurasi Model": f"4. Lexicon + SVM (A + B, w_A={w_ab:.2f})",
            "Komponen": "A + B",
            "Accuracy": f"{m_ab['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_ab['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_ab['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_ab['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": f"{(m_ab['f1_macro'] - baseline_f1)*100:+.2f}%",
        },
        {
            "Konfigurasi Model": f"5. Lexicon + IndoBERT (A + C, w_A={w_ac:.2f})",
            "Komponen": "A + C",
            "Accuracy": f"{m_ac['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_ac['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_ac['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_ac['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": f"{(m_ac['f1_macro'] - baseline_f1)*100:+.2f}%",
        },
        {
            "Konfigurasi Model": f"6. SVM + IndoBERT (B + C, w_B={w_bc:.2f})",
            "Komponen": "B + C",
            "Accuracy": f"{m_bc['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_bc['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_bc['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_bc['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": f"{(m_bc['f1_macro'] - baseline_f1)*100:+.2f}%",
        },
        {
            "Konfigurasi Model": "7. Full Stacking Ensemble (A + B + C)",
            "Komponen": "A + B + C (Stacking)",
            "Accuracy": f"{m_stack['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_stack['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_stack['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_stack['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": f"{(m_stack['f1_macro'] - baseline_f1)*100:+.2f}%",
        },
        {
            "Konfigurasi Model": "8. Full Weighted Soft Voting (A + B + C) [TERBAIK]",
            "Komponen": "A + B + C (Voting)",
            "Accuracy": f"{m_soft['accuracy']*100:.2f}%",
            "Precision-Macro": f"{m_soft['precision_macro']*100:.2f}%",
            "Recall-Macro": f"{m_soft['recall_macro']*100:.2f}%",
            "F1-Macro": f"{m_soft['f1_macro']*100:.2f}%",
            "Delta F1 vs Best Single": f"{(m_soft['f1_macro'] - baseline_f1)*100:+.2f}%",
        },
    ]

    df_ablation = pd.DataFrame(rows)

    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ablation_study.csv"
    df_ablation.to_csv(out_file, index=False, encoding="utf-8-sig")
    logger.info("Tabel Ablation Study disimpan ke: %s", out_file)

    return df_ablation


def main() -> None:
    """CLI untuk menjalankan Ablation Study."""
    parser = argparse.ArgumentParser(description="Jalankan Ablation Study pada model sentimen.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    df = pd.read_csv(args.data)
    _, oof_bundle = run_stratified_kfold_evaluation(df, cfg, n_splits=5, classical_model_name="svm")

    df_ablation = run_ablation_study(oof_bundle, tables_dir=cfg["paths"]["tables_dir"])

    print("\n" + "=" * 90)
    print("  HASIL ABLATION STUDY (ANALISIS KONTRIBUSI KOMPONEN HYBRID ENSEMBLE)")
    print("=" * 90)
    print(df_ablation[["Konfigurasi Model", "Accuracy", "F1-Macro", "Delta F1 vs Best Single"]].to_string(index=False))
    print("=" * 90)
    print("\nFile hasil disimpan ke: reports/tables/ablation_study.csv\n")


if __name__ == "__main__":
    main()
