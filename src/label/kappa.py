"""
src/label/kappa.py
==================
Perhitungan inter-annotator agreement (Cohen's Kappa) dan rekonsiliasi label.

Sesuai PRD §6 (F3) dan standar Landis & Koch (1977):
    - Kappa < 0.20 : Poor
    - 0.21 - 0.40  : Fair
    - 0.41 - 0.60  : Moderate
    - 0.61 - 0.80  : Substantial (Target Proyek: >= 0.60)
    - 0.81 - 1.00  : Almost Perfect
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

logger = logging.getLogger(__name__)

STANDARD_SENTIMENT_LABELS = ["positif", "netral", "negatif"]


def interpret_kappa(kappa: float) -> str:
    """Menginterpretasikan nilai Cohen's Kappa menurut Landis & Koch (1977).

    Args:
        kappa: Nilai koefisien Kappa.

    Returns:
        String kategori interpretasi.
    """
    if kappa < 0.0:
        return "Poor (Kurang dari kebetulan)"
    elif kappa <= 0.20:
        return "Slight (Sangat Rendah)"
    elif kappa <= 0.40:
        return "Fair (Cukup)"
    elif kappa <= 0.60:
        return "Moderate (Sedang)"
    elif kappa <= 0.80:
        return "Substantial (Tinggi - Target Tercapai)"
    else:
        return "Almost Perfect (Hampir Sempurna)"


def calculate_agreement_metrics(
    y_true_a: list[str] | pd.Series,
    y_true_b: list[str] | pd.Series,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Menghitung metrik kesepakatan Cohen's Kappa, persentase kesepakatan, dan confusion matrix.

    Args:
        y_true_a: Label dari Annotator A.
        y_true_b: Label dari Annotator B.
        labels: Urutan label yang dievaluasi (default: ['positif', 'netral', 'negatif']).

    Returns:
        Dict berisi skor kappa, observed_agreement, confusion_matrix, dan interpretasi.
    """
    if labels is None:
        labels = STANDARD_SENTIMENT_LABELS

    s_a = pd.Series(y_true_a).astype(str).str.strip().str.lower()
    s_b = pd.Series(y_true_b).astype(str).str.strip().str.lower()

    if len(s_a) != len(s_b):
        raise ValueError(f"Panjang kedua label tidak sama: {len(s_a)} vs {len(s_b)}")

    # Cohen's Kappa
    kappa = float(cohen_kappa_score(s_a, s_b, labels=labels))

    # Observed agreement (persentase kecocokan langsung)
    agreements = (s_a == s_b).sum()
    total = len(s_a)
    observed_agreement = float(agreements / total) if total > 0 else 0.0

    # Confusion Matrix
    cm = confusion_matrix(s_a, s_b, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"A_{lb}" for lb in labels], columns=[f"B_{lb}" for lb in labels])

    return {
        "cohen_kappa": round(kappa, 4),
        "observed_agreement": round(observed_agreement, 4),
        "observed_agreement_pct": f"{observed_agreement * 100:.2f}%",
        "total_evaluated": total,
        "agreements_count": int(agreements),
        "disagreements_count": int(total - agreements),
        "interpretation": interpret_kappa(kappa),
        "confusion_matrix": cm_df,
    }


def reconcile_annotations(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_col: str = "sentiment",
    id_col: str = "id_anon",
) -> pd.DataFrame:
    """Menggabungkan dua set anotasi dan menandai baris yang memerlukan rekonsiliasi.

    Args:
        df_a: DataFrame Anotasi dari Annotator A.
        df_b: DataFrame Anotasi dari Annotator B.
        label_col: Kolom label sentimen.
        id_col: Kolom identifier baris.

    Returns:
        DataFrame gabungan dengan kolom consensus `sentiment_final` dan flag `is_conflict`.
    """
    merged = pd.merge(
        df_a,
        df_b[[id_col, label_col] + [c for c in ["aspect", "notes"] if c in df_b.columns]],
        on=id_col,
        suffixes=("_a", "_b"),
    )

    merged[f"{label_col}_a"] = merged[f"{label_col}_a"].astype(str).str.strip().str.lower()
    merged[f"{label_col}_b"] = merged[f"{label_col}_b"].astype(str).str.strip().str.lower()

    is_conflict = merged[f"{label_col}_a"] != merged[f"{label_col}_b"]
    merged["is_conflict"] = is_conflict

    # Default konsensus: jika sama, gunakan nilai tersebut; jika beda, tandai 'KONFLIK'
    merged["sentiment_final"] = np.where(
        ~is_conflict,
        merged[f"{label_col}_a"],
        "KONFLIK",
    )

    return merged


def main() -> None:
    """Entry point CLI untuk evaluasi Cohen's Kappa antar dua file anotasi."""
    parser = argparse.ArgumentParser(description="Evaluasi kesepakatan Cohen's Kappa antara dua annotator.")
    parser.add_argument("--file-a", required=True, help="Path CSV dari Annotator A")
    parser.add_argument("--file-b", required=True, help="Path CSV dari Annotator B")
    parser.add_argument("--output-reconciled", default=None, help="Path output untuk data hasil rekonsiliasi")
    args = parser.parse_args()

    df_a = pd.read_csv(args.file_a)
    df_b = pd.read_csv(args.file_b)

    metrics = calculate_agreement_metrics(df_a["sentiment"], df_b["sentiment"])
    print("\n" + "=" * 50)
    print("  HASIL EVALUASI INTER-ANNOTATOR AGREEMENT")
    print("=" * 50)
    print(f"Total baris        : {metrics['total_evaluated']}")
    print(f"Kesepakatan Cocok  : {metrics['agreements_count']}")
    print(f"Perbedaan/Konflik  : {metrics['disagreements_count']}")
    print(f"Observed Agreement : {metrics['observed_agreement_pct']}")
    print(f"Cohen's Kappa      : {metrics['cohen_kappa']}")
    print(f"Interpretasi       : {metrics['interpretation']}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 50 + "\n")

    if args.output_reconciled:
        reconciled_df = reconcile_annotations(df_a, df_b)
        out_path = Path(args.output_reconciled)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reconciled_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"File rekonsiliasi disimpan ke: {out_path}")


if __name__ == "__main__":
    main()
