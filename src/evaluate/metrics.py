"""
src/evaluate/metrics.py
=======================
Modul evaluasi metrik performa klasifikasi sentimen (PRD §6 F9, F16).

Menyediakan:
1. Metrik makro, mikro, dan weighted: Accuracy, Precision, Recall, F1-score.
2. Rincian performa per kelas (negatif, netral, positif).
3. Perhitungan Confusion Matrix berlabel dan persentase normalisasi.
4. Utilitas ekspor laporan ke CSV dan Markdown untuk bab hasil laporan magang.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["negatif", "netral", "positif"]
LABEL_MAP = {"negatif": 0, "netral": 1, "positif": 2}
INV_LABEL_MAP = {0: "negatif", 1: "netral", 2: "positif"}


def calculate_comprehensive_metrics(
    y_true: np.ndarray | list[int] | list[str],
    y_pred: np.ndarray | list[int] | list[str],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Menghitung metrik performa klasifikasi lengkap (overall & per kelas).

    Args:
        y_true: Ground truth target (integer atau string).
        y_pred: Predicted target (integer atau string).
        class_names: Daftar nama kelas (default: ['negatif', 'netral', 'positif']).

    Returns:
        Dict berisi metrik ringkasan, laporan per kelas, dan confusion matrix.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Konversi string ke integer jika perlu
    if len(y_true) > 0 and isinstance(y_true[0], str):
        y_t = np.array([LABEL_MAP.get(str(s).lower(), 1) for s in y_true])
    else:
        y_t = np.array(y_true, dtype=int)

    if len(y_pred) > 0 and isinstance(y_pred[0], str):
        y_p = np.array([LABEL_MAP.get(str(s).lower(), 1) for s in y_pred])
    else:
        y_p = np.array(y_pred, dtype=int)

    if len(y_t) != len(y_p):
        raise ValueError(f"Ukuran target berbeda: y_true={len(y_t)}, y_pred={len(y_p)}")

    # Overall Metrics
    acc = accuracy_score(y_t, y_p)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_t, y_p, average="micro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_t, y_p, average="weighted", zero_division=0
    )

    # Per-Class Metrics
    p_classes, r_classes, f1_classes, support_classes = precision_recall_fscore_support(
        y_t, y_p, labels=[0, 1, 2], average=None, zero_division=0
    )

    per_class_dict = {}
    for i, cls in enumerate(class_names):
        per_class_dict[cls] = {
            "precision": round(float(p_classes[i]), 4),
            "recall": round(float(r_classes[i]), 4),
            "f1_score": round(float(f1_classes[i]), 4),
            "support": int(support_classes[i]),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names],
    )

    # Normalized Confusion Matrix (persentase recall per baris)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])
    cm_norm_df = pd.DataFrame(
        np.round(cm_norm, 4),
        index=[f"Actual_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names],
    )

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_micro": round(float(p_micro), 4),
        "recall_micro": round(float(r_micro), 4),
        "f1_micro": round(float(f1_micro), 4),
        "f1_weighted": round(float(f1_weight), 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm_df,
        "confusion_matrix_norm": cm_norm_df,
        "total_samples": len(y_t),
    }


def format_metrics_table(metrics_dict: dict[str, Any], model_name: str = "Model") -> str:
    """Memformat metrik ke bentuk tabel teks rapi siap dicetak atau disimpan ke log."""
    m = metrics_dict
    s = [
        f"Laporan Metrik Evaluasi: {model_name}",
        "=" * 55,
        f"Total Sampel : {m['total_samples']}",
        f"Accuracy      : {m['accuracy'] * 100:.2f}%",
        f"F1-Macro      : {m['f1_macro'] * 100:.2f}%",
        f"Precision-M   : {m['precision_macro'] * 100:.2f}%",
        f"Recall-M      : {m['recall_macro'] * 100:.2f}%",
        f"F1-Weighted   : {m['f1_weighted'] * 100:.2f}%",
        "-" * 55,
        "Rincian Per Kelas:",
    ]
    for cls, val in m["per_class"].items():
        s.append(
            f"  • {cls.upper():<8} -> Prec: {val['precision']*100:.1f}% | "
            f"Rec: {val['recall']*100:.1f}% | F1: {val['f1_score']*100:.1f}% (Support: {val['support']})"
        )
    s.append("=" * 55)
    return "\n".join(s)


def export_confusion_matrix_csv(
    cm_df: pd.DataFrame,
    filepath: str | Path,
) -> Path:
    """Mengekspor matriks kontingensi / confusion matrix ke CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(path, index=True, encoding="utf-8-sig")
    logger.info("Confusion matrix disimpan: %s", path)
    return path
