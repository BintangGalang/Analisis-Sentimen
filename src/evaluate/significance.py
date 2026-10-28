"""
src/evaluate/significance.py
============================
Uji signifikansi statistik performa model sentimen (PRD §6 F9, PRD §11).

Metode:
1. McNemar's Test (dengan Edwards' continuity correction dan exact binomial test):
   - Standar emas dalam literatur NLP / ML untuk membandingkan dua model pada dataset uji yang berpasangan (Dietterich, 1998).
2. Paired Student's t-test pada skor K-Fold CV.
3. Wilcoxon Signed-Rank Test untuk data non-parametrik.

Kriteria Keputusan:
- Nilai signifikansi alpha = 0.05.
- Jika p-value < 0.05, hipotesis nol (H0) ditolak -> Perbedaan performa model signifikan secara statistik.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def mcnemar_test(
    y_true: np.ndarray | list[int],
    y_pred_a: np.ndarray | list[int],
    y_pred_b: np.ndarray | list[int],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    exact: bool | None = None,
) -> dict[str, Any]:
    """Menjalankan McNemar's Test untuk membandingkan dua model klasifikasi pada dataset berpasangan.

    Tabel Kontingensi 2x2:
                    Model B Benar     Model B Salah
    Model A Benar       n_11 (keduanya benar)  n_10 (A benar, B salah)
    Model A Salah       n_01 (A salah, B benar)  n_00 (keduanya salah)

    Args:
        y_true: Target ground truth.
        y_pred_a: Prediksi Model A.
        y_pred_b: Prediksi Model B.
        model_a_name: Nama Model A.
        model_b_name: Nama Model B.
        exact: Jika True gunakan exact binomial test; jika None, otomatis jika (n01+n10) < 25.

    Returns:
        Dict hasil uji McNemar (chi2_stat, p_value, is_significant, contingency_table).
    """
    yt = np.array(y_true)
    ya = np.array(y_pred_a)
    yb = np.array(y_pred_b)

    correct_a = (ya == yt)
    correct_b = (yb == yt)

    n_11 = int(np.sum(correct_a & correct_b))
    n_10 = int(np.sum(correct_a & (~correct_b)))  # A benar, B salah
    n_01 = int(np.sum((~correct_a) & correct_b))  # A salah, B benar
    n_00 = int(np.sum((~correct_a) & (~correct_b)))

    discordant = n_01 + n_10

    # Tentukan exact test vs chi2
    use_exact = exact if exact is not None else (discordant < 25)

    if use_exact:
        # Exact Binomial Test pada pasangan diskordan
        k = min(n_01, n_10)
        # Dua arah (two-tailed)
        p_val = float(2.0 * stats.binom.cdf(k, discordant, 0.5))
        p_val = min(1.0, p_val)
        stat_val = float(k)
        method_name = "Exact Binomial McNemar Test"
    else:
        # Chi-squared dengan koreksi kontinuitas Edwards: (|b - c| - 1)^2 / (b + c)
        if discordant == 0:
            stat_val = 0.0
            p_val = 1.0
        else:
            chi2 = (abs(n_01 - n_10) - 1.0) ** 2 / discordant
            stat_val = float(chi2)
            p_val = float(stats.chi2.sf(chi2, df=1))
        method_name = "Edwards' Continuity-Corrected McNemar Test (Chi-Squared)"

    is_significant = p_val < 0.05

    contingency_matrix = pd.DataFrame(
        [
            [n_11, n_10],
            [n_01, n_00],
        ],
        index=[f"{model_a_name} Benar", f"{model_a_name} Salah"],
        columns=[f"{model_b_name} Benar", f"{model_b_name} Salah"],
    )

    # Menentukan model mana yang unggul jika signifikan
    if is_significant:
        superior_model = model_b_name if n_01 > n_10 else model_a_name
        conclusion = (
            f"Perbedaan SIGNIFIKAN secara statistik (p = {p_val:.4e} < 0.05). "
            f"{superior_model} terbukti lebih unggul."
        )
    else:
        conclusion = f"Perbedaan TIDAK signifikan secara statistik (p = {p_val:.4f} >= 0.05)."

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "method": method_name,
        "statistic": round(stat_val, 4),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}",
        "is_significant": is_significant,
        "n_a_better": n_10,
        "n_b_better": n_01,
        "contingency_table": contingency_matrix,
        "conclusion": conclusion,
    }


def paired_cv_t_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict[str, Any]:
    """Menjalankan Paired Student's t-test pada metrik skor K-Fold antar dua model."""
    a = np.array(scores_a)
    b = np.array(scores_b)

    diff = b - a
    res = stats.ttest_rel(b, a)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)
    is_significant = p_val < 0.05

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "mean_diff": round(float(np.mean(diff)), 4),
        "t_statistic": round(t_stat, 4),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}",
        "is_significant": is_significant,
    }


def run_pairwise_significance_matrix(
    predictions_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Menjalankan uji McNemar berpasangan antara semua model dan menyusun tabel matriks p-value.

    Args:
        predictions_dict: Dict {nama_model: array_prediksi_label}.
        y_true: Array ground truth target label.

    Returns:
        DataFrame matriks p-value dan penanda signifikansi (* jika p < 0.05).
    """
    model_names = list(predictions_dict.keys())
    n = len(model_names)
    matrix = pd.DataFrame(index=model_names, columns=model_names)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix.iloc[i, j] = "-"
            else:
                m1 = model_names[i]
                m2 = model_names[j]
                res = mcnemar_test(
                    y_true,
                    predictions_dict[m1],
                    predictions_dict[m2],
                    model_a_name=m1,
                    model_b_name=m2,
                )
                sig_star = " *" if res["is_significant"] else ""
                matrix.iloc[i, j] = f"{res['p_value_formatted']}{sig_star}"

    return matrix
