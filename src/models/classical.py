"""
src/models/classical.py
=======================
Komponen B: Model Machine Learning Klasik + TF-IDF (PRD §6 F5).

Algoritma:
1. Support Vector Machine (SVM)
2. Multinomial Naive Bayes (NB)
3. Random Forest (RF)

Kemampuan Utama:
- Hyperparameter dari configs/config.yaml.
- Prediksi probabilitas (predict_proba) untuk soft voting.
- Out-of-fold (OOF) cross-validation predictions untuk stacking tanpa leakage.
- Evaluasi metrik lengkap: Accuracy, Precision-macro, Recall-macro, F1-macro.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.tfidf_vectorizer import (
    CLASS_MAPPING,
    INV_CLASS_MAPPING,
    prepare_features,
)
from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)

CLASS_NAMES = ["negatif", "netral", "positif"]


def build_classical_model(model_name: str, config: dict[str, Any]) -> Any:
    """Membangun instance model ML klasik berdasarkan konfigurasi.

    Args:
        model_name: 'svm', 'naive_bayes', atau 'random_forest'.
        config: Dict konfigurasi.

    Returns:
        Instance estimator scikit-learn.
    """
    seed = config.get("seed", 42)
    cfg_models = config.get("classical_models", {})

    if model_name == "svm":
        p = cfg_models.get("svm", {})
        return SVC(
            kernel=p.get("kernel", "rbf"),
            C=p.get("C", 1.0),
            class_weight=p.get("class_weight", "balanced"),
            probability=p.get("probability", True),
            random_state=seed,
        )
    elif model_name == "naive_bayes":
        p = cfg_models.get("naive_bayes", {})
        return MultinomialNB(alpha=p.get("alpha", 1.0))
    elif model_name == "random_forest":
        p = cfg_models.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", None),
            class_weight=p.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        p = cfg_models.get("xgboost", {})
        return XGBClassifier(
            n_estimators=p.get("n_estimators", 150),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.1),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            objective="multi:softprob",
            num_class=3,
            random_state=seed,
            n_jobs=-1,
            eval_metric=p.get("eval_metric", "mlogloss"),
        )
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}. Pilih: svm, naive_bayes, random_forest, xgboost")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Menghitung metrik performa klasifikasi sentimen secara komprehensif.

    Args:
        y_true: Ground truth target label (integer).
        y_pred: Predicted target label (integer).

    Returns:
        Dict metrik performa.
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = np.mean(y_true == y_pred)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        labels=[0, 1, 2],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "report": report_str,
    }


def generate_oof_predictions(
    model_name: str,
    X: Any,
    y: np.ndarray,
    config: dict[str, Any],
    n_splits: int = 5,
) -> tuple[np.ndarray, float]:
    """Menghasilkan Out-Of-Fold (OOF) prediction probabilities pada training data.

    PENTING UNTUK STACKING ENSEMBLE:
    Fitur meta-learner tidak boleh dihasilkan dari model yang dilatih pada data yang sama (kebocoran).
    OOF memastikan prediksi pada fold k dihasilkan oleh model yang hanya melihat fold lainnya.

    Args:
        model_name: Nama model.
        X: Matriks fitur.
        y: Label numerik.
        config: Dict konfigurasi.
        n_splits: Jumlah fold cross-validation.

    Returns:
        Tuple (oof_probabilities [shape N, 3], mean_f1_score).
    """
    seed = config.get("seed", 42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probas = np.zeros((X.shape[0], len(CLASS_NAMES)))
    oof_preds = np.zeros(X.shape[0], dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        clf = build_classical_model(model_name, config)
        clf.fit(X_tr, y_tr)

        # Ambil probabilitas (jika model mendukung)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_va)
            oof_probas[val_idx] = probs
            oof_preds[val_idx] = np.argmax(probs, axis=1)
        else:
            preds = clf.predict(X_va)
            oof_preds[val_idx] = preds
            for idx, p in zip(val_idx, preds):
                oof_probas[idx, p] = 1.0

    overall_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    logger.info("Model %s OOF 5-Fold F1-Macro: %.4f", model_name, overall_f1)
    return oof_probas, float(overall_f1)


def train_and_evaluate_all_classical(
    data_bundle: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Melatih dan mengevaluasi seluruh model ML klasik (SVM, NB, RF) pada data validasi dan test.

    Args:
        data_bundle: Bundle fitur dari tfidf_vectorizer.prepare_features().
        config: Dict konfigurasi.

    Returns:
        Dict hasil evaluasi untuk setiap model.
    """
    X_train = data_bundle["X_train"]
    y_train = data_bundle["y_train"]
    X_val   = data_bundle["X_val"]
    y_val   = data_bundle["y_val"]
    X_test  = data_bundle["X_test"]
    y_test  = data_bundle["y_test"]

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    for model_name in ["svm", "naive_bayes", "random_forest"]:
        logger.info("=== Melatih Model: %s ===", model_name.upper())
        clf = build_classical_model(model_name, config)
        clf.fit(X_train, y_train)

        # Evaluasi pada Validation Set
        val_pred = clf.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_pred)

        # Evaluasi pada Test Set (unbiased evaluation)
        test_pred = clf.predict(X_test)
        test_metrics = evaluate_predictions(y_test, test_pred)

        # Simpan model
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(clf, model_path)

        results[model_name] = {
            "model_path": str(model_path),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "fitted_model": clf,
        }

    return results


def main() -> None:
    """CLI untuk melatih dan mengevaluasi ketiga model ML klasik."""
    parser = argparse.ArgumentParser(description="Latih dan evaluasi model ML klasik.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    df = pd.read_csv(args.data)
    data_bundle = prepare_features(df, cfg)

    results = train_and_evaluate_all_classical(data_bundle, cfg)

    print("\n" + "=" * 65)
    print("  HASIL EVALUASI KOMPONEN B (ML KLASIK + TF-IDF)")
    print("=" * 65)
    for name, res in results.items():
        tm = res["test_metrics"]
        print(f"\nModel: {name.upper()}")
        print(f"  Test Accuracy : {tm['accuracy']:.4f}")
        print(f"  Test F1-Macro : {tm['f1_macro']:.4f}")
        print(f"  Test Precision: {tm['precision_macro']:.4f}")
        print(f"  Test Recall   : {tm['recall_macro']:.4f}")
        print("\nLaporan Test Set:")
        print(tm["report"])
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
