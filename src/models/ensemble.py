"""
src/models/ensemble.py
======================
Implementasi Hybrid Ensemble: Stacking & Weighted Soft Voting (PRD §6 F7, F8, F9).

Komponen yang digabungkan:
- Komponen A: Lexicon-based (InSet)
- Komponen B: Machine Learning Klasik + TF-IDF (SVM / Random Forest)
- Komponen C: IndoBERT Pretrained Transformer (indobenchmark/indobert-base-p1)

Metode Ensemble:
1. Stacking Ensemble:
   - Menggunakan Out-Of-Fold (OOF) prediction probabilities (shape [N, 9])
   - Meta-learner: Logistic Regression (multinomial, balanced class weight)
2. Weighted Soft Voting:
   - Kombinasi linear terbobot dari probabilitas ketiga komponen:
     P_ensemble = w_A * P_A + w_B * P_B + w_C * P_C (w_A + w_B + w_C = 1, w_i >= 0)
   - Bobot dioptimasi secara otomatis dari data validasi / OOF untuk memaksimalkan F1-macro.

Kemampuan:
- Evaluasi Stratified K-Fold (default: 5-Fold).
- Perbandingan komprehensif: Komponen A, Komponen B, Komponen C, Stacking, dan Soft Voting.
- Ekspor tabel hasil evaluasi ke reports/tables/model_comparison.csv.
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.tfidf_vectorizer import TextFeatureExtractor
from src.models.classical import build_classical_model
from src.models.lexicon import InSetLexiconModel
from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)

CLASS_NAMES = ["negatif", "netral", "positif"]
LABEL_MAPPING = {"negatif": 0, "netral": 1, "positif": 2}


class StackingEnsemble:
    """Stacking Ensemble Classifier dengan Meta-Learner Logistic Regression."""

    def __init__(
        self,
        meta_learner: Any | None = None,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """Inisialisasi Stacking Ensemble.

        Args:
            meta_learner: Estimator scikit-learn untuk meta-learner (default: LogisticRegression).
            C: Parameter regularisasi Logistic Regression.
            random_state: Random seed.
        """
        self.random_state = random_state
        if meta_learner is not None:
            self.meta_learner = meta_learner
        else:
            self.meta_learner = LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
        self.is_fitted = False

    def _build_meta_features(
        self,
        probas_a: np.ndarray,
        probas_b: np.ndarray,
        probas_c: np.ndarray,
    ) -> np.ndarray:
        """Menggabungkan probabilitas ketiga komponen menjadi matriks fitur meta (shape [N, 9])."""
        return np.hstack([probas_a, probas_b, probas_c])

    def fit(
        self,
        probas_a: np.ndarray,
        probas_b: np.ndarray,
        probas_c: np.ndarray,
        y: np.ndarray | list[int],
    ) -> StackingEnsemble:
        """Melatih meta-learner pada probabilitas out-of-fold ketiga komponen.

        Args:
            probas_a: Probabilitas OOF Komponen A (Lexicon).
            probas_b: Probabilitas OOF Komponen B (ML Klasik).
            probas_c: Probabilitas OOF Komponen C (IndoBERT).
            y: Target label numerik.

        Returns:
            Self.
        """
        X_meta = self._build_meta_features(probas_a, probas_b, probas_c)
        self.meta_learner.fit(X_meta, y)
        self.is_fitted = True
        logger.info("Stacking Meta-Learner (Logistic Regression) berhasil dilatih pada shape %s.", X_meta.shape)
        return self

    def predict_proba(
        self,
        probas_a: np.ndarray,
        probas_b: np.ndarray,
        probas_c: np.ndarray,
    ) -> np.ndarray:
        """Menghasilkan probabilitas prediksi ensemble."""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble belum di-fit!")
        X_meta = self._build_meta_features(probas_a, probas_b, probas_c)
        return self.meta_learner.predict_proba(X_meta)

    def predict(
        self,
        probas_a: np.ndarray,
        probas_b: np.ndarray,
        probas_c: np.ndarray,
    ) -> np.ndarray:
        """Menghasilkan prediksi kelas integer."""
        probs = self.predict_proba(probas_a, probas_b, probas_c)
        return np.argmax(probs, axis=1)

    def get_meta_coefficients(self) -> pd.DataFrame:
        """Mengambil koefisien bobot meta-learner per komponen untuk analisis kontribusi."""
        if not self.is_fitted or not hasattr(self.meta_learner, "coef_"):
            return pd.DataFrame()

        feature_names = [
            f"lexicon_{cls}" for cls in CLASS_NAMES
        ] + [
            f"classical_{cls}" for cls in CLASS_NAMES
        ] + [
            f"indobert_{cls}" for cls in CLASS_NAMES
        ]

        coefs = self.meta_learner.coef_
        df_coef = pd.DataFrame(coefs, index=[f"target_{c}" for c in CLASS_NAMES], columns=feature_names)
        return df_coef


class WeightedSoftVotingEnsemble:
    """Weighted Soft Voting Ensemble dengan optimasi bobot berbasis validasi."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Inisialisasi Soft Voting Ensemble.

        Args:
            weights: Dict bobot awal {'lexicon': w_a, 'classical': w_b, 'bert': w_c}.
        """
        self.weights = weights or {"lexicon": 0.3333, "classical": 0.3333, "bert": 0.3334}
        self.is_fitted = weights is not None

    def fit(
        self,
        probas_a: np.ndarray,
        probas_b: np.ndarray,
        probas_c: np.ndarray,
        y: np.ndarray | list[int],
        step: float = 0.02,
    ) -> WeightedSoftVotingEnsemble:
        """Menemukan bobot optimal (w_A, w_B, w_C) pada data validasi / OOF untuk memaksimalkan F1-macro.

        Args:
            probas_a: Probabilitas validasi Komponen A (Lexicon).
            probas_b: Probabilitas validasi Komponen B (ML Klasik).
            probas_c: Probabilitas validasi Komponen C (IndoBERT).
            y: Target label ground truth.
            step: Step size grid search simplex (default: 0.02).

        Returns:
            Self.
        """
        y_arr = np.array(y)
        best_f1 = -1.0
        best_weights = (0.33, 0.33, 0.34)

        # Grid search pada 2-simplex: wA + wB + wC = 1, w_i >= 0
        w_range = np.arange(0.0, 1.0 + step / 2, step)
        for w_a in w_range:
            for w_b in w_range:
                w_c = round(1.0 - w_a - w_b, 4)
                if w_c < 0.0:
                    continue

                p_combined = w_a * probas_a + w_b * probas_b + w_c * probas_c
                preds = np.argmax(p_combined, axis=1)
                _, _, f1, _ = precision_recall_fscore_support(
                    y_arr, preds, average="macro", zero_division=0
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = (round(w_a, 4), round(w_b, 4), round(w_c, 4))

        self.weights = {
            "lexicon": best_weights[0],
            "classical": best_weights[1],
            "bert": best_weights[2],
        }
        self.is_fitted = True
        logger.info(
            "Soft Voting Bobot Optimal Ditemukan: Lexicon=%.2f, ML=%.2f, IndoBERT=%.2f (OOF F1: %.4f)",
            best_weights[0],
            best_weights[1],
            best_weights[2],
            best_f1,
        )
        return self

    def predict_proba(
        self,
        probas_a: np.ndarray,
        probas_b: np.ndarray,
        probas_c: np.ndarray,
    ) -> np.ndarray:
        """Menghasilkan probabilitas berbobot ensemble."""
        w_a = self.weights.get("lexicon", 0.3333)
        w_b = self.weights.get("classical", 0.3333)
        w_c = self.weights.get("bert", 0.3334)
        return w_a * probas_a + w_b * probas_b + w_c * probas_c

    def predict(
        self,
        probas_a: np.ndarray,
        probas_b: np.ndarray,
        probas_c: np.ndarray,
    ) -> np.ndarray:
        """Menghasilkan prediksi kelas integer."""
        probs = self.predict_proba(probas_a, probas_b, probas_c)
        return np.argmax(probs, axis=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Menghitung metrik evaluasi standar: Accuracy, Precision-macro, Recall-macro, F1-macro."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1), 4),
    }


def extract_bert_embeddings(
    texts: list[str],
    model_name: str = "indobenchmark/indobert-base-p1",
    batch_size: int = 32,
    max_length: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Mengekstrak representasi embedding [CLS] IndoBERT (shape [N, 768]).

    Args:
        texts: Koleksi teks.
        model_name: Identifier model IndoBERT HuggingFace.
        batch_size: Ukuran batch ekstraksi.
        max_length: Panjang maksimum token.
        device: Device komputasi ('cuda' atau 'cpu').

    Returns:
        Array numpy 2D embeddings [N, 768].
    """
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Mengekstrak embeddings IndoBERT (%d teks, device: %s)...", len(texts), dev)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.to(dev)
    bert_model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) if (t is not None and pd.notna(t)) else "" for t in texts[i : i + batch_size]]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(dev)
            outputs = bert_model(**inputs)
            cls_repr = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_repr)

    all_embs = np.vstack(embeddings)
    logger.info("Ekstraksi embeddings selesai. Bentuk: %s", all_embs.shape)
    return all_embs


def run_stratified_kfold_evaluation(
    df: pd.DataFrame,
    config: dict[str, Any],
    n_splits: int = 5,
    classical_model_name: str = "svm",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Menjalankan evaluasi Stratified K-Fold pada seluruh komponen dan ensemble.

    Alur Anti-Leakage:
    1. K-Fold split pada dataset.
    2. Pada setiap fold:
       - Komponen A: Lexicon InSet (zero-shot) menghasilkan probabilitas fold validasi.
       - Komponen B: TF-IDF di-fit HANYA pada train fold, model ML klasik dilatih pada train fold, lalu predict_proba pada fold validasi.
       - Komponen C: IndoBERT classifier dilatih pada train fold, lalu predict_proba pada fold validasi.
    3. Setelah OOF probabilitas seluruh dataset lengkap:
       - Weighted Soft Voting: optimasi bobot pada OOF, prediksi OOF.
       - Stacking: 5-Fold cross-validation meta-learner Logistic Regression pada fitur meta [P_A, P_B, P_C].
    4. Perhitungan metrik per model per fold dan agregasi (Mean ± Std).

    Args:
        df: DataFrame berlabel ground truth (misal data/labeled/youtube_labeled.csv).
        config: Dict konfigurasi config.yaml.
        n_splits: Jumlah fold (default: 5).
        classical_model_name: Model ML klasik ('svm', 'random_forest', atau 'naive_bayes').

    Returns:
        Tuple (DataFrame tabel perbandingan model, Dict data probabilitas OOF).
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    texts = [str(t) if (t is not None and pd.notna(t)) else "" for t in df["text_clean"]]
    y = np.array([LABEL_MAPPING.get(s, 1) for s in df["sentiment"]])
    N = len(df)

    logger.info("Memulai evaluasi Stratified %d-Fold Ensemble (Total Data: %d baris)...", n_splits, N)

    # 1. Komponen A: Lexicon InSet (Zero-shot deterministic)
    lex_model = InSetLexiconModel(config["paths"]["inset_lexicon"])
    oof_a = lex_model.predict_proba(texts)

    # 2. Komponen B: ML Klasik + TF-IDF (OOF)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_b = np.zeros((N, 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y), start=1):
        ext = TextFeatureExtractor(config)
        X_train_tfidf = ext.fit_transform([texts[i] for i in train_idx])
        X_val_tfidf = ext.transform([texts[i] for i in val_idx])

        clf = build_classical_model(classical_model_name, config)
        clf.fit(X_train_tfidf, y[train_idx])
        oof_b[val_idx] = clf.predict_proba(X_val_tfidf)

    # 3. Komponen C: IndoBERT (OOF)
    bert_cfg = config.get("bert", {})
    bert_model_name = bert_cfg.get("model_name", "indobenchmark/indobert-base-p1")
    X_bert = extract_bert_embeddings(
        texts,
        model_name=bert_model_name,
        batch_size=32,
        max_length=64,
    )

    oof_c = np.zeros((N, 3))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_bert, y), start=1):
        bert_meta_clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        )
        bert_meta_clf.fit(X_bert[train_idx], y[train_idx])
        oof_c[val_idx] = bert_meta_clf.predict_proba(X_bert[val_idx])

    # 4. Ensemble 1: Weighted Soft Voting
    soft_voter = WeightedSoftVotingEnsemble()
    soft_voter.fit(oof_a, oof_b, oof_c, y, step=0.02)
    oof_soft_vote = soft_voter.predict_proba(oof_a, oof_b, oof_c)

    # 5. Ensemble 2: Stacking (Meta-Learner Logistic Regression)
    stacker = StackingEnsemble(random_state=seed)
    X_meta = stacker._build_meta_features(oof_a, oof_b, oof_c)
    oof_stacking = np.zeros((N, 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y), start=1):
        meta_lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        )
        meta_lr.fit(X_meta[train_idx], y[train_idx])
        oof_stacking[val_idx] = meta_lr.predict_proba(X_meta[val_idx])

    # Latih meta-learner Stacking akhir pada seluruh data untuk deployment
    stacker.fit(oof_a, oof_b, oof_c, y)

    # Evaluasi metrik per fold untuk mengukur Mean ± Std
    models_dict = {
        "Komponen A (Lexicon InSet)": oof_a,
        f"Komponen B (ML Klasik: {classical_model_name.upper()})": oof_b,
        "Komponen C (IndoBERT)": oof_c,
        "Ensemble: Weighted Soft Voting": oof_soft_vote,
        "Ensemble: Stacking (Meta-LR)": oof_stacking,
    }

    comparison_rows = []

    for model_label, prob_matrix in models_dict.items():
        fold_accs = []
        fold_precs = []
        fold_recs = []
        fold_f1s = []

        for train_idx, val_idx in skf.split(texts, y):
            m = compute_metrics(y[val_idx], np.argmax(prob_matrix[val_idx], axis=1))
            fold_accs.append(m["accuracy"])
            fold_precs.append(m["precision_macro"])
            fold_recs.append(m["recall_macro"])
            fold_f1s.append(m["f1_macro"])

        overall = compute_metrics(y, np.argmax(prob_matrix, axis=1))

        comparison_rows.append({
            "Model": model_label,
            "Accuracy": f"{overall['accuracy'] * 100:.2f}% (±{np.std(fold_accs) * 100:.2f})",
            "Precision-Macro": f"{overall['precision_macro'] * 100:.2f}% (±{np.std(fold_precs) * 100:.2f})",
            "Recall-Macro": f"{overall['recall_macro'] * 100:.2f}% (±{np.std(fold_recs) * 100:.2f})",
            "F1-Macro": f"{overall['f1_macro'] * 100:.2f}% (±{np.std(fold_f1s) * 100:.2f})",
            "raw_f1": overall["f1_macro"],
            "raw_acc": overall["accuracy"],
        })

    df_comparison = pd.DataFrame(comparison_rows)

    # Simpan model ensemble
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stacker, models_dir / "stacking_ensemble.joblib")
    joblib.dump(soft_voter, models_dir / "soft_voting_ensemble.joblib")

    # Simpan tabel hasil ke reports/tables/
    tables_dir = Path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_table_path = tables_dir / "model_comparison.csv"
    df_comparison.drop(columns=["raw_f1", "raw_acc"]).to_csv(out_table_path, index=False, encoding="utf-8-sig")
    logger.info("Tabel perbandingan model disimpan: %s", out_table_path)

    oof_bundle = {
        "oof_a": oof_a,
        "oof_b": oof_b,
        "oof_c": oof_c,
        "oof_soft_vote": oof_soft_vote,
        "oof_stacking": oof_stacking,
        "y_true": y,
        "weights": soft_voter.weights,
        "meta_coefs": stacker.get_meta_coefficients(),
    }

    return df_comparison, oof_bundle


def main() -> None:
    """CLI untuk menjalankan evaluasi ensemble dan perbandingan model."""
    parser = argparse.ArgumentParser(description="Evaluasi Hybrid Ensemble (Stacking & Soft Voting).")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    parser.add_argument("--kfold", type=int, default=5, help="Jumlah fold cross-validation")
    parser.add_argument("--model-b", default="svm", help="Model klasik Komponen B (svm/random_forest/naive_bayes)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    df = pd.read_csv(args.data)
    df_comparison, oof_bundle = run_stratified_kfold_evaluation(
        df,
        config=cfg,
        n_splits=args.kfold,
        classical_model_name=args.model_b,
    )

    print("\n" + "=" * 90)
    print(f"  HASIL EVALUASI STRATIFIED {args.kfold}-FOLD HYBRID ENSEMBLE (DATASET: YOUTUBE)")
    print("=" * 90)
    print(df_comparison[["Model", "Accuracy", "Precision-Macro", "Recall-Macro", "F1-Macro"]].to_string(index=False))
    print("=" * 90)

    weights = oof_bundle["weights"]
    print(f"\nBobot Optimal Soft Voting: Lexicon = {weights['lexicon']:.2f}, "
          f"ML Klasik = {weights['classical']:.2f}, IndoBERT = {weights['bert']:.2f}")

    print("\nKoefisien Meta-Learner Logistic Regression (Stacking):")
    print(oof_bundle["meta_coefs"])
    print("\nFile tabel disimpan ke: reports/tables/model_comparison.csv\n")


if __name__ == "__main__":
    main()
