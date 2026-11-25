"""
Modul Pemodelan & Klasifikasi Sentimen (Machine Learning)
Melatih model klasifikasi teks berbasis dataset yang telah dilabeli,
mengevaluasi metrik performa, dan menyimpan model untuk inferensi.
Mendukung:
- Linear SVM (LinearSVC - Balanced)
- Complement Naïve Bayes (Imbalance-Tolerant)
- Random Forest (Balanced)
- Logistic Regression (Balanced)
- Multinomial Naïve Bayes
"""

import os
import pickle
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from rich.console import Console
from rich.table import Table

from ..utils.logger import logger


class SentimentClassifier:
    """Kelas untuk melatih, mengevaluasi, dan menjalankan prediksi model sentimen."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        model_cfg = self.config.get("sentiment_analysis", {}).get("modeling", {})
        
        self.default_model = model_cfg.get("default_model", "svm")
        self.test_size = model_cfg.get("test_size", 0.2)
        self.random_state = model_cfg.get("random_state", 42)
        self.models_dir = model_cfg.get("models_dir", "data/models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.console = Console()
        self.classes_ = ["Negatif", "Netral", "Positif"]

    def train_and_evaluate(
        self,
        dataset_path: str = "data/processed/undiksha_dataset_labeled.csv",
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Melatih model klasifikasi menggunakan dataset berlabel dan mengembalikan metrik evaluasi.
        Mendukung Linear SVM, Complement Naïve Bayes, Random Forest, Logistic Regression, dan Multinomial NB
        dengan penanganan ketimpangan data (class_weight='balanced').
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset berlabel tidak ditemukan di: {dataset_path}")

        logger.info(f"[MODELING] Membaca dataset berlabel dari: {dataset_path}")
        df = pd.read_csv(dataset_path)

        text_col = "clean_content" if "clean_content" in df.columns else "content"
        # Filter baris yang memiliki teks dan label
        df = df.dropna(subset=[text_col, "sentiment"])
        df = df[df[text_col].str.strip() != ""]

        if len(df) < 5:
            raise ValueError("Dataset terlalu sedikit untuk pelatihan model (minimal 5 baris).")

        X = df[text_col].astype(str)
        y = df["sentiment"].astype(str)

        # Cek apakah setiap kelas memiliki sampel yang cukup untuk stratified split
        class_counts = y.value_counts()
        stratify = y if (class_counts.min() >= 2 and len(df) >= 10) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

        logger.info(f"[MODELING] Ukuran data latih: {len(X_train)}, data uji: {len(X_test)}")

        # 1. Ekstraksi Fitur TF-IDF dengan sublinear scaling untuk peredam frekuensi kata tinggi
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
            min_df=1
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # 2. Inisialisasi & Pelatihan Model
        selected_type = (model_type or self.default_model).lower().strip()

        if selected_type in ["svm", "linear_svm", "linearsvc"]:
            self.model = LinearSVC(
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=2000
            )
            self.model_name = "Linear SVM (LinearSVC - Balanced)"
        elif selected_type in ["complement_nb", "cnb"]:
            # ComplementNB secara matematis dirancang khusus untuk dataset teks yang tidak seimbang (imbalanced)
            self.model = ComplementNB(alpha=1.0)
            self.model_name = "Complement Naïve Bayes (Imbalance-Tolerant)"
        elif selected_type in ["random_forest", "rf"]:
            self.model = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model_name = "Random Forest (Balanced)"
        elif selected_type in ["naive_bayes", "multinomial_nb", "mnb"]:
            self.model = MultinomialNB(alpha=1.0)
            self.model_name = "Multinomial Naïve Bayes"
        elif selected_type in ["logistic_regression", "lr"]:
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )
            self.model_name = "Logistic Regression (Balanced)"
        else:
            logger.warning(f"[MODELING] Algoritma '{selected_type}' tidak dikenal, beralih ke default Linear SVM.")
            self.model = LinearSVC(
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=2000
            )
            self.model_name = "Linear SVM (LinearSVC - Balanced)"

        logger.info(f"[MODELING] Melatih algoritma: {self.model_name}...")
        self.model.fit(X_train_vec, y_train)

        # 3. Prediksi & Evaluasi
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Ambil label unik yang muncul
        labels = sorted(list(set(y_test) | set(y_pred)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report_dict = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

        # Simpan Model & Vectorizer
        model_path = os.path.join(self.models_dir, "sentiment_model.pkl")
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(vec_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        logger.info(f"[MODELING] Model ({self.model_name}) tersimpan di: {model_path}")
        logger.info(f"[MODELING] TF-IDF Vectorizer tersimpan di: {vec_path}")

        # Tampilkan Hasil Evaluasi di Terminal
        self.display_evaluation(self.model_name, accuracy, report_dict, labels)

        return {
            "model_name": self.model_name,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "labels": labels,
            "report": report_dict,
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        }

    def display_evaluation(self, model_name: str, accuracy: float, report: Dict[str, Any], labels: list) -> None:
        """Menampilkan metrik evaluasi model di terminal menggunakan Rich."""
        table = Table(title=f"=== Hasil Evaluasi Model: {model_name} ===", show_header=True, header_style="bold cyan")
        table.add_column("Kelas Sentimen", justify="center")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-Score", justify="right")
        table.add_column("Support (Jumlah)", justify="right")

        for lbl in labels:
            if lbl in report:
                r = report[lbl]
                table.add_row(
                    lbl,
                    f"{r['precision']:.2f}",
                    f"{r['recall']:.2f}",
                    f"{r['f1-score']:.2f}",
                    str(int(r['support']))
                )

        table.add_section()
        acc_pct = accuracy * 100
        table.add_row("[bold green]AKURASI TOTAL[/bold green]", "-", "-", f"[bold green]{acc_pct:.1f}%[/bold green]", str(int(report.get("macro avg", {}).get("support", 0))))
        self.console.print(table)

    def predict(self, text: str) -> Dict[str, Any]:
        """Memprediksi sentimen untuk teks baru menggunakan model terlatih."""
        if self.model is None or self.vectorizer is None:
            self.load_model()

        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        confidence = 0.0

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            confidence = float(np.max(probs))
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(vec)[0]
            decision = np.array(decision, dtype=float)
            exp_d = np.exp(decision - np.max(decision))
            probs = exp_d / exp_d.sum()
            confidence = float(np.max(probs))

        return {
            "text": text,
            "predicted_sentiment": pred,
            "confidence": round(confidence, 4)
        }

    def load_model(self) -> None:
        """Memuat model dan vectorizer dari penyimpanan lokal."""
        model_path = os.path.join(self.models_dir, "sentiment_model.pkl")
        vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            raise FileNotFoundError("Model belum dilatih. Silakan jalankan train_and_evaluate() terlebih dahulu.")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        logger.info("[MODELING] Model dan TF-IDF Vectorizer berhasil dimuat.")
