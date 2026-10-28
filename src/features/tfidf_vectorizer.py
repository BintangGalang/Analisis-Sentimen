"""
src/features/tfidf_vectorizer.py
================================
Ekstraksi fitur TF-IDF dan pipeline Anti-Leakage data split (PRD §6 F5, F8).

Prinsip Anti-Leakage (sesuai project-rules):
1. Stratified Train / Val / Test Split (70/15/15) dilakukan SEBELUM fit TF-IDF.
2. TF-IDF di-fit HANYA pada data latih (X_train).
3. SMOTE oversampling HANYA diaplikasikan pada X_train_tfidf, tidak pernah pada val/test.
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
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)

CLASS_MAPPING = {"negatif": 0, "netral": 1, "positif": 2}
INV_CLASS_MAPPING = {0: "negatif", 1: "netral", 2: "positif"}


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    stratify_col: str = "sentiment",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Membagi dataset menjadi Train, Validation, dan Test secara stratified.

    Contoh: test_size=0.15, val_size=0.15 -> Train 70%, Val 15%, Test 15%.

    Args:
        df: DataFrame berlabel.
        test_size: Proporsi test set (misal 0.15).
        val_size: Proporsi validation set (misal 0.15).
        seed: Random seed.
        stratify_col: Kolom target stratifikasi.

    Returns:
        Tuple (df_train, df_val, df_test).
    """
    stratify = df[stratify_col] if stratify_col in df.columns else None

    # Step 1: Pisahkan Test set
    df_temp, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    # Step 2: Dari sisa data (df_temp), pisahkan Validation set
    # Penyesuaian proporsi val terhadap sisa
    val_rel_size = val_size / (1.0 - test_size)
    stratify_temp = df_temp[stratify_col] if stratify_col in df_temp.columns else None

    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_rel_size,
        random_state=seed,
        stratify=stratify_temp,
    )

    logger.info(
        "Dataset dibagi (Anti-Leakage Split): Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(df_train),
        len(df_train) / len(df) * 100,
        len(df_val),
        len(df_val) / len(df) * 100,
        len(df_test),
        len(df_test) / len(df) * 100,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


class TextFeatureExtractor:
    """Ekstraktor fitur teks berbasis TF-IDF dengan kepatuhan anti-leakage."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Inisialisasi konfigurasi TF-IDF dan SMOTE.

        Args:
            config: Dict konfigurasi dari config.yaml.
        """
        self.config = config or {}
        tfidf_cfg = self.config.get("tfidf", {})

        self.max_features = tfidf_cfg.get("max_features", 10000)
        ngram = tfidf_cfg.get("ngram_range", [1, 2])
        self.ngram_range = tuple(ngram)
        self.sublinear_tf = tfidf_cfg.get("sublinear_tf", True)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.is_fitted = False

    def fit(self, texts_train: list[str] | pd.Series) -> TextFeatureExtractor:
        """Fit vectorizer HANYA pada data latih (Anti-Leakage).

        Args:
            texts_train: Teks dari kumpulan data latih.

        Returns:
            Self.
        """
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts_train]
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True
        logger.info(
            "TF-IDF Vectorizer di-fit pada %d teks latih (vocabulary: %d fitur).",
            len(clean_texts),
            len(self.vectorizer.vocabulary_),
        )
        return self

    def transform(self, texts: list[str] | pd.Series) -> Any:
        """Transform teks menjadi sparse matrix TF-IDF.

        Args:
            texts: Teks yang akan ditransformasi.

        Returns:
            Scipy sparse matrix hasil TF-IDF.
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer belum di-fit! Panggil fit() terlebih dahulu.")

        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        return self.vectorizer.transform(clean_texts)

    def fit_transform(self, texts_train: list[str] | pd.Series) -> Any:
        """Fit pada data latih dan langsung transform."""
        return self.fit(texts_train).transform(texts_train)

    def apply_smote(
        self,
        X_train_tfidf: Any,
        y_train: np.ndarray | list[int],
        k_neighbors: int = 5,
        seed: int = 42,
    ) -> tuple[Any, np.ndarray]:
        """Menerapkan SMOTE oversampling HANYA pada training set (F8).

        Args:
            X_train_tfidf: Matriks TF-IDF data latih.
            y_train: Label numerik data latih.
            k_neighbors: Jumlah tetangga terdekat untuk SMOTE.
            seed: Random seed.

        Returns:
            Tuple (X_train_resampled, y_train_resampled).
        """
        # Sesuaikan k_neighbors jika jumlah sampel kelas minoritas sangat sedikit
        y_arr = np.array(y_train)
        min_class_count = min(np.bincount(y_arr))
        actual_k = max(1, min(k_neighbors, min_class_count - 1))

        if actual_k < 1:
            logger.warning("Kelas minoritas terlalu sedikit untuk SMOTE, lewati oversampling.")
            return X_train_tfidf, y_arr

        smote = SMOTE(k_neighbors=actual_k, random_state=seed)
        X_res, y_res = smote.fit_resample(X_train_tfidf, y_arr)
        logger.info(
            "SMOTE diaplikasikan pada X_train: %d sampel -> %d sampel (distribusi: %s)",
            X_train_tfidf.shape[0],
            X_res.shape[0],
            dict(pd.Series(y_res).value_counts()),
        )
        return X_res, y_res

    def save(self, filepath: str | Path) -> None:
        """Menyimpan vectorizer ke file joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info("TF-IDF Vectorizer disimpan ke: %s", path)

    def load(self, filepath: str | Path) -> None:
        """Memuat vectorizer dari file joblib."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File model vectorizer tidak ada: {path}")
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        logger.info("TF-IDF Vectorizer dimuat dari: %s", path)


def prepare_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    text_col: str = "text_clean",
    use_smote: bool = True,
) -> dict[str, Any]:
    """Fungsi end-to-end untuk split dataset, ekstraksi fitur TF-IDF, dan SMOTE pada train set.

    Args:
        df: DataFrame berlabel.
        config: Konfigurasi proyek.
        text_col: Nama kolom teks.
        use_smote: Apakah menggunakan SMOTE pada train set.

    Returns:
        Dict berisi X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, extractor.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    split_cfg = config.get("split", {})
    df_train, df_val, df_test = split_dataset(
        df,
        test_size=split_cfg.get("test_size", 0.15),
        val_size=split_cfg.get("val_size", 0.15),
        seed=seed,
    )

    extractor = TextFeatureExtractor(config)

    # ANTI-LEAKAGE: Fit HANYA pada train
    X_train = extractor.fit_transform(df_train[text_col])
    X_val = extractor.transform(df_val[text_col])
    X_test = extractor.transform(df_test[text_col])

    # Mapping target string ke integer
    y_train = np.array([CLASS_MAPPING.get(s, 1) for s in df_train["sentiment"]])
    y_val = np.array([CLASS_MAPPING.get(s, 1) for s in df_val["sentiment"]])
    y_test = np.array([CLASS_MAPPING.get(s, 1) for s in df_test["sentiment"]])

    if use_smote and config.get("imbalance", {}).get("method") == "smote":
        k_neigh = config.get("imbalance", {}).get("k_neighbors", 5)
        X_train, y_train = extractor.apply_smote(X_train, y_train, k_neighbors=k_neigh, seed=seed)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "extractor": extractor,
    }


def main() -> None:
    """CLI untuk pengujian ekstraksi fitur TF-IDF."""
    parser = argparse.ArgumentParser(description="Ekstraksi fitur TF-IDF dengan anti-leakage.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.data)

    data_bundle = prepare_features(df, cfg)
    print("\n" + "=" * 55)
    print("  RINGKASAN FITUR TF-IDF & ANTI-LEAKAGE SPLIT")
    print("=" * 55)
    print(f"Bentuk X_train : {data_bundle['X_train'].shape}")
    print(f"Bentuk X_val   : {data_bundle['X_val'].shape}")
    print(f"Bentuk X_test  : {data_bundle['X_test'].shape}")
    print(f"Distribusi y_train : {np.bincount(data_bundle['y_train'])}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
