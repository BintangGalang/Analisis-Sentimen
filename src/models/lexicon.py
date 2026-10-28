"""
src/models/lexicon.py
=====================
Komponen A: Model berbasis Leksikon InSet (Indonesian Sentiment Lexicon) (PRD §6 F4).

Menghasilkan:
1. Skor polaritas (skor positif, skor negatif, skor net).
2. Prediksi label diskret: 'positif', 'netral', 'negatif'.
3. Fitur probabilitas (shape [N, 3]) untuk soft voting / stacking ensemble.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)

# Urutan kelas standar untuk konsistensi probabilitas
CLASS_ORDER = ["negatif", "netral", "positif"]


class InSetLexiconModel:
    """Model klasifikasi sentimen berbasis InSet Lexicon dengan skor polaritas."""

    def __init__(
        self,
        lexicon_path: str | Path = "data/interim/inset_lexicon.tsv",
        neutral_threshold: float = 0.5,
    ) -> None:
        """Inisialisasi model leksikon.

        Args:
            lexicon_path: Path ke file TSV leksikon InSet.
            neutral_threshold: Ambang batas selisih skor untuk dikategorikan netral.
        """
        self.lexicon_path = Path(lexicon_path)
        self.neutral_threshold = float(neutral_threshold)
        self.lexicon: dict[str, float] = {}
        self._load_lexicon()

    def _load_lexicon(self) -> None:
        """Memuat kamus kata dan bobot dari file TSV."""
        if not self.lexicon_path.exists():
            raise FileNotFoundError(f"File leksikon tidak ditemukan: {self.lexicon_path}")

        df_lex = pd.read_csv(self.lexicon_path, sep="\t")
        for _, row in df_lex.iterrows():
            word = str(row["word"]).strip().lower()
            weight = float(row["weight"])
            self.lexicon[word] = weight

        logger.info(
            "InSet lexicon dimuat dari %s (%d kata)",
            self.lexicon_path,
            len(self.lexicon),
        )

    def score_text(self, text: str) -> dict[str, float]:
        """Menghitung rincian skor leksikon untuk satu teks.

        Args:
            text: Teks yang akan dianalisis.

        Returns:
            Dict berisi pos_score, neg_score (absolut), net_score, dan word_count.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "pos_score": 0.0,
                "neg_score": 0.0,
                "net_score": 0.0,
                "matched_words": 0,
            }

        text_lower = text.lower()
        tokens = re.findall(r"\b\w+\b", text_lower)

        pos_sum = 0.0
        neg_sum = 0.0
        matched = 0

        # Pencocokan unigram
        for tok in tokens:
            if tok in self.lexicon:
                w = self.lexicon[tok]
                if w > 0:
                    pos_sum += w
                else:
                    neg_sum += abs(w)
                matched += 1

        # Pencocokan n-gram multi-kata di leksikon (misal: "terima kasih", "luar biasa")
        for phrase, w in self.lexicon.items():
            if " " in phrase and phrase in text_lower:
                if w > 0:
                    pos_sum += w
                else:
                    neg_sum += abs(w)
                matched += 1

        net_score = pos_sum - neg_sum
        return {
            "pos_score": float(pos_sum),
            "neg_score": float(neg_sum),
            "net_score": float(net_score),
            "matched_words": matched,
        }

    def predict_one(self, text: str) -> str:
        """Memprediksi label sentimen untuk satu teks.

        Args:
            text: Teks input.

        Returns:
            'positif', 'netral', atau 'negatif'.
        """
        scores = self.score_text(text)
        net = scores["net_score"]

        if net > self.neutral_threshold:
            return "positif"
        elif net < -self.neutral_threshold:
            return "negatif"
        else:
            return "netral"

    def predict(self, texts: list[str] | pd.Series) -> list[str]:
        """Memprediksi label untuk daftar teks.

        Args:
            texts: Koleksi teks.

        Returns:
            List label prediksi.
        """
        return [self.predict_one(t) for t in texts]

    def predict_proba(self, texts: list[str] | pd.Series) -> np.ndarray:
        """Menghasilkan probabilitas pseudo-softmax untuk kelas [negatif, netral, positif].

        Fitur probabilitas ini akan menjadi input bagi Stacking meta-learner di Tahap 6.

        Args:
            texts: Koleksi teks.

        Returns:
            Array numpy 2D dengan shape (N, 3), kolom: [p_negatif, p_netral, p_positif].
        """
        probas = []
        for text in texts:
            sc = self.score_text(text)
            pos = sc["pos_score"]
            neg = sc["neg_score"]

            # Logit sederhana
            logit_neg = neg
            logit_pos = pos
            logit_net = max(0.5, 2.0 - (pos + neg) * 0.2)

            logits = np.array([logit_neg, logit_net, logit_pos], dtype=np.float64)
            # Softmax yang stabil
            exp_logits = np.exp(logits - np.max(logits))
            prob = exp_logits / np.sum(exp_logits)
            probas.append(prob)

        return np.array(probas)


def main() -> None:
    """CLI evaluasi model InSet Lexicon pada data ground truth."""
    parser = argparse.ArgumentParser(description="Evaluasi model Lexicon InSet.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset %s tidak ditemukan.", data_path)
        return

    df = pd.read_csv(data_path)
    model = InSetLexiconModel(lexicon_path=cfg["paths"]["inset_lexicon"])

    y_true = df["sentiment"].astype(str).str.lower().tolist()
    y_pred = model.predict(df["text_clean"].tolist())

    f1_macro = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 55)
    print("  HASIL EVALUASI KOMPONEN A (INSET LEXICON)")
    print("=" * 55)
    print(f"Dataset      : {data_path} ({len(df)} baris)")
    print(f"F1-Macro     : {f1_macro:.4f}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_true, y_pred, labels=CLASS_ORDER, digits=4))
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
