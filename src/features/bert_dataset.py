"""
src/features/bert_dataset.py
============================
PyTorch Dataset dan DataLoader untuk tokenisasi IndoBERT (PRD §6 F6).

Mendukung:
- Tokenizer HuggingFace (indobenchmark/indobert-base-p1).
- Truncation dan padding dinamis hingga max_length.
- Pemetaan label sentimen: 'negatif': 0, 'netral': 1, 'positif': 2.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.tfidf_vectorizer import CLASS_MAPPING

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """PyTorch Dataset untuk teks sentimen dengan tokenisasi IndoBERT."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        tokenizer: Any = None,
        max_length: int = 128,
    ) -> None:
        """Inisialisasi dataset sentimen.

        Args:
            texts: Daftar teks input.
            labels: Daftar label numerik (opsional jika inferensi).
            tokenizer: HuggingFace PreTrainedTokenizer.
            max_length: Panjang maksimum token (padding/truncation).
        """
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def create_bert_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str = "indobenchmark/indobert-base-p1",
    max_length: int = 128,
    batch_size: int = 16,
    text_col: str = "text_clean",
) -> tuple[DataLoader, DataLoader, DataLoader, Any]:
    """Membuat DataLoader PyTorch untuk train, validation, dan test set.

    Args:
        df_train: Data latih.
        df_val: Data validasi.
        df_test: Data uji.
        model_name: Nama model pretrained di HuggingFace.
        max_length: Panjang maksimum sequence.
        batch_size: Batch size untuk DataLoader.
        text_col: Nama kolom teks.

    Returns:
        Tuple (train_loader, val_loader, test_loader, tokenizer).
    """
    logger.info("Memuat tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Konversi target teks ke numerik
    y_train = [CLASS_MAPPING.get(s, 1) for s in df_train["sentiment"]]
    y_val   = [CLASS_MAPPING.get(s, 1) for s in df_val["sentiment"]]
    y_test  = [CLASS_MAPPING.get(s, 1) for s in df_test["sentiment"]]

    train_ds = SentimentDataset(df_train[text_col].tolist(), y_train, tokenizer, max_length)
    val_ds   = SentimentDataset(df_val[text_col].tolist(), y_val, tokenizer, max_length)
    test_ds  = SentimentDataset(df_test[text_col].tolist(), y_test, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "DataLoaders dibuat: Train=%d batch, Val=%d batch, Test=%d batch (Batch Size=%d)",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        batch_size,
    )

    return train_loader, val_loader, test_loader, tokenizer
