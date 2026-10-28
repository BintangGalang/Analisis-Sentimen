"""
tests/test_bert_dataset.py
==========================
Unit tests untuk src/features/bert_dataset.py.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from src.features.bert_dataset import SentimentDataset, create_bert_dataloaders


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")


class TestSentimentDataset:
    def test_dataset_item_shape(self, tokenizer) -> None:
        texts = ["kampus undiksha bagus", "siakad error lelet"]
        labels = [2, 0]
        ds = SentimentDataset(texts, labels, tokenizer, max_length=32)

        assert len(ds) == 2
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["labels"].item() == 2

    def test_create_bert_dataloaders(self) -> None:
        df_tr = pd.DataFrame({"text_clean": ["undiksha bagus", "siakad mantap"], "sentiment": ["positif", "positif"]})
        df_va = pd.DataFrame({"text_clean": ["krs kapan"], "sentiment": ["netral"]})
        df_te = pd.DataFrame({"text_clean": ["lelet parah"], "sentiment": ["negatif"]})

        tr_loader, va_loader, te_loader, tok = create_bert_dataloaders(
            df_tr, df_va, df_te,
            max_length=16,
            batch_size=2,
        )

        assert len(tr_loader) == 1
        assert len(va_loader) == 1
        assert len(te_loader) == 1

        batch = next(iter(tr_loader))
        assert batch["input_ids"].shape == (2, 16)
