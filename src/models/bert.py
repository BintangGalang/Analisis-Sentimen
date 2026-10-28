"""
src/models/bert.py
==================
Komponen C: IndoBERT Fine-tuning untuk Analisis Sentimen (PRD §6 F6).

Mendukung:
1. Arsitektur Transformer pretrained (indobenchmark/indobert-base-p1).
2. Fine-tuning loop PyTorch dengan AdamW, linear warmup scheduler, dan gradient clipping.
3. Class-weighted cross entropy untuk menangani data imbalance (PRD §6 F8).
4. Penyimpanan bobot terbaik berdasarkan validation F1-macro.
5. Ekstraksi probabilitas inferensi (predict_proba) untuk ensemble di Tahap 6.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.bert_dataset import SentimentDataset, create_bert_dataloaders
from src.features.tfidf_vectorizer import (
    CLASS_MAPPING,
    INV_CLASS_MAPPING,
    split_dataset,
)
from src.utils import load_config, set_seed

logger = logging.getLogger(__name__)

CLASS_NAMES = ["negatif", "netral", "positif"]


class IndoBERTClassifier:
    """Classifier berbasis fine-tuned IndoBERT untuk 3 kelas sentimen."""

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        num_labels: int = 3,
        device: str | None = None,
        max_length: int = 128,
    ) -> None:
        """Inisialisasi model IndoBERT.

        Args:
            model_name: HuggingFace model identifier.
            num_labels: Jumlah kelas output (3: negatif, netral, positif).
            device: 'cuda' atau 'cpu' (otomatis terdeteksi jika None).
            max_length: Panjang sequence maksimum.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Menginisialisasi IndoBERT (%s) pada device: %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 3,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        class_weights: torch.Tensor | None = None,
        save_path: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Fine-tuning IndoBERT pada training set dengan evaluasi per epoch pada validation set.

        Args:
            train_loader: DataLoader data latih.
            val_loader: DataLoader data validasi.
            num_epochs: Jumlah epoch training.
            lr: Learning rate awal.
            warmup_ratio: Rasio warmup scheduler.
            weight_decay: Weight decay regularizer AdamW.
            gradient_clip: Nilai maksimum gradient clipping.
            class_weights: Tensor bobot kelas untuk CrossEntropyLoss (opsional).
            save_path: Path untuk menyimpan model checkpoint dengan val F1 terbaik.

        Returns:
            Dict riwayat training (train_losses, val_losses, val_f1s).
        """
        if class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val_f1 = -1.0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))

            # Evaluasi pada Validation Set
            val_loss, val_f1 = self._evaluate_loss_and_f1(val_loader, loss_fn)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val F1-Macro: %.4f",
                epoch,
                num_epochs,
                avg_train_loss,
                val_loss,
                val_f1,
            )

            # Simpan bobot terbaik
            if val_f1 > best_val_f1 and save_path is not None:
                best_val_f1 = val_f1
                self.save(save_path)
                logger.info("Checkpoint terbaik baru disimpan (Val F1: %.4f): %s", best_val_f1, save_path)

        return history

    def _evaluate_loss_and_f1(
        self,
        loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
    ) -> tuple[float, float]:
        """Evaluasi loss dan F1-macro pada DataLoader tertentu."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(1, len(loader))
        f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        return avg_loss, f1

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> dict[str, Any]:
        """Evaluasi metrik lengkap (Accuracy, Precision, Recall, F1) pada test set.

        Args:
            test_loader: DataLoader data uji.

        Returns:
            Dict berisi metrik evaluasi.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                if "labels" in batch:
                    all_labels.extend(batch["labels"].cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        acc = float(np.mean(y_true == y_pred))

        report_str = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            labels=[0, 1, 2],
            digits=4,
            zero_division=0,
        )

        return {
            "accuracy": round(acc, 4),
            "precision_macro": round(float(prec), 4),
            "recall_macro": round(float(rec), 4),
            "f1_macro": round(float(f1), 4),
            "report": report_str,
        }

    def predict_proba(
        self,
        texts: list[str] | pd.Series,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Menghasilkan probabilitas softmax (shape [N, 3]) untuk input ke meta-learner ensemble.

        Kolom: [p_negatif, p_netral, p_positif].

        Args:
            texts: Koleksi teks input.
            batch_size: Batch size inferensi.

        Returns:
            Array numpy 2D probabilitas prediksi.
        """
        self.model.eval()
        dataset = SentimentDataset(
            texts=list(texts),
            labels=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probas = []
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = softmax(outputs.logits).cpu().numpy()
                all_probas.append(probs)

        return np.vstack(all_probas)

    def predict(
        self,
        texts: list[str] | pd.Series,
        batch_size: int = 16,
    ) -> list[str]:
        """Memprediksi label string untuk daftar teks.

        Args:
            texts: Koleksi teks input.
            batch_size: Batch size inferensi.

        Returns:
            List label prediksi ('negatif', 'netral', 'positif').
        """
        probas = self.predict_proba(texts, batch_size=batch_size)
        indices = np.argmax(probas, axis=1)
        return [INV_CLASS_MAPPING[idx] for idx in indices]

    def save(self, filepath: str | Path) -> None:
        """Menyimpan model PyTorch state_dict."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Model IndoBERT disimpan ke: %s", path)

    def load(self, filepath: str | Path) -> None:
        """Memuat model PyTorch state_dict."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint model tidak ditemukan: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info("Model IndoBERT dimuat dari: %s", path)


def main() -> None:
    """CLI untuk fine-tuning dan evaluasi IndoBERT."""
    parser = argparse.ArgumentParser(description="Fine-tuning IndoBERT pada data labeled.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default="data/labeled/youtube_labeled.csv")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    df = pd.read_csv(args.data)
    split_cfg = cfg.get("split", {})
    df_train, df_val, df_test = split_dataset(
        df,
        test_size=split_cfg.get("test_size", 0.15),
        val_size=split_cfg.get("val_size", 0.15),
        seed=cfg.get("seed", 42),
    )

    bert_cfg = cfg.get("bert", {})
    model_name = bert_cfg.get("model_name", "indobenchmark/indobert-base-p1")
    max_length = bert_cfg.get("max_length", 128)
    batch_size = bert_cfg.get("batch_size", 16)
    num_epochs = args.epochs if args.epochs is not None else bert_cfg.get("num_epochs", 3)
    lr = float(bert_cfg.get("learning_rate", 2e-5))

    train_loader, val_loader, test_loader, _ = create_bert_dataloaders(
        df_train, df_val, df_test,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    # Hitung bobot kelas untuk penanganan class imbalance (PRD §6 F8)
    y_train_num = [CLASS_MAPPING.get(s, 1) for s in df_train["sentiment"]]
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=np.array(y_train_num))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)
    logger.info("Bobot kelas dihitung: %s", weights)

    classifier = IndoBERTClassifier(
        model_name=model_name,
        num_labels=3,
        max_length=max_length,
    )

    save_path = Path(cfg["paths"]["models_dir"]) / "indobert_best.pt"
    logger.info("Memulai fine-tuning IndoBERT (%d epochs)...", num_epochs)

    classifier.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        class_weights=class_weights_tensor,
        save_path=save_path,
    )

    # Muat checkpoint terbaik untuk evaluasi test set
    if save_path.exists():
        classifier.load(save_path)

    test_metrics = classifier.evaluate(test_loader)

    print("\n" + "=" * 65)
    print("  HASIL EVALUASI KOMPONEN C (INDOBERT)")
    print("=" * 65)
    print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Macro : {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision: {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall   : {test_metrics['recall_macro']:.4f}")
    print("\nLaporan Test Set:")
    print(test_metrics["report"])
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
