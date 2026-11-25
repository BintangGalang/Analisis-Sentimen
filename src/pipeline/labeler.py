"""
Modul Pelabelan Sentimen Tingkat Lanjut (Multi-Option Sentiment Labeling Suite)
Mendukung 4 opsi pelabelan profesional untuk dataset skala besar:
1. 'hybrid'   : Ensemble IndoBERT + Expanded Lexicon + Confidence Scoring (Rekomendasi Utama)
2. 'indobert' : Deep Learning Transformer (mdhugol/indonesia-bert-sentiment-classification)
3. 'lexicon'  : Enhanced Rule-Based Lexicon (InSet + Balinese + Slang, 100% Offline & Cepat)
4. 'audit'    : Human-in-the-Loop CLI & Excel Audit Workflow
"""

import os
import re
import sys
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Pastikan environment aman dari konflik Protobuf TensorFlow di Windows
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from .lexicon_data import (
    POSITIVE_LEXICON,
    NEGATIVE_LEXICON,
    NEGATION_WORDS,
    BOOSTER_WORDS,
    CONTRASTIVE_CONJUNCTIONS,
    EMOJI_SENTIMENT
)
from ..utils.logger import logger


class LexiconLabeler:
    """Kelas pelabelan sentimen berbasis aturan leksikon tingkat lanjut."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        sentiment_cfg = self.config.get("sentiment_analysis", {}).get("lexicon", {})
        
        self.neutral_threshold = sentiment_cfg.get("neutral_threshold", 0.0)
        self.negation_window = sentiment_cfg.get("negation_window", 2)
        self.booster_multiplier = sentiment_cfg.get("booster_multiplier", 1.6)

        self.pos_dict = POSITIVE_LEXICON
        self.neg_dict = NEGATIVE_LEXICON
        self.negations = NEGATION_WORDS
        self.boosters = BOOSTER_WORDS
        self.contrastives = CONTRASTIVE_CONJUNCTIONS
        self.emojis = EMOJI_SENTIMENT
        self.console = Console()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Menganalisis teks dengan memperhitungkan:
        - Deteksi emoji sentimen
        - Jendela negasi (membalik polaritas)
        - Kata booster (meningkatkan bobot)
        - Konjungsi kontrastif (tapi, namun, sayangnya -> klausa akhir dibobot lebih berat)
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": "Netral",
                "sentiment_score": 0.0,
                "confidence": 0.50,
                "pos_score": 0.0,
                "neg_score": 0.0,
                "pos_words": [],
                "neg_words": []
            }

        cleaned = text.lower().strip()
        tokens = re.findall(r"\b\w+\b", cleaned)

        total_score = 0.0
        pos_score = 0.0
        neg_score = 0.0
        detected_pos: List[str] = []
        detected_neg: List[str] = []

        # 1. Deteksi Emoji
        for em, val in self.emojis.items():
            count = text.count(em)
            if count > 0:
                em_score = val * count
                total_score += em_score
                if val > 0:
                    pos_score += em_score
                    detected_pos.append(f"{em}(x{count})")
                else:
                    neg_score += abs(em_score)
                    detected_neg.append(f"{em}(x{count})")

        # 2. Analisis Kata dengan Negasi, Booster, dan Konjungsi Kontras
        i = 0
        n = len(tokens)
        clause_weight = 1.0  # Bobot klausa aktif

        while i < n:
            word = tokens[i]

            # Cek konjungsi kontrastif (contoh: "tapi", "namun", "sayangnya")
            if word in self.contrastives:
                clause_weight = 1.5  # Klausa setelah konjungsi kontras diberi bobot lebih dominan
                i += 1
                continue

            # Cek apakah kata saat ini adalah booster
            booster = 1.0
            if word in self.boosters:
                booster = self.boosters[word]
                if i + 1 < n:
                    i += 1
                    word = tokens[i]

            # Cek negasi sebelum kata ini
            has_negation = False
            start_check = max(0, i - self.negation_window)
            for j in range(start_check, i):
                if tokens[j] in self.negations:
                    has_negation = True
                    break

            # Cek frasa dua kata (bigram)
            bigram = f"{tokens[i]} {tokens[i+1]}" if i + 1 < n else ""

            # Periksa Leksikon Positif
            if bigram and bigram in self.pos_dict:
                score = self.pos_dict[bigram] * booster * clause_weight
                if has_negation:
                    score = -score
                    total_score += score
                    neg_score += abs(score)
                    detected_neg.append(f"tidak_{bigram}")
                else:
                    total_score += score
                    pos_score += score
                    detected_pos.append(bigram)
                i += 2
                continue

            elif word in self.pos_dict:
                score = self.pos_dict[word] * booster * clause_weight
                if has_negation:
                    score = -score
                    total_score += score
                    neg_score += abs(score)
                    detected_neg.append(f"tidak_{word}")
                else:
                    total_score += score
                    pos_score += score
                    detected_pos.append(word)

            # Periksa Leksikon Negatif
            elif bigram and bigram in self.neg_dict:
                score = self.neg_dict[bigram] * booster * clause_weight
                if has_negation:
                    score = abs(score) * 0.8
                    total_score += score
                    pos_score += score
                    detected_pos.append(f"tidak_{bigram}")
                else:
                    total_score += score
                    neg_score += abs(score)
                    detected_neg.append(bigram)
                i += 2
                continue

            elif word in self.neg_dict:
                score = self.neg_dict[word] * booster * clause_weight
                if has_negation:
                    score = abs(score) * 0.8
                    total_score += score
                    pos_score += score
                    detected_pos.append(f"tidak_{word}")
                else:
                    total_score += score
                    neg_score += abs(score)
                    detected_neg.append(word)

            i += 1

        # 3. Penentuan Label Sentimen & Estimasi Confidence
        if total_score > self.neutral_threshold:
            sentiment_label = "Positif"
            confidence = min(0.95, 0.60 + (abs(total_score) * 0.08))
        elif total_score < -self.neutral_threshold:
            sentiment_label = "Negatif"
            confidence = min(0.95, 0.60 + (abs(total_score) * 0.08))
        else:
            sentiment_label = "Netral"
            # Jika tidak ada kata sentimen sama sekali, confidence netral moderat
            confidence = 0.70 if (len(detected_pos) == 0 and len(detected_neg) == 0) else 0.50

        return {
            "sentiment": sentiment_label,
            "sentiment_score": round(total_score, 2),
            "confidence": round(confidence, 2),
            "pos_score": round(pos_score, 2),
            "neg_score": round(neg_score, 2),
            "pos_words": detected_pos,
            "neg_words": detected_neg
        }

    def label_dataset(
        self,
        input_file: str = "data/processed/undiksha_dataset.csv",
        output_prefix: str = "undiksha_dataset_labeled",
        processed_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """Melabeli seluruh dataset menggunakan aturan leksikon dan menyimpan hasilnya."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File dataset tidak ditemukan: {input_file}")

        df = pd.read_csv(input_file)
        text_col = "clean_content" if "clean_content" in df.columns else ("text_clean" if "text_clean" in df.columns else "content")
        
        results = [self.analyze_text(str(t) if pd.notna(t) else "") for t in df[text_col]]
        df["sentiment"] = [r["sentiment"] for r in results]
        df["sentiment_score"] = [r["sentiment_score"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["pos_words"] = [r["pos_words"] for r in results]
        df["neg_words"] = [r["neg_words"] for r in results]
        df["labeling_method"] = "lexicon"

        out_dir = processed_dir or self.config.get("paths", {}).get("processed_dir", "data/processed")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{output_prefix}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df


class IndoBERTLabeler:
    """Kelas pelabelan sentimen berbasis Deep Learning Transformer (IndoBERT)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        label_cfg = self.config.get("sentiment_analysis", {}).get("labeling", {})
        self.model_name = label_cfg.get("model_name", "mdhugol/indonesia-bert-sentiment-classification")
        self.batch_size = label_cfg.get("batch_size", 32)
        
        self.tokenizer = None
        self.model = None
        self.id2label = {0: "Positif", 1: "Netral", 2: "Negatif"}
        self.console = Console()

    def _load_model(self):
        """Memuat model IndoBERT dan tokenizer dari HuggingFace atau cache lokal."""
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"[INDOBERT] Memuat model deep learning: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # Pindahkan ke GPU jika tersedia
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"[INDOBERT] Model siap pada device: {self.device}")

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Menjalankan prediksi batch dengan optimasi PyTorch tanpa gradient tracking."""
        self._load_model()
        import torch

        results: List[Dict[str, Any]] = []
        total = len(texts)

        for start_idx in range(0, total, self.batch_size):
            batch_texts = texts[start_idx : start_idx + self.batch_size]
            cleaned_batch = [str(t).strip() if str(t).strip() else "netral" for t in batch_texts]

            inputs = self.tokenizer(
                cleaned_batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for probs in probabilities:
                pos_prob = float(probs[0])
                neu_prob = float(probs[1])
                neg_prob = float(probs[2])

                best_idx = int(probs.argmax())
                pred_label = self.id2label[best_idx]
                confidence = float(probs[best_idx])

                results.append({
                    "sentiment": pred_label,
                    "confidence": round(confidence, 4),
                    "pos_prob": round(pos_prob, 4),
                    "neu_prob": round(neu_prob, 4),
                    "neg_prob": round(neg_prob, 4)
                })

        return results


class HybridLabeler:
    """
    Kelas pelabelan Ensemble Hybrid: Menggabungkan IndoBERT Transformer dan Expanded Lexicon.
    Menghitung tingkat kesepakatan (agreement), mendeteksi konflik, dan menghasilkan label terpercaya.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.lexicon = LexiconLabeler(config=config)
        self.bert = IndoBERTLabeler(config=config)
        self.console = Console()

    def label_data(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Mengevaluasi seluruh teks menggunakan IndoBERT dan Leksikon secara terpadu."""
        total = len(texts)
        logger.info(f"[HYBRID] Menjalankan pelabelan Hybrid untuk {total} baris...")

        # 1. Jalankan Analisis Leksikon
        lex_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]1/2 Evaluasi Leksikon & Heuristik...[/bold cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as p:
            task = p.add_task("lexicon", total=total)
            for t in texts:
                lex_results.append(self.lexicon.analyze_text(t))
                p.update(task, advance=1)

        # 2. Jalankan Prediksi Deep Learning IndoBERT
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]2/2 Inferensi IndoBERT Deep Learning...[/bold magenta]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as p:
            task = p.add_task("indobert", total=total)
            bert_results = self.bert.predict_batch(texts)
            p.update(task, completed=total)

        # 3. Rekonsiliasi & Penentuan Consensus Label
        combined: List[Dict[str, Any]] = []
        agree_count = 0

        for i in range(total):
            lex = lex_results[i]
            bert = bert_results[i]

            lex_label = lex["sentiment"]
            bert_label = bert["sentiment"]
            bert_conf = bert["confidence"]

            is_concordant = (lex_label == bert_label)
            if is_concordant:
                agree_count += 1
                final_label = bert_label
                # Kesepakatan kedua metode menghasilkan confidence sangat tinggi
                final_confidence = min(0.98, max(bert_conf, 0.88))
                is_conflict = False
            else:
                is_conflict = True
                # Jika ada kata negatif kuat di leksikon (misal: penipuan, rusak, kecewa berat)
                # sementara BERT memprediksi netral dengan confidence rendah
                if len(lex["neg_words"]) >= 2 and bert_label == "Netral":
                    final_label = "Negatif"
                    final_confidence = 0.72
                elif len(lex["pos_words"]) >= 2 and bert_label == "Netral":
                    final_label = "Positif"
                    final_confidence = 0.72
                elif bert_conf >= 0.70:
                    # IndoBERT lebih memahami konteks dan sarkasme
                    final_label = bert_label
                    final_confidence = bert_conf
                else:
                    # Jika confidence bert rendah (< 0.70), gunakan prediksi leksikon jika ada kata sentimen
                    if lex["sentiment_score"] != 0:
                        final_label = lex_label
                        final_confidence = 0.65
                    else:
                        final_label = bert_label
                        final_confidence = bert_conf

            combined.append({
                "sentiment": final_label,
                "confidence": round(final_confidence, 2),
                "is_conflict": is_conflict,
                "bert_sentiment": bert_label,
                "bert_confidence": bert["confidence"],
                "lexicon_sentiment": lex_label,
                "lexicon_score": lex["sentiment_score"],
                "pos_words": ", ".join(lex["pos_words"]),
                "neg_words": ", ".join(lex["neg_words"]),
                "pos_prob": bert["pos_prob"],
                "neu_prob": bert["neu_prob"],
                "neg_prob": bert["neg_prob"]
            })

        agreement_pct = (agree_count / total * 100) if total > 0 else 0
        logger.info(f"[HYBRID] Kesepakatan IndoBERT & Leksikon: {agree_count}/{total} ({agreement_pct:.1f}%)")
        return combined


class InteractiveLabeler:
    """Kelas antarmuka interaktif CLI untuk Human-in-the-Loop review & Active Learning."""

    def __init__(self, dataset_path: str = "data/processed/undiksha_dataset_labeled.csv"):
        self.dataset_path = dataset_path
        self.console = Console()

    def run_interactive_session(self, filter_mode: str = "conflicts") -> None:
        """
        Menjalankan sesi audit interaktif di terminal:
        filter_mode:
        - 'conflicts': Hanya tampilkan baris yang berbeda prediksi antara IndoBERT & Leksikon
        - 'uncertain': Baris dengan confidence < 0.75
        - 'all': Seluruh baris
        """
        if not os.path.exists(self.dataset_path):
            self.console.print(f"[red]Dataset {self.dataset_path} belum ditemukan. Jalankan pelabelan terlebih dahulu.[/red]")
            return

        df = pd.read_csv(self.dataset_path)
        if "sentiment" not in df.columns:
            self.console.print("[red]Kolom 'sentiment' belum ada pada dataset.[/red]")
            return

        text_col = "clean_content" if "clean_content" in df.columns else "content"

        # Tentukan baris yang perlu diaudit
        if filter_mode == "conflicts" and "is_conflict" in df.columns:
            audit_indices = df[df["is_conflict"] == True].index.tolist()
        elif filter_mode == "uncertain" and "confidence" in df.columns:
            audit_indices = df[df["confidence"] < 0.75].index.tolist()
        else:
            audit_indices = df.index.tolist()

        total_to_audit = len(audit_indices)
        if total_to_audit == 0:
            self.console.print("[green]Tidak ada data yang perlu diaudit sesuai filter yang dipilih![/green]")
            return

        self.console.print(Panel(
            f"[bold cyan]SESI AUDIT PELABELAN INTERAKTIF (HUMAN-IN-THE-LOOP)[/bold cyan]\n"
            f"• Mode Filter   : [yellow]{filter_mode}[/yellow]\n"
            f"• Total Antrean : [yellow]{total_to_audit}[/yellow] komentar/post\n\n"
            f"[white]Pilihan Tombol:[/white]\n"
            f"  [bold green][1][/bold green] : Set [green]Positif[/green]\n"
            f"  [bold yellow][2][/bold yellow] : Set [yellow]Netral[/yellow]\n"
            f"  [bold red][3][/bold red] : Set [red]Negatif[/red]\n"
            f"  [bold cyan][Enter][/bold cyan] : Terima Saran Model Saat Ini\n"
            f"  [bold][s][/bold] : Lewati (Skip)\n"
            f"  [bold][q][/bold] : Simpan Hasil & Selesai",
            border_style="cyan"
        ))

        reviewed_count = 0
        try:
            for count, idx in enumerate(audit_indices, 1):
                row = df.loc[idx]
                text = str(row.get(text_col, ""))
                platform = row.get("platform", "medsos")
                curr_sent = row.get("sentiment", "Netral")
                conf = row.get("confidence", 0.0)
                bert_s = row.get("bert_sentiment", "-")
                lex_s = row.get("lexicon_sentiment", "-")

                self.console.print("\n" + "=" * 70)
                self.console.print(f"[bold cyan]Data [{count}/{total_to_audit}] | ID: {row.get('id', idx)} | Platform: {platform}[/bold cyan]")
                self.console.print(f"[white]\"{text}\"[/white]")
                self.console.print(f"Saran IndoBERT: [magenta]{bert_s}[/magenta] | Saran Leksikon: [blue]{lex_s}[/blue] | Current: [bold]{curr_sent}[/bold] (Conf: {conf})")

                choice = input("Pilihan Anda [1:Pos, 2:Net, 3:Neg, Enter:Terima, s:Lewati, q:Keluar]: ").strip().lower()

                if choice == "q":
                    break
                elif choice == "1" or choice == "p":
                    df.at[idx, "sentiment"] = "Positif"
                    df.at[idx, "confidence"] = 1.0
                    df.at[idx, "is_conflict"] = False
                    df.at[idx, "audit_status"] = "human_verified"
                    reviewed_count += 1
                elif choice == "2" or choice == "n":
                    df.at[idx, "sentiment"] = "Netral"
                    df.at[idx, "confidence"] = 1.0
                    df.at[idx, "is_conflict"] = False
                    df.at[idx, "audit_status"] = "human_verified"
                    reviewed_count += 1
                elif choice == "3" or choice == "neg":
                    df.at[idx, "sentiment"] = "Negatif"
                    df.at[idx, "confidence"] = 1.0
                    df.at[idx, "is_conflict"] = False
                    df.at[idx, "audit_status"] = "human_verified"
                    reviewed_count += 1
                elif choice == "":
                    df.at[idx, "confidence"] = 1.0
                    df.at[idx, "is_conflict"] = False
                    df.at[idx, "audit_status"] = "human_approved"
                    reviewed_count += 1
                elif choice == "s":
                    continue
        finally:
            # Simpan perubahan
            df.to_csv(self.dataset_path, index=False, encoding="utf-8")
            xlsx_path = self.dataset_path.replace(".csv", ".xlsx")
            try:
                df.to_excel(xlsx_path, index=False, engine="openpyxl")
            except Exception:
                pass

            # Simpan ke SQLite
            db_path = os.path.join(os.path.dirname(self.dataset_path), "undiksha_dataset.db")
            try:
                conn = sqlite3.connect(db_path)
                df.to_sql("labeled_sentiment_data", conn, if_exists="replace", index=False)
                conn.close()
            except Exception:
                pass

            self.console.print(f"\n[bold green]Selesai! {reviewed_count} data berhasil diverifikasi dan disimpan ke CSV, Excel, dan SQLite.[/bold green]")


class AuditManager:
    """Manajer ekspor dan impor audit Excel untuk verifikasi tim / pembimbing."""

    @staticmethod
    def export_for_audit(
        input_csv: str = "data/processed/undiksha_dataset_labeled.csv",
        output_excel: str = "data/processed/undiksha_labeling_audit.xlsx",
        only_conflicts: bool = True
    ) -> str:
        """Mengekspor data yang memerlukan audit ke format Excel dengan kolom verifikasi manual."""
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"File {input_csv} tidak ditemukan.")

        df = pd.read_csv(input_csv)
        if only_conflicts and "is_conflict" in df.columns:
            export_df = df[df["is_conflict"] == True].copy()
            if len(export_df) == 0:
                export_df = df.sample(min(100, len(df)), random_state=42).copy()
        else:
            export_df = df.copy()

        export_df["manual_label"] = ""
        export_df["catatan_auditor"] = ""

        cols_order = [
            col for col in [
                "id", "platform", "clean_content", "sentiment", "confidence",
                "manual_label", "catatan_auditor", "bert_sentiment", "lexicon_sentiment",
                "pos_words", "neg_words"
            ] if col in export_df.columns
        ]
        export_df = export_df[cols_order]

        export_df.to_excel(output_excel, index=False, engine="openpyxl")
        logger.info(f"[AUDIT] Berkas audit Excel berhasil dibuat di: {output_excel} ({len(export_df)} baris)")
        return output_excel

    @staticmethod
    def import_from_audit(
        audit_excel: str = "data/processed/undiksha_labeling_audit.xlsx",
        target_csv: str = "data/processed/undiksha_dataset_labeled.csv"
    ) -> int:
        """Mengimpor kembali hasil koreksi manual dari file Excel ke dataset master."""
        if not os.path.exists(audit_excel) or not os.path.exists(target_csv):
            raise FileNotFoundError("Berkas audit Excel atau dataset target tidak ditemukan.")

        audit_df = pd.read_excel(audit_excel)
        target_df = pd.read_csv(target_csv)

        if "manual_label" not in audit_df.columns:
            logger.warning("[AUDIT] Kolom 'manual_label' tidak ditemukan pada file Excel.")
            return 0

        # Filter baris yang telah diisi manual_label oleh pengguna
        valid_audits = audit_df[audit_df["manual_label"].notna() & (audit_df["manual_label"].astype(str).str.strip() != "")]
        update_count = 0

        target_df.set_index("id", inplace=True)
        for _, row in valid_audits.iterrows():
            item_id = row["id"]
            new_label = str(row["manual_label"]).strip().capitalize()
            if new_label in ["Positif", "Netral", "Negatif"] and item_id in target_df.index:
                target_df.at[item_id, "sentiment"] = new_label
                target_df.at[item_id, "confidence"] = 1.0
                target_df.at[item_id, "is_conflict"] = False
                target_df.at[item_id, "audit_status"] = "human_verified"
                update_count += 1

        target_df.reset_index(inplace=True)
        target_df.to_csv(target_csv, index=False, encoding="utf-8")
        
        # Update SQLite
        db_path = os.path.join(os.path.dirname(target_csv), "undiksha_dataset.db")
        try:
            conn = sqlite3.connect(db_path)
            target_df.to_sql("labeled_sentiment_data", conn, if_exists="replace", index=False)
            conn.close()
        except Exception as e:
            logger.warning(f"[AUDIT] Gagal memperbarui SQLite: {e}")

        logger.info(f"[AUDIT] Berhasil memperbarui {update_count} label dari Excel ke dataset master.")
        return update_count


class SentimentLabelingOrchestrator:
    """Orkestrator utama untuk menjalankan pelabelan sesuai metode pilihan."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.console = Console()

    def label_dataset(
        self,
        method: str = "hybrid",
        input_file: str = "data/processed/undiksha_dataset.csv",
        output_prefix: str = "undiksha_dataset_labeled",
        processed_dir: str = "data/processed"
    ) -> pd.DataFrame:
        """
        Menjalankan alur pelabelan sentimen:
        - method='hybrid'   : Ensemble IndoBERT + Leksikon (Rekomendasi Utama)
        - method='indobert' : SOTA Transformer Deep Learning
        - method='lexicon'  : Enhanced Rule-Based Leksikon
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File dataset input tidak ditemukan: {input_file}")

        logger.info(f"[LABELING] Memuat dataset dari: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"[LABELING] Total baris data: {len(df)} | Metode: {method.upper()}")

        text_column = "clean_content" if "clean_content" in df.columns else "content"
        texts = df[text_column].fillna("").astype(str).tolist()

        if method == "indobert":
            bert_labeler = IndoBERTLabeler(config=self.config)
            results = bert_labeler.predict_batch(texts)
            df["sentiment"] = [r["sentiment"] for r in results]
            df["confidence"] = [r["confidence"] for r in results]
            df["pos_prob"] = [r["pos_prob"] for r in results]
            df["neu_prob"] = [r["neu_prob"] for r in results]
            df["neg_prob"] = [r["neg_prob"] for r in results]
            df["labeling_method"] = "indobert"
            df["is_conflict"] = False

        elif method == "lexicon":
            lex_labeler = LexiconLabeler(config=self.config)
            results = [lex_labeler.analyze_text(t) for t in texts]
            df["sentiment"] = [r["sentiment"] for r in results]
            df["sentiment_score"] = [r["sentiment_score"] for r in results]
            df["confidence"] = [r["confidence"] for r in results]
            df["pos_words"] = [", ".join(r["pos_words"]) for r in results]
            df["neg_words"] = [", ".join(r["neg_words"]) for r in results]
            df["labeling_method"] = "lexicon"
            df["is_conflict"] = False

        else:  # Default 'hybrid'
            hybrid_labeler = HybridLabeler(config=self.config)
            results = hybrid_labeler.label_data(texts)
            df["sentiment"] = [r["sentiment"] for r in results]
            df["confidence"] = [r["confidence"] for r in results]
            df["is_conflict"] = [r["is_conflict"] for r in results]
            df["bert_sentiment"] = [r["bert_sentiment"] for r in results]
            df["bert_confidence"] = [r["bert_confidence"] for r in results]
            df["lexicon_sentiment"] = [r["lexicon_sentiment"] for r in results]
            df["lexicon_score"] = [r["lexicon_score"] for r in results]
            df["pos_words"] = [r["pos_words"] for r in results]
            df["neg_words"] = [r["neg_words"] for r in results]
            df["pos_prob"] = [r["pos_prob"] for r in results]
            df["neu_prob"] = [r["neu_prob"] for r in results]
            df["neg_prob"] = [r["neg_prob"] for r in results]
            df["labeling_method"] = "hybrid"

        # Klasifikasi 7 Aspek Reputasi Kampus
        try:
            from src.evaluate.aspect import classify_aspect
            df["aspect"] = [classify_aspect(t) for t in texts]
            logger.info("[LABELING] Klasifikasi 7 aspek reputasi berhasil disematkan.")
        except Exception as e:
            logger.warning(f"[LABELING] Klasifikasi aspek dilewati: {e}")

        # Sinkronisasi nama kolom standar untuk kompatibilitas dashboard
        if "clean_content" in df.columns:
            df["text_clean"] = df["clean_content"]
        if "content" in df.columns:
            df["text_raw"] = df["content"]

        # Simpan ke CSV
        csv_path = os.path.join(processed_dir, f"{output_prefix}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"[LABELING] Dataset CSV tersimpan di: {csv_path}")

        # Sinkronisasi ke data/labeled/combined_labeled.csv untuk Dashboard & Evaluasi
        try:
            labeled_master = "data/labeled/combined_labeled.csv"
            os.makedirs(os.path.dirname(labeled_master), exist_ok=True)
            df.to_csv(labeled_master, index=False, encoding="utf-8")
            logger.info(f"[LABELING] Master dataset tersinkronisasi di: {labeled_master}")
        except Exception as e:
            logger.warning(f"[LABELING] Gagal menyinkronkan master dataset: {e}")

        # Simpan ke Excel
        xlsx_path = os.path.join(processed_dir, f"{output_prefix}.xlsx")
        try:
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
            logger.info(f"[LABELING] Dataset Excel tersimpan di: {xlsx_path}")
        except Exception as e:
            logger.warning(f"[LABELING] Gagal menyimpan Excel: {e}")

        # Simpan ke SQLite
        db_path = os.path.join(processed_dir, "undiksha_dataset.db")
        try:
            conn = sqlite3.connect(db_path)
            df.to_sql("labeled_sentiment_data", conn, if_exists="replace", index=False)
            conn.close()
            logger.info(f"[LABELING] SQLite table 'labeled_sentiment_data' tersimpan di: {db_path}")
        except Exception as e:
            logger.warning(f"[LABELING] Gagal menyimpan ke SQLite: {e}")

        self.display_summary(df, method)
        return df

    def display_summary(self, df: pd.DataFrame, method: str) -> None:
        """Menampilkan tabel ringkasan hasil pelabelan dan metrik kualitas di terminal."""
        counts = df["sentiment"].value_counts()
        total = len(df)

        table = Table(
            title=f"=== Ringkasan Pelabelan Sentimen (Metode: {method.upper()}) ===",
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("Sentimen", justify="center")
        table.add_column("Jumlah Data", justify="right")
        table.add_column("Persentase (%)", justify="right")

        for label in ["Positif", "Netral", "Negatif"]:
            c = counts.get(label, 0)
            pct = (c / total * 100) if total > 0 else 0
            style = "green" if label == "Positif" else ("yellow" if label == "Netral" else "red")
            table.add_row(f"[{style}]{label}[/{style}]", str(c), f"{pct:.1f}%")

        table.add_section()
        table.add_row("[bold]TOTAL DATA[/bold]", f"[bold]{total}[/bold]", "[bold]100.0%[/bold]")
        self.console.print(table)

        # Informasi metrik kualitas jika ada
        if "confidence" in df.columns:
            avg_conf = df["confidence"].mean() * 100
            conf_str = f"• Rata-rata Skor Keyakinan (Confidence): [bold green]{avg_conf:.1f}%[/bold green]"
            if "is_conflict" in df.columns:
                conflicts = df["is_conflict"].sum()
                conflict_pct = (conflicts / total * 100) if total > 0 else 0
                conf_str += f"\n• Jumlah Potensi Ambiguitas/Konflik: [bold yellow]{conflicts} ({conflict_pct:.1f}%)[/bold yellow]"
                conf_str += f"\n• Tips: Jalankan [cyan]python main.py --audit[/cyan] untuk memverifikasi data ambigu secara cepat."
            self.console.print(Panel(conf_str, title="[bold]Metrik Kualitas Pelabelan[/bold]", border_style="green"))
