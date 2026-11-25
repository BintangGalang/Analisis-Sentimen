"""
Modul Transformer & Normalizer
Menggabungkan data mentah JSONL dari seluruh platform, membersihkan teks,
menghilangkan duplikasi, dan mengekspor ke format CSV, Excel, dan SQLite.
"""

import os
import sqlite3
from typing import Dict, Any, List, Optional
import pandas as pd
from rich.table import Table

from ..utils.logger import logger, console
from ..utils.helpers import load_config, read_jsonl
from .cleaner import TextCleaner


class DataTransformer:
    """Kelas untuk memproses data mentah menjadi dataset terstruktur siap pakai."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None
    ):
        self.config = load_config(config_path)
        self.cleaner = TextCleaner(self.config)
        self.raw_dir = raw_dir or self.config.get("paths", {}).get("raw_dir", self.config.get("paths", {}).get("raw_data_dir", "data/raw"))
        self.processed_dir = processed_dir or self.config.get("paths", {}).get("processed_dir", self.config.get("paths", {}).get("processed_data_dir", "data/processed"))
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_all_raw_data(self) -> List[Dict[str, Any]]:
        """Membaca semua file data mentah JSONL (IG, FB, Threads, TikTok) dan CSV (YouTube, Twitter/X) dari folder raw_data_dir."""
        all_records = []
        if not os.path.exists(self.raw_dir):
            logger.warning(f"Direktori raw data {self.raw_dir} belum dibuat atau masih kosong.")
            return all_records

        for filename in os.listdir(self.raw_dir):
            filepath = os.path.join(self.raw_dir, filename)
            if filename.endswith(".jsonl"):
                records = read_jsonl(filepath)
                logger.info(f"Memuat {len(records)} baris data dari {filename} (JSONL)")
                all_records.extend(records)
            elif filename.endswith(".csv") and not filename.startswith("."):
                try:
                    df_csv = pd.read_csv(filepath)
                    csv_records = []
                    for _, row in df_csv.iterrows():
                        content = str(row.get("text", row.get("content", row.get("text_raw", ""))))
                        if not content or content.strip() == "" or content == "nan":
                            continue
                        plat = str(row.get("platform", "")).lower()
                        if not plat or plat == "nan":
                            if "youtube" in filename.lower():
                                plat = "youtube"
                            elif "twitter" in filename.lower():
                                plat = "twitter"
                            else:
                                plat = "unknown"

                        csv_records.append({
                            "id": str(row.get("id_anon", row.get("id", ""))),
                            "platform": plat,
                            "data_type": "comment" if "video_id_anon" in row or "parent_id" in row else "post",
                            "parent_id": str(row.get("video_id_anon", row.get("parent_id", ""))),
                            "author": str(row.get("id_anon", row.get("author", "anon"))),
                            "content": content,
                            "created_at": str(row.get("created_at", "")),
                            "url": str(row.get("url", "")),
                            "engagement": {
                                "likes": int(row.get("like_count", row.get("likes", 0)) or 0)
                            }
                        })
                    logger.info(f"Memuat {len(csv_records)} baris data dari {filename} (CSV)")
                    all_records.extend(csv_records)
                except Exception as e:
                    logger.warning(f"Gagal membaca data CSV {filename}: {e}")

        return all_records

    def process(
        self,
        export_csv: bool = True,
        export_excel: bool = True,
        export_sqlite: bool = True,
        output_prefix: str = "undiksha_dataset"
    ) -> Optional[pd.DataFrame]:
        """Menjalankan seluruh siklus pembersihan, transformasi, dan ekspor."""
        records = self.load_all_raw_data()
        if not records:
            logger.warning("Tidak ada data untuk diproses. Silakan jalankan scraper terlebih dahulu.")
            return None

        # Buat DataFrame
        rows = []
        spam_filtered_count = 0
        for r in records:
            raw_content = str(r.get("content", ""))
            
            # Lewati jika iklan komersial non-opini kampus (properti, kost, joki, dll)
            if self.cleaner.is_commercial_spam(raw_content):
                spam_filtered_count += 1
                continue

            display_content = self.cleaner.clean_display(raw_content)
            clean_content = self.cleaner.clean(raw_content)

            # Jika setelah dibersihkan teksnya tidak valid (kosong, cuma simbol, atau artifact UI), lewati
            if not self.cleaner.is_valid_text(clean_content):
                continue

            # Jika struktur bersarang engagement
            engagement = r.get("engagement", {})
            rows.append({
                "id": r.get("id"),
                "platform": r.get("platform"),
                "data_type": r.get("data_type"),
                "parent_id": r.get("parent_id"),
                "author": r.get("author"),
                "content": display_content,
                "clean_content": clean_content,
                "created_at": r.get("created_at"),
                "url": r.get("url"),
                "likes": engagement.get("likes", 0) if isinstance(engagement, dict) else r.get("likes", 0),
                "comments_count": engagement.get("comments_count", 0) if isinstance(engagement, dict) else r.get("comments_count", 0),
                "shares": engagement.get("shares", 0) if isinstance(engagement, dict) else r.get("shares", 0),
                "views": engagement.get("views", 0) if isinstance(engagement, dict) else r.get("views", 0),
                "hashtag_query": r.get("hashtag_query", "#undiksha"),
                "scraped_at": r.get("scraped_at"),
            })

        if spam_filtered_count > 0:
            logger.info(f"Penyaringan selesai: Memfilter keluar {spam_filtered_count} baris iklan/spam komersial non-opini.")

        df = pd.DataFrame(rows)

        # 1. Hapus Duplikasi berdasarkan ID dan Konten
        initial_len = len(df)
        if "id" in df.columns:
            has_id = df["id"].notna() & (df["id"].astype(str).str.strip() != "") & (df["id"] != "nan")
            df_with_id = df[has_id].drop_duplicates(subset=["id"], keep="first")
            df_no_id = df[~has_id]
            df = pd.concat([df_with_id, df_no_id], ignore_index=True)

        df.drop_duplicates(subset=["platform", "clean_content"], keep="first", inplace=True)
        logger.info(f"Deduplikasi selesai: {initial_len} -> {len(df)} baris (Dihapus {initial_len - len(df)} duplikat).")

        # 2. Filter baris dengan teks kosong atau tidak valid
        df = df[df["clean_content"].apply(self.cleaner.is_valid_text)]

        # 3. Ekspor Data
        csv_path = os.path.join(self.processed_dir, f"{output_prefix}.csv")
        excel_path = os.path.join(self.processed_dir, f"{output_prefix}.xlsx")
        db_path = os.path.join(self.processed_dir, f"{output_prefix}.db")

        if export_csv:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"Dataset CSV tersimpan di: {csv_path}")

        if export_excel:
            try:
                df.to_excel(excel_path, index=False, engine="openpyxl")
                logger.info(f"Dataset Excel tersimpan di: {excel_path}")
            except Exception as e:
                logger.warning(f"Gagal mengekspor Excel: {e}")

        if export_sqlite:
            try:
                conn = sqlite3.connect(db_path)
                df.to_sql("sentiment_data", conn, if_exists="replace", index=False)
                conn.close()
                logger.info(f"Database SQLite tersimpan di: {db_path}")
            except Exception as e:
                logger.warning(f"Gagal mengekspor SQLite: {e}")

        # Tampilkan Ringkasan Statistik
        self.print_summary(df)
        return df

    def print_summary(self, df: pd.DataFrame) -> None:
        """Menampilkan tabel ringkasan dataset menggunakan Rich Table."""
        table = Table(title="[bold cyan]=== Ringkasan Dataset Sentimen Undiksha ===[/bold cyan]")
        table.add_column("Platform", style="cyan", justify="center")
        table.add_column("Total Post", style="green", justify="right")
        table.add_column("Total Komentar", style="magenta", justify="right")
        table.add_column("Total Keseluruhan", style="bold yellow", justify="right")

        platforms = df["platform"].unique()
        for plat in sorted(platforms):
            plat_df = df[df["platform"] == plat]
            posts_count = len(plat_df[plat_df["data_type"] == "post"])
            cmts_count = len(plat_df[plat_df["data_type"] == "comment"])
            total = len(plat_df)
            table.add_row(plat.capitalize(), str(posts_count), str(cmts_count), str(total))

        # Baris Total
        tot_post = len(df[df["data_type"] == "post"])
        tot_cmt = len(df[df["data_type"] == "comment"])
        table.add_section()
        table.add_row("TOTAL SEMUA", str(tot_post), str(tot_cmt), str(len(df)), style="bold green")

        console.print(table)
