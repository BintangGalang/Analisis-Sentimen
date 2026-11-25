"""
scripts/convert_csv_to_jsonl.py
================================
Menyeragamkan data mentah Twitter/X dan YouTube dari format CSV menjadi JSONL
agar seragam dengan 4 platform lainnya (Instagram, TikTok, Threads, Facebook).
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def convert_youtube():
    print("=== Mengonversi YouTube CSV -> JSONL ===")
    yt_files = [
        "youtube_raw_20261025.csv",
        "youtube_raw_20261026.csv",
        "youtube_raw_20261123.csv",
        "youtube_negative_targeted.csv",
    ]
    
    seen_ids = set()
    seen_texts = set()
    yt_records = []
    
    for fname in yt_files:
        fpath = RAW_DIR / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            print(f"  Memuat {fname}: {len(df)} baris")
            for _, r in df.iterrows():
                content = str(r.get("text", r.get("content", ""))).strip()
                if not content or content == "nan":
                    continue
                
                content_lower = content.lower()
                row_id = str(r.get("id_anon", r.get("id", ""))).strip()
                if not row_id or row_id == "nan":
                    row_id = f"yt_{len(yt_records)}"
                
                if row_id in seen_ids or content_lower in seen_texts:
                    continue
                
                seen_ids.add(row_id)
                seen_texts.add(content_lower)
                
                try:
                    likes = int(float(r.get("like_count", r.get("likes", 0)) or 0))
                except (ValueError, TypeError):
                    likes = 0
                
                rec = {
                    "id": row_id,
                    "platform": "youtube",
                    "data_type": "comment",
                    "parent_id": str(r.get("video_id_anon", r.get("parent_id", ""))) if pd.notna(r.get("video_id_anon", r.get("parent_id"))) else None,
                    "author": str(r.get("author", row_id[:16])),
                    "content": content,
                    "clean_content": None,
                    "created_at": str(r.get("created_at", "")) if pd.notna(r.get("created_at")) else None,
                    "url": str(r.get("url", "")) if pd.notna(r.get("url")) else None,
                    "engagement": {
                        "likes": likes,
                        "comments_count": 0,
                        "shares": 0,
                        "views": 0,
                    },
                    "hashtag_query": "#undiksha",
                    "scraped_at": str(r.get("collected_at", r.get("scraped_at", ""))) if pd.notna(r.get("collected_at", r.get("scraped_at"))) else None,
                    "metadata": {
                        "is_official": bool(r.get("is_official", False)),
                        "is_spam_flag": bool(r.get("is_spam_flag", False)),
                    }
                }
                yt_records.append(rec)
        except Exception as e:
            print(f"  [ERROR] Gagal memproses {fname}: {e}")
            
    out_yt = RAW_DIR / "youtube_raw.jsonl"
    with open(out_yt, "w", encoding="utf-8") as f:
        for rec in yt_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"--> Selesai! {len(yt_records)} baris YouTube unik disimpan ke: {out_yt}\n")


def convert_twitter():
    print("=== Mengonversi Twitter/X CSV -> JSONL ===")
    tw_files = [
        "twitter_raw_20261026.csv",
        "twitter_raw_20261123.csv",
    ]
    
    seen_ids = set()
    seen_texts = set()
    tw_records = []
    
    for fname in tw_files:
        fpath = RAW_DIR / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            print(f"  Memuat {fname}: {len(df)} baris")
            for _, r in df.iterrows():
                content = str(r.get("text", r.get("content", ""))).strip()
                if not content or content == "nan":
                    continue
                
                content_lower = content.lower()
                row_id = str(r.get("id_anon", r.get("id", ""))).strip()
                if not row_id or row_id == "nan":
                    row_id = f"tw_{len(tw_records)}"
                
                if row_id in seen_ids or content_lower in seen_texts:
                    continue
                
                seen_ids.add(row_id)
                seen_texts.add(content_lower)
                
                try:
                    likes = int(float(r.get("like_count", r.get("likes", 0)) or 0))
                except (ValueError, TypeError):
                    likes = 0
                
                rec = {
                    "id": row_id,
                    "platform": "twitter",
                    "data_type": "post",
                    "parent_id": None,
                    "author": str(r.get("author", row_id[:16])),
                    "content": content,
                    "clean_content": None,
                    "created_at": str(r.get("created_at", "")) if pd.notna(r.get("created_at")) else None,
                    "url": str(r.get("url", "")) if pd.notna(r.get("url")) else None,
                    "engagement": {
                        "likes": likes,
                        "comments_count": 0,
                        "shares": 0,
                        "views": 0,
                    },
                    "hashtag_query": "#undiksha",
                    "scraped_at": str(r.get("collected_at", r.get("scraped_at", ""))) if pd.notna(r.get("collected_at", r.get("scraped_at"))) else None,
                    "metadata": {
                        "is_official": bool(r.get("is_official", False)),
                        "is_spam_flag": bool(r.get("is_spam_flag", False)),
                    }
                }
                tw_records.append(rec)
        except Exception as e:
            print(f"  [ERROR] Gagal memproses {fname}: {e}")
            
    out_tw = RAW_DIR / "twitter_raw.jsonl"
    with open(out_tw, "w", encoding="utf-8") as f:
        for rec in tw_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"--> Selesai! {len(tw_records)} baris Twitter unik disimpan ke: {out_tw}\n")


if __name__ == "__main__":
    convert_youtube()
    convert_twitter()

