"""
scripts/collect_twitter_1500.py
===============================
Skrip terpadu untuk mengumpulkan data tweet X/Twitter hingga mencapai minimal 1.500 baris data mentah.
Menggunakan twscrape yang terhubung dengan akun aktif di data/raw/twitter_accounts.db.
Memprioritaskan user_tweets (@undiksha & @BEM_Undiksha) yang memiliki ratusan tweet riil
dan segera menyimpan data secara inkremental ke file CSV.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from twscrape import API
from twscrape.logger import set_log_level

# Atur logging twscrape
set_log_level("INFO")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect_twitter")

RAW_CSV_PATH = Path("data/raw/twitter_raw_20261026.csv")
BACKUP_PATH = Path("data/raw/twitter_raw_20261026_bak.csv")
ACCOUNT_DB_PATH = "data/raw/twitter_accounts.db"
TARGET_TOTAL = 1550  # Target minimal 1500, kita set 1550 sebagai safety margin

USER_ACCOUNTS = [
    (986940060, "undiksha", True),
    (1285233610002333698, "BEM_Undiksha", False),
]

KEYWORDS = [
    "undiksha",
    "@undiksha",
    "dosen undiksha",
    "mahasiswa undiksha",
    "rektor undiksha",
    "siakad undiksha",
    "ukt undiksha",
    "kkn undiksha",
    "wisuda undiksha",
    "snbp undiksha",
    "snbt undiksha",
    "fip undiksha",
    "fmipa undiksha",
    "fbs undiksha",
    "ftk undiksha",
    "fe undiksha",
    "fk undiksha",
    "fok undiksha",
    "fhis undiksha",
    "undiksha bali",
    "bem undiksha",
    "pkkmb undiksha",
    "kuliah undiksha",
    "perpustakaan undiksha",
]


async def wait_until_queue_available(api: API, queue_name: str) -> None:
    """Menunggu hingga antrean twscrape tidak terkunci."""
    while True:
        accs = await api.pool.get_all()
        active_accs = [a for a in accs if a.active]
        if not active_accs:
            logger.error("Tidak ada akun aktif di pool!")
            return

        acc = active_accs[0]
        lock_dt = acc.locks.get(queue_name)
        if not lock_dt:
            logger.info("Antrean '%s' siap digunakan.", queue_name)
            return

        now_utc = datetime.now(timezone.utc)
        if now_utc >= lock_dt:
            logger.info("Kunci '%s' telah kedaluwarsa (%s). Siap digunakan.", queue_name, lock_dt)
            await api.pool.unlock(acc.username, queue_name)
            return

        wait_seconds = int((lock_dt - now_utc).total_seconds()) + 2
        logger.info(
            "Antrean '%s' terkunci hingga %s UTC. Menunggu %d detik...",
            queue_name,
            lock_dt.strftime("%H:%M:%S"),
            wait_seconds,
        )
        await asyncio.sleep(min(wait_seconds, 10))


def save_merged(df_old: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """Menggabungkan dan menyimpan ke CSV secara aman."""
    if not new_rows:
        return df_old
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["id_anon"])
    df_combined = df_combined.drop_duplicates(subset=["url"])
    df_combined.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info(
        "--> [DISK SAVE] Tersimpan di %s: total %d baris (lama: %d, baru unik: %d)",
        RAW_CSV_PATH,
        len(df_combined),
        len(df_old),
        len(df_combined) - len(df_old),
    )
    return df_combined


async def main():
    if not RAW_CSV_PATH.exists():
        logger.error("File mentah tidak ditemukan: %s", RAW_CSV_PATH)
        return

    df_current = pd.read_csv(RAW_CSV_PATH)
    logger.info("Memuat data mentah saat ini: %d baris", len(df_current))

    # Backup jika belum ada
    if not BACKUP_PATH.exists():
        df_current.to_csv(BACKUP_PATH, index=False, encoding="utf-8-sig")
        logger.info("Backup data awal disimpan di %s", BACKUP_PATH)

    seen_ids = set()
    for _, row in df_current.iterrows():
        url_str = str(row.get("url", ""))
        if "status/" in url_str:
            tid = url_str.split("status/")[-1].strip()
            seen_ids.add(tid)
        if pd.notna(row.get("id_anon")):
            seen_ids.add(str(row["id_anon"]))

    seen_texts = set(df_current["text"].dropna().astype(str).str.strip().str.lower())
    logger.info("Total ID/URL unik tersimpan: %d, teks unik: %d", len(seen_ids), len(seen_texts))

    if len(df_current) >= TARGET_TOTAL:
        logger.info("Data mentah sudah mencapai %d (target >= 1500). Selesai!", len(df_current))
        return

    logger.info("Target: minimal %d data. Masih perlu: %d data.", TARGET_TOTAL, TARGET_TOTAL - len(df_current))

    api = API(ACCOUNT_DB_PATH)
    new_rows = []

    # TAHAP 1: Ambil dari user_tweets (Prioritas utama karena akun @undiksha punya 2200+ tweet)
    logger.info("=== TAHAP 1: Scraping Timeline Akun Resmi/Terkait Undiksha ===")
    for uid, handle, is_off in USER_ACCOUNTS:
        if len(df_current) + len(new_rows) >= TARGET_TOTAL:
            break

        logger.info("Menunggu antrean 'UserTweets' untuk @%s...", handle)
        await wait_until_queue_available(api, "UserTweets")

        logger.info("Mengambil timeline dari @%s (ID: %d)...", handle, uid)
        u_count = 0
        try:
            async for tweet in api.user_tweets(uid, limit=800):
                if tweet.retweetedTweet is not None or (tweet.rawContent and tweet.rawContent.startswith("RT @")):
                    continue

                tweet_id = str(tweet.id)
                if tweet_id in seen_ids:
                    continue

                text_content = str(tweet.rawContent).strip()
                text_lower = text_content.lower()
                if text_lower in seen_texts:
                    continue

                seen_ids.add(tweet_id)
                seen_texts.add(text_lower)
                u_count += 1

                id_anon = hashlib.sha256(tweet_id.encode()).hexdigest()[:16]
                row = {
                    "id_anon": id_anon,
                    "platform": "twitter",
                    "text": text_content,
                    "created_at": tweet.date.isoformat() if tweet.date else None,
                    "is_official": is_off,
                    "is_spam_flag": False,
                    "url": f"https://twitter.com/i/web/status/{tweet_id}",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
                new_rows.append(row)

                # Simpan inkremental tiap 100 baris
                if len(new_rows) % 100 == 0:
                    df_current = save_merged(df_current, new_rows)
                    new_rows = []

                if len(df_current) + len(new_rows) >= TARGET_TOTAL:
                    logger.info("Target minimal >= 1500 telah tercapai via user_tweets!")
                    break

        except Exception as exc:
            logger.warning("Error saat mengambil user_tweets untuk @%s: %s", handle, exc)

        logger.info("Berhasil mengambil %d tweet baru dari @%s.", u_count, handle)
        df_current = save_merged(df_current, new_rows)
        new_rows = []

    # TAHAP 2: Jika masih belum cukup, ambil dari SearchTimeline
    if len(df_current) < TARGET_TOTAL:
        logger.info("=== TAHAP 2: Scraping Keyword SearchTimeline ===")
        for kw in KEYWORDS:
            if len(df_current) + len(new_rows) >= TARGET_TOTAL:
                break

            await wait_until_queue_available(api, "SearchTimeline")

            logger.info("Mencari keyword: '%s'...", kw)
            kw_new = 0
            try:
                async for tweet in api.search(kw, limit=100):
                    if tweet.retweetedTweet is not None or (tweet.rawContent and tweet.rawContent.startswith("RT @")):
                        continue

                    tweet_id = str(tweet.id)
                    if tweet_id in seen_ids:
                        continue

                    text_content = str(tweet.rawContent).strip()
                    text_lower = text_content.lower()
                    if text_lower in seen_texts:
                        continue

                    seen_ids.add(tweet_id)
                    seen_texts.add(text_lower)
                    kw_new += 1

                    id_anon = hashlib.sha256(tweet_id.encode()).hexdigest()[:16]
                    row = {
                        "id_anon": id_anon,
                        "platform": "twitter",
                        "text": text_content,
                        "created_at": tweet.date.isoformat() if tweet.date else None,
                        "is_official": False,
                        "is_spam_flag": False,
                        "url": f"https://twitter.com/i/web/status/{tweet_id}",
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                    new_rows.append(row)

                    if len(new_rows) % 50 == 0:
                        df_current = save_merged(df_current, new_rows)
                        new_rows = []

                    if len(df_current) + len(new_rows) >= TARGET_TOTAL:
                        break

            except Exception as exc:
                logger.warning("Error saat scraping '%s': %s", kw, exc)

            logger.info("  Keyword '%s': %d tweet baru.", kw, kw_new)
            df_current = save_merged(df_current, new_rows)
            new_rows = []
            await asyncio.sleep(1.0)

    # Simpan sisa baris
    df_current = save_merged(df_current, new_rows)

    logger.info("=" * 60)
    logger.info("PROSES KOLEKSI TWITTER SELESAI!")
    logger.info("Total Data Akhir di %s: %d baris", RAW_CSV_PATH, len(df_current))
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
