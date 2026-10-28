"""
src/collect/twitter_real.py
============================
Pengumpulan tweet REAL dari Twitter/X menggunakan twscrape.
twscrape menggunakan Twitter internal API via akun yang login.

Penggunaan:
    1. Tambahkan akun Twitter ke database:
       python src/collect/twitter_real.py --add-account
    2. Kumpulkan tweet keyword Undiksha:
       python src/collect/twitter_real.py --collect --count 500
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.collect.utils_collect import STANDARD_COLUMNS, save_raw
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

PLATFORM = "twitter"

# Keyword pencarian terkait Undiksha
SEARCH_KEYWORDS = [
    "undiksha",
    "universitas pendidikan ganesha",
    "#undiksha",
    "undiksha singaraja",
    "kampus undiksha",
    "pmb undiksha",
    "mahasiswa undiksha",
]


async def collect_tweets_async(
    keywords: list[str],
    n_per_keyword: int = 100,
    output_dir: str | Path = "data/raw",
    account_db: str = "data/raw/twitter_accounts.db",
) -> pd.DataFrame:
    """Mengumpulkan tweet real menggunakan twscrape.

    Args:
        keywords: Daftar keyword pencarian.
        n_per_keyword: Jumlah tweet per keyword.
        output_dir: Direktori output.
        account_db: Path ke database akun twscrape.

    Returns:
        DataFrame berformat STANDARD_COLUMNS.
    """
    try:
        from twscrape import API, gather
        from twscrape.logger import set_log_level
        set_log_level("ERROR")
    except ImportError:
        raise ImportError("twscrape belum diinstall. Jalankan: pip install twscrape")

    api = API(account_db)

    # Cek apakah ada akun yang login
    accounts = await api.pool.get_all()
    active_accounts = [a for a in accounts if a.active]
    if not active_accounts:
        raise RuntimeError(
            "Tidak ada akun Twitter aktif di twscrape!\n"
            "Jalankan: python src/collect/twitter_real.py --add-account\n"
            "Kemudian masukkan username, password, dan email akun Twitter kamu."
        )
    logger.info("Menggunakan %d akun Twitter aktif.", len(active_accounts))

    all_rows = []
    seen_ids: set[str] = set()

    for keyword in keywords:
        logger.info("Mengumpulkan tweet untuk keyword: '%s' (%d tweets)", keyword, n_per_keyword)
        query = keyword
        count = 0
        try:
            async for tweet in api.search(query, limit=n_per_keyword):
                # Lewati retweet
                if tweet.retweetedTweet is not None or (tweet.rawContent and tweet.rawContent.startswith("RT @")):
                    continue

                tweet_id = str(tweet.id)
                if tweet_id in seen_ids:
                    continue
                seen_ids.add(tweet_id)

                # Anonimkan ID
                id_anon = hashlib.sha256(tweet_id.encode()).hexdigest()[:16]

                row = {
                    "id_anon": id_anon,
                    "platform": PLATFORM,
                    "text": tweet.rawContent,
                    "created_at": tweet.date.isoformat() if tweet.date else None,
                    "is_official": False,
                    "is_spam_flag": False,
                    "url": f"https://twitter.com/i/web/status/{tweet_id}",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
                all_rows.append(row)
                count += 1

        except Exception as exc:
            logger.warning("Error saat scraping keyword '%s': %s", keyword, exc)

        logger.info("  -> Berhasil mengumpulkan %d tweet untuk '%s'", count, keyword)

    if not all_rows:
        logger.warning("Tidak ada tweet yang berhasil dikumpulkan.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(all_rows, columns=STANDARD_COLUMNS)
    df = df.drop_duplicates(subset=["id_anon"])

    saved_path = save_raw(df, platform=PLATFORM, raw_dir=output_dir)
    logger.info("Total %d tweet unik disimpan ke %s", len(df), saved_path)
    return df


async def add_account_async(account_db: str = "data/raw/twitter_accounts.db") -> None:
    """Menambahkan akun Twitter ke pool twscrape secara interaktif."""
    try:
        from twscrape import API
    except ImportError:
        raise ImportError("twscrape belum diinstall. Jalankan: pip install twscrape")

    api = API(account_db)

    print("\n" + "=" * 55)
    print("  Tambah Akun Twitter ke twscrape Pool")
    print("=" * 55)
    print("  Akun ini digunakan untuk scraping tweet publik.")
    print("  Gunakan akun yang TIDAK penting (buat akun baru).")
    print("=" * 55)

    username = input("\nMasukkan username Twitter (tanpa @): ").strip()
    password = input("Masukkan password: ").strip()
    email = input("Masukkan email terdaftar: ").strip()
    email_password = input("Masukkan password email (opsional, tekan Enter skip): ").strip() or None

    await api.pool.add_account(
        username=username,
        password=password,
        email=email,
        email_password=email_password,
    )

    print("\nLogin ke Twitter...")
    await api.pool.login_all()

    accounts = await api.pool.get_all()
    active = [a for a in accounts if a.active]
    print(f"\nStatus: {len(active)}/{len(accounts)} akun aktif.")
    if active:
        print(f"Akun berhasil ditambahkan: @{username}")
    else:
        print("Login gagal. Periksa kembali username/password/email.")


async def add_account_cookie_async(
    account_db: str = "data/raw/twitter_accounts.db",
) -> None:
    """Menambahkan akun Twitter menggunakan Cookie browser (Bypass Cloudflare)."""
    try:
        from twscrape import API
    except ImportError:
        raise ImportError("twscrape belum diinstall. Jalankan: pip install twscrape")

    api = API(account_db)

    print("\n" + "=" * 55)
    print("  Tambah Akun Twitter via Cookie (Bypass Cloudflare)")
    print("=" * 55)
    print("  Cara mendapatkan Cookie:")
    print("  1. Buka https://x.com di browser dan pastikan sudah login.")
    print("  2. Tekan F12 -> tab 'Application' (atau 'Storage') -> 'Cookies' -> 'https://x.com'.")
    print("  3. Salin nilai cookie 'auth_token' dan 'ct0'.")
    print("=" * 55)

    username = input("\nMasukkan username Twitter: ").strip().lstrip("@")
    auth_token = input("Masukkan nilai auth_token: ").strip()
    ct0 = input("Masukkan nilai ct0: ").strip()

    if not auth_token or not ct0:
        print("\n[ERROR] auth_token dan ct0 wajib diisi!")
        return

    cookie_str = f"auth_token={auth_token}; ct0={ct0}"

    try:
        await api.pool.add_account_cookies(username=username, cookies=cookie_str)
    except Exception as exc:
        print(f"\n[ERROR] Gagal menambahkan cookies: {exc}")
        return

    accounts = await api.pool.get_all()
    active = [a for a in accounts if a.active and a.username.lower() == username.lower()]
    if active:
        print(f"\n[SUKSES] Akun @{username} berhasil ditambahkan dan AKTIF!")
        print("Sekarang kamu bisa menjalankan: python src/collect/twitter_real.py --collect --count 200")
    else:
        print("\n[GAGAL] Akun belum aktif. Periksa apakah nilai cookie auth_token dan ct0 sudah benar.")


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Pengumpulan Tweet Real Undiksha via twscrape.")
    parser.add_argument("--add-account", action="store_true", help="Tambah akun Twitter via username/password.")
    parser.add_argument("--add-cookie", action="store_true", help="Tambah akun Twitter via Cookie browser (Bypass Cloudflare).")
    parser.add_argument("--collect", action="store_true", help="Kumpulkan tweet real Undiksha.")
    parser.add_argument("--count", type=int, default=150, help="Jumlah tweet per keyword (default: 150).")
    parser.add_argument("--output-dir", default="data/raw", help="Direktori output data mentah.")
    parser.add_argument("--account-db", default="data/raw/twitter_accounts.db", help="Path database akun twscrape.")
    args = parser.parse_args()

    if args.add_account:
        asyncio.run(add_account_async(account_db=args.account_db))
    elif args.add_cookie:
        asyncio.run(add_account_cookie_async(account_db=args.account_db))
    elif args.collect:
        df = asyncio.run(collect_tweets_async(
            keywords=SEARCH_KEYWORDS,
            n_per_keyword=args.count,
            output_dir=args.output_dir,
            account_db=args.account_db,
        ))
        print(f"\nTotal tweet real terkumpul: {len(df)}")
        for i, r in df.head(5).iterrows():
            safe_text = str(r['text']).encode('ascii', errors='replace').decode('ascii')
            print(f"[{i+1}] {r['created_at']} | {safe_text[:80]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
