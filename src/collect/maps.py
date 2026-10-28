"""
src/collect/maps.py
===================
Pengumpulan ulasan REAL dari Google Maps Review terkait
Universitas Pendidikan Ganesha (Undiksha).

Mendukung 2 metode:
1. **Scraper Otomatis** (`--collect`):
   Menggunakan Playwright untuk mengambil ulasan dari berbagai lokasi kampus Undiksha
   (Kampus Pusat Singaraja, Kampus Jinengdalem, Kampus Denpasar, dan FK Undiksha).
2. **Import Ekspor** (`--export-file`):
   Memuat file CSV/JSON/JSONL ulasan Google Maps dari hasil ekspor browser extension
   atau Google Takeout.

Penggunaan:
    # Mengambil ulasan secara otomatis:
    python src/collect/maps.py --collect

    # Mengimpor dari file ekspor:
    python src/collect/maps.py --export-file data/raw/maps_export.csv
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.collect.utils_collect import (
    STANDARD_COLUMNS,
    build_standard_row,
    save_raw,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

PLATFORM = "maps"

# Lokasi Google Maps resmi Undiksha
UNDIKSHA_MAPS_PLACES = [
    {
        "name": "Universitas Pendidikan Ganesha (Kampus Pusat Singaraja)",
        "url": "https://www.google.co.id/maps/place/Universitas+Pendidikan+Ganesha/@-8.116532,115.0877326,15z/data=!4m6!3m5!1s0x2dd21665cef49255:0x1abe2912c8d6af5a!8m2!3d-8.116532!4d115.0877326!16s%2Fm%2F0g57p_d?hl=id",
    },
    {
        "name": "Kampus Undiksha Jinengdalem",
        "url": "https://www.google.co.id/maps/place/Kampus+Undiksha+Jinengdalem/@-8.1384261,115.1164925,17z/data=!4m6!3m5!1s0x2dd19a008c2a41d9:0x8e5f2cf69c735d48!8m2!3d-8.1384261!4d115.1164925!16s%2Fg%2F11b6cr4j21?hl=id",
    },
    {
        "name": "Fakultas Kedokteran Undiksha",
        "url": "https://www.google.co.id/maps/place/Fakultas+Kedokteran+Undiksha/@-8.1388487,115.1166408,17z/data=!4m6!3m5!1s0x2dd19a00b65f7c37:0xefb0981bdf9ecf01!8m2!3d-8.1388487!4d115.1166408!16s%2Fg%2F11hbtl9n42?hl=id",
    },
    {
        "name": "Kampus Undiksha Denpasar",
        "url": "https://www.google.co.id/maps/place/Universitas+Pendidikan+Ganesha+(Undiksha)+Denpasar/@-8.6657271,115.2289139,17z/data=!4m6!3m5!1s0x2dd240974b94f923:0xa95e1c07cfbe0772!8m2!3d-8.6657271!4d115.2289139!16s%2Fg%2F11b6crq7vf?hl=id",
    },
]


def load_from_export(
    export_path: str | Path,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Memuat data ulasan Google Maps dari file ekspor CSV/JSON/JSONL."""
    export_path = Path(export_path)
    logger.info("Memuat ulasan Google Maps dari ekspor: %s", export_path)

    if not export_path.exists():
        raise FileNotFoundError(f"File ekspor tidak ditemukan: {export_path}")

    suffix = export_path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        raw_df = pd.read_csv(export_path, sep=sep, dtype=str, encoding="utf-8-sig")
    elif suffix == ".json":
        raw_df = pd.read_json(export_path, dtype=str)
    elif suffix == ".jsonl":
        raw_df = pd.read_json(export_path, lines=True, dtype=str)
    else:
        raise ValueError(f"Format {suffix} tidak didukung.")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for i, rec in raw_df.iterrows():
        raw_text = (
            rec.get("text")
            or rec.get("review")
            or rec.get("ulasan")
            or rec.get("content")
            or rec.get("snippet")
            or ""
        )
        text = str(raw_text).strip()
        if not text or text.lower() == "nan" or len(text) < 3:
            continue

        text = text.encode("utf-8", errors="replace").decode("utf-8")

        username = str(rec.get("author") or rec.get("username") or rec.get("name") or f"maps_user_{i}")
        created_at = (
            rec.get("created_at")
            or rec.get("date")
            or rec.get("timestamp")
            or datetime.now(timezone.utc).isoformat()
        )
        raw_id = str(rec.get("id") or rec.get("review_id") or f"gmaps_{i}")
        url = str(rec.get("url") or "https://maps.google.com/?cid=undiksha")

        id_anon = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        if id_anon in seen_ids:
            continue
        seen_ids.add(id_anon)

        row = build_standard_row(
            raw_id=raw_id,
            platform=PLATFORM,
            text=text,
            created_at=created_at,
            username=username,
            url=url,
        )
        rows.append(row)

    if not rows:
        logger.warning("Tidak ada ulasan valid yang diekstrak dari file ekspor.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id_anon", "text"]).reset_index(drop=True)
    saved_path = save_raw(df, platform=PLATFORM, raw_dir=raw_dir)
    logger.info("Total %d ulasan Google Maps disimpan ke %s", len(df), saved_path)
    return df


async def collect_maps_reviews_async(
    output_dir: str | Path = "data/raw",
    max_reviews_per_place: int = 150,
) -> pd.DataFrame:
    """Mengumpulkan ulasan Google Maps secara langsung menggunakan Playwright."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError("Playwright belum diinstall. Jalankan: pip install playwright && playwright install chromium")

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="id-ID",
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        for place in UNDIKSHA_MAPS_PLACES:
            p_name = place["name"]
            p_url = place["url"]
            logger.info("Mengambil ulasan untuk: %s", p_name)

            try:
                await page.goto(p_url, wait_until="domcontentloaded", timeout=25000)
                await page.wait_for_timeout(4000)

                # Klik tab Ulasan jika tersedia
                tab = page.locator("button").filter(has_text="Ulasan").first
                if await tab.count() > 0:
                    await tab.click()
                    await page.wait_for_timeout(3000)

                # Ekstrak kartu ulasan
                rev_elements = page.locator("div.jftiEf")
                card_count = await rev_elements.count()
                logger.info("  Ditemukan %d ulasan di halaman untuk %s", card_count, p_name)

                for i in range(card_count):
                    el = rev_elements.nth(i)
                    author = await el.locator(".d4r55").inner_text() if await el.locator(".d4r55").count() > 0 else "anon"
                    text = await el.locator(".wiI7pd").inner_text() if await el.locator(".wiI7pd").count() > 0 else ""
                    date_str = await el.locator(".rsqaWe").inner_text() if await el.locator(".rsqaWe").count() > 0 else ""

                    if not text.strip() or len(text.strip()) < 3:
                        continue

                    raw_id = f"gmaps_{p_name}_{author}_{i}"
                    id_anon = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
                    if id_anon in seen_ids:
                        continue
                    seen_ids.add(id_anon)

                    row = build_standard_row(
                        raw_id=raw_id,
                        platform=PLATFORM,
                        text=text.strip(),
                        created_at=datetime.now(timezone.utc).isoformat(),
                        username=author.strip(),
                        url=p_url,
                    )
                    all_rows.append(row)

            except Exception as exc:
                logger.warning("  Gagal memproses %s: %s", p_name, exc)

        await browser.close()

    if not all_rows:
        logger.warning("Tidak ada ulasan yang terkumpul dari scraper browser.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(all_rows, columns=STANDARD_COLUMNS)
    df = df.drop_duplicates(subset=["id_anon", "text"])
    saved_path = save_raw(df, platform=PLATFORM, raw_dir=output_dir)
    logger.info("Total %d ulasan Google Maps disimpan ke %s", len(df), saved_path)
    return df


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Pengumpulan Data Ulasan Google Maps Undiksha.")
    parser.add_argument("--collect", action="store_true", help="Kumpulkan ulasan Google Maps via Playwright.")
    parser.add_argument("--export-file", type=str, default=None, help="Muat data dari file ekspor (CSV/JSON/JSONL).")
    parser.add_argument("--output-dir", default="data/raw", help="Direktori output.")
    args = parser.parse_args()

    if args.export_file:
        print("\n" + "=" * 55)
        print("  Memuat Ulasan Google Maps dari File Ekspor...")
        print("=" * 55)
        df = load_from_export(args.export_file, raw_dir=args.output_dir)
        print("\n" + "=" * 55)
        print(f"  HASIL IMPORT MAPS: {len(df)} ulasan berhasil disimpan.")
        print("=" * 55)

    elif args.collect:
        print("\n" + "=" * 55)
        print("  Mulai Pengumpulan Ulasan Google Maps Undiksha...")
        print("=" * 55)
        df = asyncio.run(collect_maps_reviews_async(output_dir=args.output_dir))
        print("\n" + "=" * 55)
        print("  HASIL PENGUMPULAN DATA GOOGLE MAPS")
        print("=" * 55)
        print(f"  Total ulasan terkumpul : {len(df)}")
        if not df.empty:
            print("\n  Contoh ulasan:")
            for i, r in df.head(5).iterrows():
                safe = str(r["text"]).encode("ascii", errors="replace").decode("ascii")
                print(f"  [{i+1}] {safe[:70]}")
        print("=" * 55)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
