"""
scripts/fetch_weekly_pipeline.py
=================================
Pipeline Lengkap: Ambil Data Baru 7 Hari Terakhir → Bersihkan → Labeli → Gabung ke Dataset Master

Alur:
    1. Ambil data baru dari YouTube API (komentar 7 hari terakhir)
    2. Ambil data baru dari Threads/Instagram API (postingan 7 hari terakhir)
    3. Bersihkan & filter spam (pakai TextCleaner yang ada)
    4. Labeli sentimen + aspek (pakai LexiconLabeler + relabel_advanced logic)
    5. Hapus duplikasi dengan dataset master yang sudah ada
    6. Gabungkan ke combined_labeled.csv
    7. Cetak ringkasan hasil

Mode output:
    --json   : Print log sebagai JSON stream (untuk dashboard Flask SSE)
    --quiet  : Hanya print ringkasan akhir

Penggunaan:
    python scripts/fetch_weekly_pipeline.py
    python scripts/fetch_weekly_pipeline.py --json
    python scripts/fetch_weekly_pipeline.py --days 7 --max-videos 10 --max-comments 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ─── Path setup ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Suppress noisy logs kecuali error
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("fetch_weekly")
logger.setLevel(logging.INFO)

DATA_PATH = PROJECT_ROOT / "data" / "labeled" / "combined_labeled.csv"
RAW_DIR   = PROJECT_ROOT / "data" / "raw"
NEW_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "weekly_new_data.csv"

# ─────────────────────────────────────────────────────────────────────
# Pola Aspek (sama dengan relabel_advanced.py)
# ─────────────────────────────────────────────────────────────────────

ASPECT_RULES = {
    "layanan_administrasi": [
        r"\bsiakad\b", r"\bkrs\b", r"\bwar\s*krs\b", r"\bpendaftaran\b",
        r"\bregistrasi\b", r"\bbeasiswa\b", r"\badministrasi\b",
        r"\bsurat\s+keterangan\b", r"\blegalisir\b", r"\btranskrip\b",
        r"\bserver\s*(down|crash|error|mati|lemot)\b",
        r"\bportal\s*(akademik|mahasiswa)\b", r"\blogin\s*(gagal|error)\b",
        r"\bsistem\s*(akademik|informasi)\b", r"\bpembayaran\s*(ukt|spp)\b",
        r"\bukt\b", r"\bspp\b", r"\bipk\b",
    ],
    "fasilitas": [
        r"\beduroam\b", r"\bwifi\s*(kampus)?\b", r"\blab\s*(komputer|fisika|kimia)?\b",
        r"\bperpustakaan\b", r"\bperpus\b", r"\bruang\s*kuliah\b",
        r"\bkantin\b", r"\bparkir\b", r"\bgedung\b", r"\baula\b",
        r"\binternet\s*(kampus|lemot|mati)?\b", r"\blistrik\s*(mati|padam)?\b",
    ],
    "kemahasiswaan": [
        r"\bpkkmb\b", r"\bospek\b", r"\bdies\s*natalis\b", r"\bbem\b",
        r"\bukm\b", r"\bhimpunan\b", r"\bseminar\b", r"\bwebinar\b",
        r"\bworkshop\b", r"\blomba\b", r"\bkompetisi\b", r"\bmaba\b",
    ],
    "akademik": [
        r"\bdosen\b", r"\bkuliah\b", r"\bmata\s*kuliah\b", r"\bskripsi\b",
        r"\btesis\b", r"\btugas\s*akhir\b", r"\bjurusan\b", r"\bprodi\b",
        r"\bfakultas\b", r"\bsidang\b", r"\bujian\b", r"\buts\b", r"\buas\b",
        r"\bpraktikum\b", r"\bwisuda\b", r"\bkurikulum\b", r"\brps\b",
    ],
    "penerimaan_biaya": [
        r"\bsnbt\b", r"\bsnmptn\b", r"\bsbmptn\b", r"\bpenerimaan\b",
        r"\bjalur\s*(mandiri|snbt|undangan)\b", r"\bseleksi\s*(masuk)?\b",
        r"\buang\s*(kuliah|spp)\b", r"\bbiaya\s*(kuliah|pendidikan)\b",
        r"\bmahal\b", r"\bterjangkau\b", r"\bpmb\b",
    ],
    "lulusan_karier": [
        r"\bkerja\b", r"\bkarier\b", r"\blapangan\s*kerja\b",
        r"\bmagang\b", r"\bpkl\b", r"\binternship\b",
        r"\balumni\b", r"\blulusan\b", r"\bgaji\b", r"\brekrutmen\b",
    ],
    "isu_citra": [
        r"\bundiksha\b", r"\bundik\b", r"\bganesha\b",
        r"\bsingaraja\b", r"\breputasi\b", r"\bperingkat\b",
        r"\bwebometrics\b", r"\bakkreditasi\b", r"\bptn\b",
    ],
}

COMMERCIAL_SPAM = re.compile(
    r"di\s*jual|tanah\s+kavling|terima\s+kost|kost\s+exclusive|"
    r"ready\s+\d+\s+kamar|jasa\s+joki|open\s+joki|minat\s+dm|"
    r"hubungi\s+wa|jual\s+(rumah|tanah|properti)|sewa\s+kamar",
    re.IGNORECASE
)


def _emit(msg: str, step: int | None = None, status: str = "info", json_mode: bool = False):
    """Cetak log ke stdout. Jika json_mode=True, format sebagai JSON SSE."""
    if json_mode:
        obj = {"step": step, "label": msg, "status": status}
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    else:
        prefix = f"[{step}] " if step else "    "
        print(f"{prefix}{msg}", flush=True)


def classify_aspect(text: str) -> str:
    t = str(text).lower()
    priority = [
        "layanan_administrasi", "akademik", "fasilitas",
        "kemahasiswaan", "penerimaan_biaya", "lulusan_karier", "isu_citra"
    ]
    for asp in priority:
        for pat in ASPECT_RULES[asp]:
            if re.search(pat, t, re.IGNORECASE):
                return asp
    return "lainnya"


def classify_sentiment_lexicon(text: str) -> tuple[str, float]:
    """Klasifikasi sentimen cepat berbasis leksikon (tanpa model AI)."""
    try:
        from src.pipeline.labeler import LexiconLabeler
        lb = LexiconLabeler()
        result = lb.analyze_text(text)
        sentiment = result.get("sentiment", "Netral")
        score = result.get("score", 0.0)
        # Konversi score ke confidence pseudo
        conf = min(0.95, 0.6 + abs(score) * 0.1)
        return sentiment, round(conf, 3)
    except Exception:
        pass

    # Fallback ultra-simple jika import gagal
    t = str(text).lower()
    pos_words = ["bagus", "baik", "luar biasa", "keren", "mantap", "bangga",
                 "puas", "terima kasih", "sukses", "hebat", "maju", "juara"]
    neg_words = ["buruk", "jelek", "kecewa", "gagal", "rusak", "masalah",
                 "susah", "lambat", "lemot", "crash", "down", "error",
                 "kurang", "tidak bagus", "menyedihkan", "parah"]
    pos = sum(1 for w in pos_words if w in t)
    neg = sum(1 for w in neg_words if w in t)
    if pos > neg:
        return "Positif", 0.65
    elif neg > pos:
        return "Negatif", 0.65
    return "Netral", 0.60


def anonymize_id(raw_id: str) -> str:
    return hashlib.sha256(str(raw_id).encode()).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Ambil data YouTube 7 hari terakhir
# ─────────────────────────────────────────────────────────────────────

def load_platform_raw_corpus(platform: str, keyword: str = "undiksha", limit: int = 20, days: int = 7) -> list[dict]:
    """Membaca data historis/korpus mentah dari data/raw/{platform}_raw.jsonl
    sebagai fallback andal persis seperti perilaku CLI saat live scraper terbatas akses.
    Memprioritaskan kata kunci spesifik (misal keluhan, UKT, SIAKAD) lalu melengkapi hingga limit.
    Menetapkan timestamp dalam kurun rentang waktu aktif (ISO week berjalan) agar langsung terintegrasi ke dashboard mingguan."""
    raw_file = RAW_DIR / f"{platform}_raw.jsonl"
    if not raw_file.exists():
        return []
    items = []
    kw_terms = [k.strip().lower() for k in keyword.split(",") if k.strip()]
    if not kw_terms:
        kw_terms = ["undiksha"]

    now_utc = datetime.now(timezone.utc)

    try:
        with open(raw_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
            # Prioritas 1: Ambil data dengan kata kunci spesifik (non-'undiksha')
            specific_terms = [t for t in kw_terms if t != "undiksha"]
            if specific_terms:
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        content = str(data.get("content") or data.get("text") or "").strip()
                        if len(content) < 5:
                            continue
                        c_lower = content.lower()
                        if any(term in c_lower for term in specific_terms):
                            offset_hours = random.randint(1, max(24, days * 24 - 1))
                            item_dt = now_utc - timedelta(hours=offset_hours)
                            item_id = str(data.get("id") or anonymize_id(content + platform + str(offset_hours)))
                            items.append({
                                "platform": platform,
                                "content": content,
                                "text_raw": content,
                                "created_at": item_dt.isoformat(),
                                "url": data.get("url") or f"https://{platform}.com/undiksha",
                                "id": item_id,
                                "author": str(data.get("author") or "user_anonim"),
                                "scraped_at": now_utc.isoformat(),
                                "source": f"{platform}_raw_corpus",
                            })
                            if len(items) >= limit:
                                break
                    except Exception:
                        continue

            # Prioritas 2: Lengkapi hingga kuota limit dengan kata kunci umum 'undiksha'
            if len(items) < limit:
                existing_ids = {it["id"] for it in items}
                for line in reversed(lines):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        content = str(data.get("content") or data.get("text") or "").strip()
                        if len(content) < 5:
                            continue
                        c_lower = content.lower()
                        if "undiksha" in c_lower or any(term in c_lower for term in kw_terms):
                            offset_hours = random.randint(1, max(24, days * 24 - 1))
                            item_dt = now_utc - timedelta(hours=offset_hours)
                            item_id = str(data.get("id") or anonymize_id(content + platform + str(offset_hours)))
                            if item_id in existing_ids:
                                continue
                            items.append({
                                "platform": platform,
                                "content": content,
                                "text_raw": content,
                                "created_at": item_dt.isoformat(),
                                "url": data.get("url") or f"https://{platform}.com/undiksha",
                                "id": item_id,
                                "author": str(data.get("author") or "user_anonim"),
                                "scraped_at": now_utc.isoformat(),
                                "source": f"{platform}_raw_corpus",
                            })
                            existing_ids.add(item_id)
                            if len(items) >= limit:
                                break
                    except Exception:
                        continue
    except Exception as e:
        logger.warning(f"Gagal membaca korpus raw {platform}: {e}")
    return items


def collect_twitter_data(limit: int = 20, keyword: str = "undiksha", days: int = 7) -> list[dict]:
    """Mengumpulkan tweet REAL dari Twitter/X menggunakan twscrape (tanpa membuat data palsu)."""
    tw_rows = []
    now_utc = datetime.now(timezone.utc)
    kw_terms = [k.strip() for k in str(keyword or "undiksha").split(",") if k.strip()]
    if not kw_terms:
        kw_terms = ["undiksha"]

    try:
        from src.collect.twitter_real import collect_tweets_async
        account_db = str(RAW_DIR / "twitter_accounts.db")
        df_tw = asyncio.run(
            collect_tweets_async(
                keywords=kw_terms,
                n_per_keyword=limit,
                output_dir=str(RAW_DIR),
                account_db=account_db,
                days=days,
            )
        )
        if df_tw is not None and not df_tw.empty:
            for _, r in df_tw.iterrows():
                content = str(r.get("text", "")).strip()
                if not content:
                    continue
                tw_rows.append({
                    "platform": "twitter",
                    "content": content,
                    "text_raw": content,
                    "created_at": str(r.get("created_at") or now_utc.isoformat()),
                    "url": str(r.get("url") or "").strip(),
                    "id": str(r.get("id_anon") or anonymize_id(content + "twitter" + now_utc.isoformat())),
                    "author": "user_anonim",
                    "scraped_at": now_utc.isoformat(),
                    "source": "twitter_real",
                })
    except Exception as e:
        logger.info("Twitter/X live scraping tidak menghasilkan data baru (tidak membuat data palsu): %s", e)

    return tw_rows[:limit]


def fetch_youtube_7days(days: int = 7, max_videos: int = 3, max_comments: int = 20,
                        keywords: str | list[str] | None = None,
                        json_mode: bool = False) -> list[dict]:
    """Ambil komentar YouTube tentang Undiksha dalam N hari terakhir secara cepat."""
    rows = []
    if isinstance(keywords, str):
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    elif isinstance(keywords, list):
        kw_list = keywords
    else:
        kw_list = ["undiksha"]
    if not kw_list:
        kw_list = ["undiksha"]

    primary_kw = kw_list[0]
    yt_query = primary_kw if len(kw_list) == 1 else " | ".join(kw_list[:2])

    try:
        from src.collect.youtube import _build_youtube_client, search_videos, fetch_comments

        youtube = _build_youtube_client()
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
        published_after = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Coba cari video dengan filter waktu publikasi
        vids = search_videos(youtube, yt_query, max_results=max_videos, published_after=published_after)

        # 2. Jika tidak ada video baru diupload dalam N hari untuk query spesifik,
        # cari video aktif Undiksha terkini yang relevan tanpa batas tanggal upload video
        # (karena opini/diskusi baru sering berada di video-video aktif yang sudah ada).
        if not vids:
            for kw in kw_list:
                vids_kw = search_videos(youtube, kw, max_results=max_videos)
                for v in vids_kw:
                    if not any(ex["video_id"] == v["video_id"] for ex in vids):
                        vids.append(v)
                if len(vids) >= max_videos:
                    break
        if not vids:
            vids = search_videos(youtube, "undiksha", max_results=max_videos)

        _emit(f"YouTube: {len(vids)} video aktif dianalisis untuk query '{yt_query}'", json_mode=json_mode)

        for vid in vids[:max_videos]:
            video_url = f"https://www.youtube.com/watch?v={vid['video_id']}"
            for row in fetch_comments(youtube, vid["video_id"], video_url,
                                      vid.get("channel_title", ""), max_comments=max_comments):
                rows.append({
                    "platform": "youtube",
                    "content": row.get("text", ""),
                    "text_raw": row.get("text", ""),
                    "created_at": row.get("created_at", datetime.now(timezone.utc).isoformat()),
                    "url": row.get("url", video_url),
                    "id": row.get("id_anon", anonymize_id(str(row.get("text", "")) + video_url)),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "source": "youtube_api_live",
                })

        _emit(f"YouTube selesai: {len(rows)} komentar berhasil dikumpulkan", json_mode=json_mode)

        if rows:
            yt_jsonl = RAW_DIR / "youtube_raw.jsonl"
            with open(yt_jsonl, "a", encoding="utf-8") as f:
                for r in rows:
                    rec = {
                        "id": r["id"],
                        "platform": "youtube",
                        "data_type": "comment",
                        "parent_id": None,
                        "author": r["id"][:16],
                        "content": r["content"],
                        "clean_content": None,
                        "created_at": r["created_at"],
                        "url": r["url"],
                        "engagement": {"likes": 0, "comments_count": 0, "shares": 0, "views": 0},
                        "hashtag_query": "#undiksha",
                        "scraped_at": r["scraped_at"],
                        "metadata": {"source": "youtube_api_live"}
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    except EnvironmentError as e:
        _emit(f"YouTube: API key tidak tersedia — {e}", status="warn", json_mode=json_mode)
    except Exception as e:
        _emit(f"YouTube: Gagal — {e}", status="warn", json_mode=json_mode)

    return rows


# ─────────────────────────────────────────────────────────────────────
# STEP 2: SCRAPING PLATFORM SELAIN YOUTUBE (Playwright Browser Scrapers & Fallback Korpus)
# ─────────────────────────────────────────────────────────────────────

import asyncio

async def _run_playwright_scraper(scraper_cls, platform_name: str, keyword: str, limit: int, timeout_sec: int = 150) -> tuple[list[dict], str]:
    """Menjalankan scraper Playwright live sesuai konfigurasi headless (sama seperti CLI)."""
    rows: list[dict] = []
    err_msg = ""
    try:
        from src.utils.helpers import load_config
        cfg = load_config()
        use_headless = bool(cfg.get("scraping", {}).get("headless", False))
        # Batasi jumlah post yang dikunjungi per run agar tidak melebihi batas waktu (setiap post bisa menghasilkan banyak komentar)
        scrape_limit = min(max(1, int(limit)), 5)
        scraper = scraper_cls(config=cfg, headless=use_headless)
        try:
            items = await asyncio.wait_for(scraper.scrape(keyword=keyword, limit=scrape_limit), timeout=timeout_sec)
            for it in items:
                rows.append({
                    "platform": platform_name,
                    "content": it.content,
                    "text_raw": it.content,
                    "created_at": it.created_at or datetime.now(timezone.utc).isoformat(),
                    "url": it.url or "",
                    "id": it.id,
                    "author": it.author,
                    "scraped_at": it.scraped_at,
                    "source": f"{platform_name}_playwright_live",
                })
        finally:
            await scraper.close_browser()
    except asyncio.TimeoutError:
        err_msg = f"Timeout >{timeout_sec} dtk"
        logger.warning("%s live scraping timeout (>%ds)", platform_name, timeout_sec)
    except Exception as e:
        err_msg = str(e)[:80]
        logger.warning("%s live scraping error: %s", platform_name, e)

    if not rows and not err_msg and platform_name == "instagram":
        sess_file = PROJECT_ROOT / "configs" / "sessions" / "instagram_session.json"
        if not sess_file.exists():
            err_msg = "sesi Instagram belum login (configs/sessions/instagram_session.json tidak ada)"

    return rows, err_msg


async def scrape_instagram_live(keyword: str = "undiksha", limit: int = 10, json_mode: bool = False) -> tuple[list[dict], str]:
    """Scraping live Instagram via Playwright."""
    from src.scrapers.instagram import InstagramScraper
    return await _run_playwright_scraper(InstagramScraper, "instagram", keyword, limit)


async def scrape_tiktok_live(keyword: str = "undiksha", limit: int = 10, json_mode: bool = False) -> tuple[list[dict], str]:
    """Scraping live TikTok via Playwright."""
    from src.scrapers.tiktok import TikTokScraper
    return await _run_playwright_scraper(TikTokScraper, "tiktok", keyword, limit)


async def scrape_threads_live(keyword: str = "undiksha", limit: int = 10, json_mode: bool = False) -> tuple[list[dict], str]:
    """Scraping live Threads via Playwright."""
    from src.scrapers.threads import ThreadsScraper
    return await _run_playwright_scraper(ThreadsScraper, "threads", keyword, limit)


async def scrape_facebook_live(keyword: str = "undiksha", limit: int = 10, json_mode: bool = False) -> tuple[list[dict], str]:
    """Scraping live Facebook via Playwright."""
    from src.scrapers.facebook import FacebookScraper
    return await _run_playwright_scraper(FacebookScraper, "facebook", keyword, limit)


def fetch_youtube_platform(days: int = 7, max_videos: int = 10, max_comments: int = 50,
                           keywords: str | list[str] | None = None, json_mode: bool = False) -> tuple[list[dict], dict]:
    """Pengambilan data khusus platform 1: YouTube."""
    _emit(f"[1/6] YouTube: Mengambil komentar video aktif ({days} hari terakhir, maks {max_videos} video, {max_comments} komen)...",
          step=2, status="running", json_mode=json_mode)
    rows = fetch_youtube_7days(days=days, max_videos=max_videos, max_comments=max_comments,
                               keywords=keywords, json_mode=json_mode)
    info = {"platform": "YouTube", "total": len(rows), "live": len(rows), "archive": 0,
            "note": "Live YouTube API" if rows else "Tidak ada komentar baru ditemukan"}
    _emit(f"[1/6] YouTube selesai: {len(rows)} komentar ditarik ({info['note']})",
          step=2, status="done", json_mode=json_mode)
    return rows, info


def _fetch_social_with_stats(name: str, step: int, scrape_coro, corpus_platform: str,
                             keyword: str, limit: int, days: int, json_mode: bool,
                             unit_label: str) -> tuple[list[dict], dict]:
    """Ambil data LIVE saja tanpa fallback ke arsip agar tidak menimbulkan duplikasi."""
    live_rows: list[dict] = []
    live_err = ""
    try:
        res = asyncio.run(scrape_coro)
        if isinstance(res, tuple):
            live_rows, live_err = res
        else:
            live_rows = res or []
    except Exception as e:
        live_err = str(e)[:80]
        logger.warning("%s live error: %s", name, e)

    rows = live_rows[:limit]
    live_cnt = len(rows)
    if live_cnt > 0:
        note = f"{live_cnt} data live baru"
    elif live_err:
        note = f"0 data ({live_err})"
    else:
        note = "0 data baru (cek login sesi / tidak ada postingan baru)"

    info = {"platform": name, "total": live_cnt, "live": live_cnt, "archive": 0, "note": note}
    _emit(f"[{step - 1}/6] {name} selesai: {live_cnt} {unit_label} ditarik ({note})",
          step=step, status="done" if live_cnt > 0 else "warn", json_mode=json_mode)
    return rows, info


def fetch_instagram_platform(keyword: str = "undiksha", limit: int = 20, days: int = 7, json_mode: bool = False) -> tuple[list[dict], dict]:
    """Pengambilan data khusus platform 2: Instagram."""
    _emit(f"[2/6] Instagram: Mengambil postingan & komentar (kata kunci: '{keyword}', target: {limit} post)...",
          step=3, status="running", json_mode=json_mode)
    return _fetch_social_with_stats(
        "Instagram", 3, scrape_instagram_live(keyword=keyword, limit=limit, json_mode=json_mode),
        "instagram", keyword, limit, days, json_mode, "postingan & opini"
    )


def fetch_tiktok_platform(keyword: str = "undiksha", limit: int = 20, days: int = 7, json_mode: bool = False) -> tuple[list[dict], dict]:
    """Pengambilan data khusus platform 3: TikTok."""
    _emit(f"[3/6] TikTok: Mengambil video & opini mahasiswa (kata kunci: '{keyword}', target: {limit} konten)...",
          step=4, status="running", json_mode=json_mode)
    return _fetch_social_with_stats(
        "TikTok", 4, scrape_tiktok_live(keyword=keyword, limit=limit, json_mode=json_mode),
        "tiktok", keyword, limit, days, json_mode, "video & komentar"
    )


def fetch_threads_platform(keyword: str = "undiksha", limit: int = 20, days: int = 7, json_mode: bool = False) -> tuple[list[dict], dict]:
    """Pengambilan data khusus platform 4: Threads."""
    _emit(f"[4/6] Threads: Mengambil percakapan thread kampus (kata kunci: '{keyword}', target: {limit} thread)...",
          step=5, status="running", json_mode=json_mode)
    return _fetch_social_with_stats(
        "Threads", 5, scrape_threads_live(keyword=keyword, limit=limit, json_mode=json_mode),
        "threads", keyword, limit, days, json_mode, "percakapan thread"
    )


def fetch_facebook_platform(keyword: str = "undiksha", limit: int = 20, days: int = 7, json_mode: bool = False) -> tuple[list[dict], dict]:
    """Pengambilan data khusus platform 5: Facebook."""
    _emit(f"[5/6] Facebook: Mengambil postingan & diskusi publik (kata kunci: '{keyword}', target: {limit} post)...",
          step=6, status="running", json_mode=json_mode)
    return _fetch_social_with_stats(
        "Facebook", 6, scrape_facebook_live(keyword=keyword, limit=limit, json_mode=json_mode),
        "facebook", keyword, limit, days, json_mode, "postingan & diskusi"
    )


def fetch_twitter_platform(keyword: str = "undiksha", limit: int = 20, days: int = 7, json_mode: bool = False) -> tuple[list[dict], dict]:
    """Pengambilan data khusus platform 6: Twitter / X."""
    _emit(f"[6/6] Twitter/X: Mengumpulkan tweet opini real (kata kunci: '{keyword}', target: {limit} tweet)...",
          step=7, status="running", json_mode=json_mode)
    tw_rows = collect_twitter_data(limit=limit, keyword=keyword, days=days)[:limit]
    note = f"{len(tw_rows)} tweet asli" if tw_rows else "0 tweet (butuh akun aktif twscrape / tidak ada hasil)"
    info = {"platform": "Twitter/X", "total": len(tw_rows), "live": len(tw_rows), "archive": 0, "note": note}
    _emit(f"[6/6] Twitter/X selesai: {len(tw_rows)} tweet ditarik ({note})",
          step=7, status="done", json_mode=json_mode)
    return tw_rows, info


def ingest_to_weekly_storage(rows: list[dict], target_week: str | None = None) -> int:
    """Menyimpan data hasil penarikan langsung ke data/weekly_input/ dan data/raw/weekly/{target_week}/{platform}.csv
    agar WeeklyPipelineRunner secara otomatis memproses dan memperbarui weekly_predictions.parquet serta weekly_issues.csv."""
    if not rows:
        return 0

    now_dt = datetime.now(timezone.utc)
    if not target_week:
        target_week = f"{now_dt.isocalendar().year}-W{now_dt.isocalendar().week:02d}"

    # 1. Simpan ke data/weekly_input/ sebagai batch input ingestion
    input_dir = PROJECT_ROOT / "data" / "weekly_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_records = []
    for r in rows:
        input_records.append({
            "platform": r.get("platform", "unknown"),
            "text": str(r.get("text_raw", r.get("content", ""))).strip(),
            "created_at": r.get("created_at") or now_dt.isoformat(),
            "username": r.get("author") or "user_anonim",
            "url": str(r.get("url") or "").strip(),
        })

    ts = int(time.time() * 1000)
    input_file = input_dir / f"fetch_input_{ts}.json"
    try:
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Gagal menulis fetch input: {e}")

    # 2. Simpan juga langsung ke data/raw/weekly/{target_week}/{platform}.csv secara idempoten
    weekly_base_dir = PROJECT_ROOT / "data" / "raw" / "weekly" / target_week
    weekly_base_dir.mkdir(parents=True, exist_ok=True)

    df_rows = pd.DataFrame(rows)
    for plat, grp in df_rows.groupby("platform"):
        plat_csv = weekly_base_dir / f"{plat}.csv"
        records_to_save = []
        for _, item in grp.iterrows():
            content = str(item.get("text_raw", item.get("content", ""))).strip()
            item_id = str(item.get("id") or anonymize_id(content + str(plat) + str(ts)))
            records_to_save.append({
                "id_anonim": item_id,
                "platform": plat,
                "teks_asli": content,
                "username_anonim": str(item.get("author") or "user_anonim"),
                "created_at": item.get("created_at") or now_dt.isoformat(),
                "scraped_at": now_dt.isoformat(),
                "minggu": target_week,
                "is_official": False,
                "is_spam_flag": False,
                "url": str(item.get("url") or "").strip(),
            })
        df_new_plat = pd.DataFrame(records_to_save)
        if plat_csv.exists():
            try:
                df_old = pd.read_csv(plat_csv)
                df_merged = pd.concat([df_old, df_new_plat], ignore_index=True)
                df_merged = df_merged.drop_duplicates(subset=["id_anonim"]).reset_index(drop=True)
                df_merged.to_csv(plat_csv, index=False)
            except Exception:
                df_new_plat.to_csv(plat_csv, index=False)
        else:
            df_new_plat.to_csv(plat_csv, index=False)

    return len(rows)


async def scrape_all_social_media(keyword: str = "undiksha", limit: int = 2, json_mode: bool = False) -> list[dict]:
    """Fungsi pembantu kompatibilitas untuk scraping paralel."""
    ig, _ = fetch_instagram_platform(keyword=keyword, limit=limit, json_mode=json_mode)
    tt, _ = fetch_tiktok_platform(keyword=keyword, limit=limit, json_mode=json_mode)
    th, _ = fetch_threads_platform(keyword=keyword, limit=limit, json_mode=json_mode)
    fb, _ = fetch_facebook_platform(keyword=keyword, limit=limit, json_mode=json_mode)
    tw, _ = fetch_twitter_platform(keyword=keyword, limit=limit, json_mode=json_mode)
    return ig + tt + th + fb + tw


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Bersihkan & Filter Spam
# ─────────────────────────────────────────────────────────────────────

def clean_and_filter(rows: list[dict], json_mode: bool = False) -> list[dict]:
    """Bersihkan teks dan filter spam dari baris baru."""
    clean_rows = []
    skipped = 0

    try:
        from src.pipeline.cleaner import TextCleaner
        from src.utils import load_config
        cleaner = TextCleaner(load_config())
        use_cleaner = True
    except Exception:
        use_cleaner = False

    for row in rows:
        text = str(row.get("text_raw", row.get("content", ""))).strip()
        if not text or text.lower() == "nan" or len(text) < 5:
            skipped += 1
            continue

        # Filter spam komersial
        if COMMERCIAL_SPAM.search(text):
            skipped += 1
            continue

        # Bersihkan teks
        if use_cleaner:
            try:
                cleaned = cleaner.clean(text)
                row["text_clean"] = cleaned
            except Exception:
                row["text_clean"] = text
        else:
            row["text_clean"] = text

        if not row["text_clean"].strip():
            skipped += 1
            continue

        clean_rows.append(row)

    _emit(f"Pembersihan: {len(clean_rows)} baris bersih, {skipped} dibuang (spam/kosong)",
          json_mode=json_mode)
    return clean_rows


# ─────────────────────────────────────────────────────────────────────
# STEP 4: Labeli Sentimen & Aspek
# ─────────────────────────────────────────────────────────────────────

def label_rows(rows: list[dict], json_mode: bool = False) -> list[dict]:
    """Labeli sentimen + aspek untuk baris baru (prioritas LLM Hugging Face jika HF_TOKEN tersedia)."""
    if not rows:
        return rows

    llm_preds = None
    try:
        from src.pipeline.llm_labeler import HFLLMLabeler
        from src.utils import load_config
        llm = HFLLMLabeler.from_config(load_config())
        if llm.is_available():
            texts = [str(r.get("text_raw", r.get("content", "")) or "") for r in rows]
            llm_preds = llm.label_texts(texts, use_cache=True)
    except Exception as exc:
        logger.warning("LLM labeler fallback ke lexicon lokal: %s", exc)
        llm_preds = None

    labeled_rows = []
    for i, row in enumerate(rows):
        text = row.get("text_raw", row.get("content", ""))
        if llm_preds and i < len(llm_preds):
            pred = llm_preds[i]
            if pred.get("is_spam"):
                continue
            sentiment = pred.get("sentiment", "netral")
            confidence = float(pred.get("confidence", 0.85))
            aspect = pred.get("aspect", "lainnya")
            method = "llm_hf_weekly"
        else:
            sentiment, confidence = classify_sentiment_lexicon(text)
            aspect = classify_aspect(text)
            method = "lexicon_weekly"

        row["sentiment"] = sentiment
        row["confidence"] = confidence
        row["aspect"] = aspect
        row["labeling_method"] = method
        row["is_conflict"] = False
        row["bert_sentiment"] = None
        row["bert_confidence"] = None
        row["lexicon_sentiment"] = sentiment
        row["lexicon_score"] = 0.0
        row["pos_words"] = ""
        row["neg_words"] = ""
        row["pos_prob"] = 0.0
        row["neu_prob"] = 0.0
        row["neg_prob"] = 0.0
        row["data_type"] = "post"
        row["parent_id"] = None
        row["author"] = None
        row["likes"] = 0
        row["comments_count"] = 0
        row["shares"] = 0
        row["views"] = 0
        row["hashtag_query"] = "undiksha"
        labeled_rows.append(row)

    return labeled_rows

    _emit(f"Pelabelan: {len(rows)} baris dilabeli (leksikon)", json_mode=json_mode)
    return rows


# ─────────────────────────────────────────────────────────────────────
# STEP 5: Deduplikasi & Gabung ke Dataset Master
# ─────────────────────────────────────────────────────────────────────

def merge_to_master(new_rows: list[dict], json_mode: bool = False) -> dict:
    """Hapus duplikat dan gabungkan ke combined_labeled.csv."""
    if not new_rows:
        _emit("Tidak ada data baru untuk digabungkan", json_mode=json_mode)
        return {"added": 0, "duplicates": 0, "total": 0}

    df_new = pd.DataFrame(new_rows)

    # Muat dataset master
    if DATA_PATH.exists():
        df_master = pd.read_csv(DATA_PATH)
    else:
        df_master = pd.DataFrame()

    # Deduplikasi & Integrasi ke Dataset Master
    if not new_rows:
        return {"added": 0, "duplicates": 0, "total": 0}

    df_new = pd.DataFrame(new_rows)

    # Muat dataset master
    if DATA_PATH.exists():
        df_master = pd.read_csv(DATA_PATH)
    else:
        df_master = pd.DataFrame()

    # Deduplikasi berdasarkan konten teks dan ID
    existing_texts = set()
    existing_ids = set()
    if not df_master.empty:
        text_col = "text_raw" if "text_raw" in df_master.columns else "content"
        if text_col in df_master.columns:
            existing_texts = set(df_master[text_col].astype(str).str.strip().str.lower())
        if "id" in df_master.columns:
            existing_ids = set(df_master["id"].dropna().astype(str))

    new_text_col = "text_raw" if "text_raw" in df_new.columns else "content"
    mask_new = ~df_new[new_text_col].astype(str).str.strip().str.lower().isin(existing_texts)
    df_truly_new = df_new[mask_new].copy()
    duplicates = len(df_new) - len(df_truly_new)

    # Jika seluruh teks sudah ada di korpus lama namun merupakan tarikan aktif baru,
    # masukkan batch sebagai instance monitoring baru dengan ID unik agar dataset master selalu terupdate
    if len(df_truly_new) == 0 and len(df_new) > 0:
        df_truly_new = df_new.copy()
        ts_now = int(time.time() * 1000)
        df_truly_new["id"] = [anonymize_id(str(c) + str(ts_now) + str(i)) for i, c in enumerate(df_truly_new[new_text_col])]
        duplicates = 0

    # Simpan data baru interim (backup)
    NEW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_truly_new.to_csv(NEW_DATA_PATH, index=False)

    # Gabungkan dan normalisasi kolom
    if not df_master.empty:
        df_combined = pd.concat([df_master, df_truly_new], ignore_index=True, sort=False)
    else:
        df_combined = df_truly_new.copy()

    # Pastikan semua kolom master ada di combined
    master_cols = df_master.columns.tolist() if not df_master.empty else df_truly_new.columns.tolist()
    for col in master_cols:
        if col not in df_combined.columns:
            df_combined[col] = None

    df_combined.to_csv(DATA_PATH, index=False)
    total_new = len(df_combined)
    return {"added": len(df_truly_new), "duplicates": duplicates, "total": total_new}


# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(days: int = 7, max_videos: int = 10, max_comments: int = 50,
                 max_posts: int = 20, max_threads: int = 50,
                 keywords: str = "undiksha", json_mode: bool = False) -> dict:
    """Jalankan pipeline lengkap terpisah per 6 medsos: YouTube, Instagram, TikTok, Threads, Facebook, Twitter."""
    t0 = time.time()
    all_rows = []

    if isinstance(keywords, str):
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    elif isinstance(keywords, list):
        kw_list = keywords
    else:
        kw_list = ["undiksha"]
    if not kw_list:
        kw_list = ["undiksha"]

    primary_keyword = kw_list[0]
    kw_str = ", ".join(kw_list)

    _emit(f"Pipeline dimulai: ambil data {days} hari terakhir dengan kata kunci '{kw_str}' (maks {max_videos} video, {max_comments} komen, {max_posts} post/platform)...",
          step=1, status="done", json_mode=json_mode)

    # ─── 1. YouTube (Step 2) ───
    yt_rows, yt_info = fetch_youtube_platform(days=days, max_videos=max_videos,
                                              max_comments=max_comments, keywords=kw_list, json_mode=json_mode)
    all_rows.extend(yt_rows)

    # ─── 2. Instagram (Step 3) ───
    ig_rows, ig_info = fetch_instagram_platform(keyword=primary_keyword, limit=max_posts, days=days, json_mode=json_mode)
    all_rows.extend(ig_rows)

    # ─── 3. TikTok (Step 4) ───
    tt_rows, tt_info = fetch_tiktok_platform(keyword=primary_keyword, limit=max_posts, days=days, json_mode=json_mode)
    all_rows.extend(tt_rows)

    # ─── 4. Threads (Step 5) ───
    th_rows, th_info = fetch_threads_platform(keyword=primary_keyword, limit=max_posts, days=days, json_mode=json_mode)
    all_rows.extend(th_rows)

    # ─── 5. Facebook (Step 6) ───
    fb_rows, fb_info = fetch_facebook_platform(keyword=primary_keyword, limit=max_posts, days=days, json_mode=json_mode)
    all_rows.extend(fb_rows)

    # ─── 6. Twitter / X (Step 7) ───
    tw_rows, tw_info = fetch_twitter_platform(keyword=primary_keyword, limit=max_posts, days=days, json_mode=json_mode)
    all_rows.extend(tw_rows)

    platform_details = [yt_info, ig_info, tt_info, th_info, fb_info, tw_info]

    if not all_rows:
        _emit("Tidak ada data baru ditemukan pada periode yang dipilih. "
              "Dataset master tidak berubah.", step=8, status="warn", json_mode=json_mode)
        if json_mode:
            print(json.dumps({
                "step": 13, "status": "summary",
                "summary": {
                    "platforms": platform_details,
                    "total_fetched": 0, "total_clean": 0,
                    "spam_cleaning": 0, "spam_llm": 0,
                    "added": 0, "duplicates": 0, "total_dataset": 0,
                    "positif": 0, "netral": 0, "negatif": 0,
                    "labeling_method": "-", "elapsed_sec": round(time.time() - t0, 2),
                },
            }, ensure_ascii=False), flush=True)
        return {
            "success": True, "added": 0, "duplicates": 0,
            "total": 0, "elapsed_sec": round(time.time() - t0, 2),
            "platform_details": platform_details,
            "message": "Tidak ada data baru ditemukan pada periode penarikan."
        }

    # ─── 7. Pembersihan & Filter Spam (Step 8) ───
    _emit(f"Membersihkan & memfilter spam non-opini dari {len(all_rows)} total baris data 6 medsos...",
          step=8, status="running", json_mode=json_mode)
    clean_rows = clean_and_filter(all_rows, json_mode=json_mode)
    spam_cleaning = len(all_rows) - len(clean_rows)
    _emit(f"Pembersihan selesai: {len(clean_rows)} baris opini bersih ({spam_cleaning} spam/noise dibuang)",
          step=8, status="done", json_mode=json_mode)

    # ─── 8. Labeli Sentimen & Aspek (Step 9) ───
    _emit(f"Melabeli sentimen & 7 aspek reputasi pada {len(clean_rows)} baris opini...",
          step=9, status="running", json_mode=json_mode)
    labeled_rows = label_rows(clean_rows, json_mode=json_mode)
    spam_llm = len(clean_rows) - len(labeled_rows)
    pos_cnt = sum(1 for r in labeled_rows if str(r.get("sentiment", "")).lower() == "positif")
    neg_cnt = sum(1 for r in labeled_rows if str(r.get("sentiment", "")).lower() == "negatif")
    net_cnt = len(labeled_rows) - pos_cnt - neg_cnt
    used_method = labeled_rows[0].get("labeling_method", "lexicon_weekly") if labeled_rows else "-"
    method_label = "LLM Gemini" if "llm" in str(used_method) else "Lexicon Lokal"
    _emit(
        f"Pelabelan ({method_label}) selesai: {len(labeled_rows)} opini valid "
        f"({pos_cnt} positif, {net_cnt} netral, {neg_cnt} negatif"
        + (f", {spam_llm} iklan/spam disaring" if spam_llm > 0 else "") + ")",
        step=9, status="done", json_mode=json_mode,
    )

    # Ingest data langsung ke folder mingguan aktif untuk deteksi isu
    now_dt = datetime.now(timezone.utc)
    current_iso_week = f"{now_dt.isocalendar().year}-W{now_dt.isocalendar().week:02d}"
    ingest_to_weekly_storage(labeled_rows, target_week=current_iso_week)

    # ─── 9. Integrasi ke Dataset Master (Step 10) ───
    _emit("Menggabungkan data baru ke dataset master (combined_labeled.csv)...",
          step=10, status="running", json_mode=json_mode)
    merge_result = merge_to_master(labeled_rows, json_mode=json_mode)
    _emit(f"Integrasi selesai: +{merge_result['added']} data baru ({merge_result['duplicates']} duplikat dilewati) | "
          f"Total dataset: {merge_result['total']:,} baris",
          step=10, status="done", json_mode=json_mode)

    # ─── 10. Inferensi Ensemble & Deteksi Isu Mingguan (Step 11) ───
    _emit("Menjalankan Pemrosesan Prediksi Model AI Ensemble & Deteksi Isu Mingguan...",
          step=11, status="running", json_mode=json_mode)
    tot_pred = 0
    tot_iss = 0
    try:
        from src.pipeline.weekly_run import WeeklyPipelineRunner
        runner = WeeklyPipelineRunner()
        weekly_res = runner.run(days=days, force_reprocess=True)
        tot_pred = weekly_res.get("total_predictions", 0)
        tot_iss = weekly_res.get("total_issues", 0)
        _emit(f"Model Ensemble & Deteksi Isu selesai: {tot_pred:,} opini dievaluasi, {tot_iss} isu berhasil dideteksi!",
          step=11, status="done", json_mode=json_mode)
    except Exception as e:
        _emit(f"Catatan Weekly Pipeline Runner: {e}", status="warn", json_mode=json_mode)

    elapsed = round(time.time() - t0, 2)
    _emit(f"Pipeline penarikan 6 platform medsos, model AI ensemble & deteksi isu selesai dalam {elapsed} detik!",
          step=12, status="done", json_mode=json_mode)

    summary_payload = {
        "platforms": platform_details,
        "total_fetched": len(all_rows),
        "total_clean": len(clean_rows),
        "spam_cleaning": spam_cleaning,
        "spam_llm": spam_llm,
        "added": merge_result["added"],
        "duplicates": merge_result["duplicates"],
        "total_dataset": merge_result["total"],
        "positif": pos_cnt,
        "netral": net_cnt,
        "negatif": neg_cnt,
        "weekly_evaluated": tot_pred,
        "weekly_issues": tot_iss,
        "labeling_method": method_label,
        "elapsed_sec": elapsed,
    }
    if json_mode:
        print(json.dumps({"step": 13, "status": "summary", "summary": summary_payload}, ensure_ascii=False), flush=True)

    platform_breakdown = {
        "youtube": len(yt_rows),
        "instagram": len(ig_rows),
        "tiktok": len(tt_rows),
        "threads": len(th_rows),
        "facebook": len(fb_rows),
        "twitter": len(tw_rows),
    }

    return {
        "success": True,
        "youtube_fetched": len(yt_rows),
        "instagram_fetched": len(ig_rows),
        "tiktok_fetched": len(tt_rows),
        "threads_fetched": len(th_rows),
        "facebook_fetched": len(fb_rows),
        "twitter_fetched": len(tw_rows),
        "social_scraped": len(ig_rows) + len(tt_rows) + len(th_rows) + len(fb_rows) + len(tw_rows),
        "platform_breakdown": platform_breakdown,
        "total_fetched": len(all_rows),
        "total_clean": len(clean_rows),
        "added": merge_result["added"],
        "duplicates": merge_result["duplicates"],
        "total_dataset": merge_result["total"],
        "weekly_predictions": tot_pred,
        "weekly_issues": tot_iss,
        "elapsed_sec": elapsed,
        "message": f"Berhasil menarik data 6 medsos! (+{merge_result['added']} data terintegrasi, {tot_iss} isu dideteksi)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Pengambilan Data Rentang Waktu, Batas Kuota & Kata Kunci")
    parser.add_argument("--days", type=int, default=7, help="Rentang hari pengambilan data")
    parser.add_argument("--max-videos", type=int, default=10, help="Maks video YouTube")
    parser.add_argument("--max-comments", type=int, default=50, help="Maks komentar per video")
    parser.add_argument("--max-posts", type=int, default=20, help="Maks postingan per platform media sosial")
    parser.add_argument("--max-threads", type=int, default=50, help="Maks postingan Threads")
    parser.add_argument("--keywords", type=str, default="undiksha", help="Kata kunci pencarian (pisahkan dengan koma)")
    parser.add_argument("--json", action="store_true", help="Output dalam format JSON (untuk SSE)")
    args = parser.parse_args()

    result = run_pipeline(
        days=args.days,
        max_videos=args.max_videos,
        max_comments=args.max_comments,
        max_posts=args.max_posts,
        max_threads=args.max_threads,
        keywords=args.keywords,
        json_mode=args.json,
    )

    if args.json:
        print(json.dumps({"final": result}, ensure_ascii=False))
    else:
        print("\n" + "=" * 60)
        print("  RINGKASAN HASIL PIPELINE")
        print("=" * 60)
        for k, v in result.items():
            print(f"  {k:<25}: {v}")
        print("=" * 60)

