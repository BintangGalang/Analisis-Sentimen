"""
Base Scraper Class
Menyediakan fungsionalitas otomasi browser Playwright dengan fitur stealth,
manajemen session/cookies, scrolling alami, dan penanganan error.
"""

import os
import random
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth

from ..models.schema import SocialMediaItem
from ..utils.logger import logger
from ..utils.helpers import random_delay, append_to_jsonl


class BaseScraper(ABC):
    """
    Kelas dasar untuk semua scraper media sosial.
    """

    def __init__(
        self,
        platform_name: str,
        config: Dict[str, Any],
        headless: Optional[bool] = None,
        session_file: Optional[str] = None
    ):
        self.platform_name = platform_name
        self.config = config
        
        # Pengaturan browser dari config
        scraping_cfg = config.get("scraping", {})
        self.headless = headless if headless is not None else scraping_cfg.get("headless", False)
        # Jika headless, gunakan slow_mo=0 agar eksekusi cepat; jika non-headless baru perlambat
        self.slow_mo = 0 if self.headless else scraping_cfg.get("slow_mo", 300)
        self.timeout = scraping_cfg.get("timeout", 25000)
        self.user_agent = scraping_cfg.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        self.viewport = scraping_cfg.get("viewport", {"width": 1280, "height": 800})
        
        # Sesi / Cookie file
        self.session_file = session_file or config.get("platforms", {}).get(platform_name, {}).get("session_file")
        
        # Path output data mentah
        raw_dir = config.get("paths", {}).get("raw_data_dir", "data/raw")
        self.raw_output_file = os.path.join(raw_dir, f"{platform_name}_raw.jsonl")
        
        # Objek Playwright
        self._playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def init_browser(self) -> Page:
        """Memulai Playwright browser dan context dengan perlindungan stealth."""
        logger.info(f"[{self.platform_name.upper()}] Menginisialisasi browser (Headless: {self.headless})...")
        self._playwright = await async_playwright().start()
        
        # Buka Chromium browser
        self.browser = await self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--start-maximized"
            ]
        )

        # Muat session/cookies jika file tersedia
        context_kwargs = {
            "viewport": self.viewport,
            "user_agent": self.user_agent,
            "locale": "id-ID",
            "timezone_id": "Asia/Makassar",  # Waktu WITA untuk Bali / Undiksha
        }
        
        if self.session_file and os.path.exists(self.session_file):
            logger.info(f"[{self.platform_name.upper()}] Memuat session tersimpan dari: {self.session_file}")
            context_kwargs["storage_state"] = self.session_file
            
        self.context = await self.browser.new_context(**context_kwargs)
        self.page = await self.context.new_page()
        self.page.set_default_timeout(self.timeout)

        # Blokir resource berat (media/font) saat headless agar loading hemat bandwidth & super cepat
        if self.headless:
            try:
                await self.page.route(
                    "**/*",
                    lambda route: route.abort() if route.request.resource_type in ["media", "font"] else route.continue_()
                )
            except Exception:
                pass

        # Aktifkan stealth mode untuk menghindari deteksi bot
        try:
            await Stealth().apply_stealth_async(self.page)
        except Exception as e:
            logger.warning(f"Gagal mengaktifkan stealth mode: {e}")

        return self.page

    async def close_browser(self) -> None:
        """Menutup browser dan melepaskan resource."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info(f"[{self.platform_name.upper()}] Browser ditutup.")

    async def save_storage_state(self, filepath: Optional[str] = None) -> None:
        """Menyimpan session/cookies browser saat ini agar bisa dipakai berikutnya."""
        path = filepath or self.session_file
        if path and self.context:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            await self.context.storage_state(path=path)
            logger.info(f"[{self.platform_name.upper()}] Session berhasil disimpan ke {path}")

    async def human_scroll(self, page: Page, times: int = 3, distance: int = 700) -> None:
        """Melakukan scroll halaman ke bawah secara berkala seperti manusia."""
        for i in range(times):
            jitter = random.randint(-100, 100)
            await page.evaluate(f"window.scrollBy(0, {distance + jitter});")
            await random_delay(1.5, 3.0)

    def save_item(self, item: SocialMediaItem) -> None:
        """Menyimpan item post/komentar ke raw JSONL storage secara real-time."""
        append_to_jsonl(self.raw_output_file, item.model_dump())

    @abstractmethod
    async def scrape(self, keyword: str = "undiksha", limit: int = 30) -> List[SocialMediaItem]:
        """
        Metode utama scraping yang harus diimplementasikan oleh setiap subclass.
        """
        pass
