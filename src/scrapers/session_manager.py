"""
Interactive Session Manager
Membuka browser headful bagi pengguna untuk melakukan login sekali,
lalu menyimpan cookies & storage state ke file JSON agar scraper berjalan otomatis tanpa login wall.
"""

import os
import asyncio
from typing import Optional
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from ..utils.logger import console
from ..utils.helpers import load_config


LOGIN_URLS = {
    "instagram": "https://www.instagram.com/accounts/login/",
    "facebook": "https://www.facebook.com/login",
    "threads": "https://www.threads.net/login",
    "tiktok": "https://www.tiktok.com/login"
}


async def interactive_login(platform: str, config_path: Optional[str] = None) -> bool:
    """
    Membuka browser Chromium interaktif agar pengguna dapat login manual.
    Setelah login sukses, pengguna menekan ENTER di terminal untuk menyimpan cookies.
    """
    platform = platform.lower()
    if platform not in LOGIN_URLS:
        console.print(f"[bold red]Platform '{platform}' tidak dikenali. Pilih: instagram, facebook, threads, atau tiktok.[/bold red]")
        return False

    config = load_config(config_path)
    sessions_dir = config.get("paths", {}).get("sessions_dir", "configs/sessions")
    session_file = config.get("platforms", {}).get(platform, {}).get("session_file", f"{sessions_dir}/{platform}_session.json")
    os.makedirs(os.path.dirname(session_file), exist_ok=True)

    login_url = LOGIN_URLS[platform]
    console.print(f"\n[bold cyan]=== SESI LOGIN INTERAKTIF: {platform.upper()} ===[/bold cyan]")
    console.print(f"[yellow]1. Jendela browser akan terbuka menuju {login_url}[/yellow]")
    console.print("[yellow]2. Silakan login menggunakan akun dummy/cadangan Anda di browser tersebut.[/yellow]")
    console.print("[yellow]3. Setelah berhasil masuk ke beranda/feed, KEMBALI ke terminal ini dan tekan tombol [ENTER].[/yellow]\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="id-ID",
            user_agent=config.get("scraping", {}).get("user_agent")
        )
        page = await context.new_page()
        try:
            await Stealth().apply_stealth_async(page)
        except Exception:
            pass

        await page.goto(login_url)

        # Tunggu input dari user di terminal
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, ">> Tekan [ENTER] jika Anda sudah selesai login di browser...")

        # Simpan session
        await context.storage_state(path=session_file)
        console.print(f"[bold green]✓ Sesi login {platform.upper()} berhasil disimpan di: {session_file}[/bold green]\n")
        await browser.close()
        return True


if __name__ == "__main__":
    import sys
    plat = sys.argv[1] if len(sys.argv) > 1 else "instagram"
    asyncio.run(interactive_login(plat))
