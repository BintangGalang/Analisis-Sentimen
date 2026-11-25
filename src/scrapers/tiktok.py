"""
Scraper TikTok
Mengambil video post dan komentar dengan tagar/query #undiksha dari tiktok.com.
"""

import asyncio
import re
import uuid
from typing import List, Dict, Any
from .base import BaseScraper
from ..models.schema import Platform, DataType, EngagementMetrics, SocialMediaItem
from ..utils.logger import logger
from ..utils.helpers import random_delay, parse_numeric_metric


class TikTokScraper(BaseScraper):
    """Scraper untuk TikTok Web."""

    def __init__(self, config: Dict[str, Any], headless: bool = False):
        super().__init__(platform_name="tiktok", config=config, headless=headless)

    async def scrape(self, keyword: str = "undiksha", limit: int = 30) -> List[SocialMediaItem]:
        page = await self.init_browser()
        collected_items: List[SocialMediaItem] = []
        video_urls = set()

        try:
            clean_kw = keyword.replace("#", "")
            # Prioritaskan halaman pencarian TikTok Web
            search_url = f"https://www.tiktok.com/search?q={clean_kw}"
            logger.info(f"[TIKTOK] Membuka halaman pencarian: {search_url}")

            await page.goto(search_url, wait_until="domcontentloaded")
            await random_delay(4.0, 6.0)

            # Cek jika muncul error server "Something went wrong" / "Coba lagi"
            server_err = await page.query_selector("button:has-text('Try again'), button:has-text('Coba lagi')")
            if server_err:
                logger.info("[TIKTOK] Terdeteksi tombol 'Try again', mencoba reload atau klik...")
                try:
                    await server_err.click()
                    await random_delay(3.0, 5.0)
                except Exception:
                    pass

            # Jika masih kosong, coba fallback ke halaman tag
            initial_links = await page.query_selector_all("a[href*='/video/']")
            if len(initial_links) == 0:
                tag_url = f"https://www.tiktok.com/tag/{clean_kw}"
                logger.info(f"[TIKTOK] Mencoba halaman alternatif tag: {tag_url}")
                try:
                    await page.goto(tag_url, wait_until="domcontentloaded")
                    await random_delay(4.0, 5.5)
                except Exception:
                    pass

            # Kumpulkan link video
            logger.info(f"[TIKTOK] Mengumpulkan tautan video (Target: {limit})...")
            attempts = 0
            max_attempts = max(6, (limit // 3) + 4)

            while len(video_urls) < limit and attempts < max_attempts:
                attempts += 1
                links = await page.query_selector_all("a[href*='/video/']")
                for link in links:
                    href = await link.get_attribute("href")
                    if href and "/video/" in href:
                        clean_url = href.split("?")[0]
                        if not clean_url.startswith("http"):
                            clean_url = f"https://www.tiktok.com{clean_url}"
                        video_urls.add(clean_url)
                        if len(video_urls) >= limit:
                            break

                await self.human_scroll(page, times=2, distance=900)
                await random_delay(2.0, 3.5)

            logger.info(f"[TIKTOK] Ditemukan {len(video_urls)} video.")

            # Kunjungi setiap video untuk mengekstrak caption & komentar
            for idx, vid_url in enumerate(list(video_urls)[:limit], start=1):
                logger.info(f"[TIKTOK] [{idx}/{min(len(video_urls), limit)}] Membuka video: {vid_url}")
                try:
                    await page.goto(vid_url, wait_until="domcontentloaded")
                    await random_delay(3.0, 5.0)

                    # Tutup popup modal / login prompt jika muncul
                    try:
                        close_btn = await page.query_selector("button[aria-label='Close'], div[aria-label='Close'], [data-e2e='modal-close-inner-button']")
                        if close_btn:
                            await close_btn.click(force=True)
                            await asyncio.sleep(1)
                    except Exception:
                        pass

                    # Ekstrak video ID
                    vid_id_match = re.search(r"/video/(\d+)", vid_url)
                    video_id = f"tiktok_post_{vid_id_match.group(1)}" if vid_id_match else f"tiktok_post_{uuid.uuid4().hex[:10]}"

                    # Ekstrak pembuat video (author) dari URL video agar tidak tertukar dengan akun login dummy
                    author_match = re.search(r"/@([^/]+)/video/", vid_url)
                    author = author_match.group(1) if author_match else "unknown"

                    # Ambil deskripsi / caption video
                    caption = ""
                    desc_selectors = [
                        "[data-e2e='browse-video-desc']",
                        "[data-e2e='video-desc']",
                        "h1[data-e2e='browse-video-desc']",
                        "div[class*='DivDescriptionContainer']"
                    ]
                    for sel in desc_selectors:
                        desc_el = await page.query_selector(sel)
                        if desc_el:
                            caption = (await desc_el.inner_text()).strip()
                            if caption:
                                break

                    # Ambil metrik likes & comments count
                    likes = 0
                    like_el = await page.query_selector("[data-e2e='like-count'], [data-e2e='browse-like-count']")
                    if like_el:
                        likes = parse_numeric_metric(await like_el.inner_text())

                    comment_cnt = 0
                    cmt_count_el = await page.query_selector("[data-e2e='comment-count']")
                    if cmt_count_el:
                        comment_cnt = parse_numeric_metric(await cmt_count_el.inner_text())

                    # Simpan Post Video
                    post_item = SocialMediaItem(
                        id=video_id,
                        platform=Platform.TIKTOK,
                        data_type=DataType.POST,
                        parent_id=None,
                        author=author,
                        content=caption,
                        url=vid_url,
                        engagement=EngagementMetrics(likes=likes, comments_count=comment_cnt),
                        hashtag_query=keyword
                    )
                    self.save_item(post_item)
                    collected_items.append(post_item)

                    # Ekstrak komentar
                    max_comments = self.config.get("scraping", {}).get("max_comments_per_post", 40)
                    logger.info(f"[TIKTOK] Mengambil komentar untuk {video_id} (Maks: {max_comments})...")

                    # Pastikan tab komentar aktif di Tab Bar TUX UI
                    try:
                        # Prioritaskan klik ikon bubble komentar
                        bubble = page.locator("[data-e2e='comment-icon'], [data-e2e='browse-comment-icon']").first
                        if await bubble.count() > 0:
                            await bubble.click(force=True)
                        else:
                            # Fallback ke tab bar
                            komentar_tab = page.locator("div[class*='tux-tabbar-item'], [data-testid='tux-web-tab-bar'] div").filter(has_text=re.compile(r"Komentar|Comments", re.I)).first
                            if await komentar_tab.count() > 0:
                                await komentar_tab.click(force=True)
                        await random_delay(2.0, 3.0)
                    except Exception:
                        pass

                    # Cek jika ada slider Captcha puzzle verifikasi
                    captcha_slider = await page.query_selector("div:has-text('Drag the slider'), div:has-text('Geser penggeser'), div:has-text('Tarik penggeser'), .secsdk-captcha-drag-icon")
                    if captcha_slider:
                        if not self.headless:
                            logger.warning("[TIKTOK] Terdeteksi slider Captcha puzzle TikTok ('Tarik penggeser')! Menunggu 15 detik jika ingin digeser secara manual di browser...")
                            await asyncio.sleep(15)
                        else:
                            logger.info("[TIKTOK] Terdeteksi captcha puzzle saat headless, melanjutkan ekstraksi tanpa jeda...")

                    # Scroll container komentar
                    for _ in range(3):
                        await page.evaluate("""() => {
                            const container = document.querySelector("[class*='DivCommentListContainer'], [class*='DivScrollingContentContainer'], [class*='RightPanel']");
                            if (container) {
                                container.scrollTop = container.scrollHeight;
                            } else {
                                window.scrollBy(0, 800);
                            }
                        }""")
                        await random_delay(1.5, 2.0)

                    # Seleksi dan ekstraksi elemen komentar menggunakan DOM modern
                    comments_data = await page.evaluate(r'''() => {
                        const results = [];
                        const seenTexts = new Set();
                        const cards = document.querySelectorAll(
                            "[data-e2e='comment-level-1'], " +
                            "div[class*='DivCommentItemContainer'], " +
                            "div[class*='DivCommentContentContainer'], " +
                            "div[class*='CommentItem']"
                        );

                        cards.forEach(card => {
                            const userLink = card.querySelector("a[href*='/@']");
                            let user = "unknown";
                            if (userLink) {
                                const href = userLink.getAttribute("href") || "";
                                const m = href.match(/\/@([^\/\?]+)/);
                                user = (m && m[1]) ? m[1] : userLink.innerText.trim();
                            }

                            const pText = card.querySelector("p");
                            const text = pText ? pText.innerText.trim() : card.innerText.trim();

                            if (text && text.length > 0 && !seenTexts.has(text)) {
                                seenTexts.add(text);
                                results.push({ author: user, text: text });
                            }
                        });
                        return results;
                    }''')

                    saved_cmt_count = 0
                    for cmt in comments_data[:max_comments]:
                        c_user = cmt.get("author") or "unknown"
                        c_text = cmt.get("text") or ""

                        if c_text:
                            comment_id = f"tiktok_comment_{uuid.uuid4().hex[:12]}"
                            cmt_item = SocialMediaItem(
                                id=comment_id,
                                platform=Platform.TIKTOK,
                                data_type=DataType.COMMENT,
                                parent_id=video_id,
                                author=c_user,
                                content=c_text,
                                url=vid_url,
                                hashtag_query=keyword
                            )
                            self.save_item(cmt_item)
                            collected_items.append(cmt_item)
                            saved_cmt_count += 1

                    logger.info(f"[TIKTOK] Berhasil menyimpan video {video_id} (@{author}) dengan {saved_cmt_count} komentar.")

                except Exception as e:
                    logger.warning(f"[TIKTOK] Gagal memproses video {vid_url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"[TIKTOK] Terjadi kesalahan: {e}", exc_info=True)
        finally:
            await self.close_browser()

        return collected_items
