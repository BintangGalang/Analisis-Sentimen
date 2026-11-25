"""
Scraper Threads (Meta)
Mengambil post dan komentar/replies dengan tagar/query #undiksha dari threads.net.
"""

import re
import uuid
from typing import List, Dict, Any
from .base import BaseScraper
from ..models.schema import Platform, DataType, EngagementMetrics, SocialMediaItem
from ..utils.logger import logger
from ..utils.helpers import random_delay, parse_numeric_metric


class ThreadsScraper(BaseScraper):
    """Scraper untuk Threads.net."""

    def __init__(self, config: Dict[str, Any], headless: bool = False):
        super().__init__(platform_name="threads", config=config, headless=headless)

    async def scrape(self, keyword: str = "undiksha", limit: int = 30) -> List[SocialMediaItem]:
        page = await self.init_browser()
        collected_items: List[SocialMediaItem] = []
        post_urls = set()

        try:
            # Query pencarian
            clean_kw = keyword.replace("#", "")
            search_url = f"https://www.threads.net/search?q=%23{clean_kw}&serp_type=default"
            logger.info(f"[THREADS] Membuka URL pencarian: {search_url}")
            
            await page.goto(search_url, wait_until="domcontentloaded")
            await random_delay(3.0, 5.0)

            # Scroll untuk mengumpulkan URL thread
            logger.info(f"[THREADS] Mengumpulkan URL thread (Target: {limit} post)...")
            attempts = 0
            max_attempts = max(5, (limit // 3) + 5)

            while len(post_urls) < limit and attempts < max_attempts:
                attempts += 1
                # Cari link thread dengan berbagai variasi selector
                links = await page.query_selector_all("a[href*='/post/'], a[href*='/t/'], div[data-pressable-container='true'] a")
                for link in links:
                    href = await link.get_attribute("href")
                    if href and ("/post/" in href or "/t/" in href):
                        full_url = f"https://www.threads.net{href}" if href.startswith("/") else href
                        # Bersihkan query params
                        clean_url = full_url.split("?")[0]
                        post_urls.add(clean_url)
                        if len(post_urls) >= limit:
                            break

                await self.human_scroll(page, times=2, distance=800)
                await random_delay(2.0, 3.5)

            logger.info(f"[THREADS] Berhasil menemukan {len(post_urls)} link thread.")

            # Kunjungi setiap thread untuk mengambil post utama & balasan (komentar)
            # Kunjungi setiap thread untuk mengambil post utama & balasan (komentar)
            for idx, post_url in enumerate(list(post_urls)[:limit], start=1):
                logger.info(f"[THREADS] [{idx}/{min(len(post_urls), limit)}] Memproses thread: {post_url}")
                try:
                    await page.goto(post_url, wait_until="domcontentloaded")
                    await random_delay(3.0, 4.5)

                    # Ekstrak post ID dari URL
                    post_id_match = re.search(r"/post/([A-Za-z0-9_-]+)", post_url)
                    post_id = f"threads_post_{post_id_match.group(1)}" if post_id_match else f"threads_post_{uuid.uuid4().hex[:10]}"

                    # Ekstrak author utama langsung dari URL agar tidak tertukar dengan sidebar / menu profil
                    url_author_match = re.search(r"/@([^/]+)/post/", post_url)
                    main_author = url_author_match.group(1) if url_author_match else "unknown"

                    # Jika belum masuk ke tampilan detail Thread, klik kartu thread pertama
                    try:
                        has_back_btn = await page.query_selector("svg[aria-label='Back'], div:has-text('← Thread')")
                        if not has_back_btn:
                            first_card = page.locator("div[data-pressable-container='true']").first
                            if await first_card.count() > 0:
                                await first_card.click()
                                await random_delay(2.0, 3.0)
                    except Exception:
                        pass

                    # Scroll sedikit ke bawah untuk memuat balasan/komentar
                    await self.human_scroll(page, times=2, distance=500)
                    await random_delay(1.5, 2.5)

                    # Ekstrak konten thread utama dan komentar menggunakan DOM parser modern
                    thread_data = await page.evaluate(r'''(targetAuthor) => {
                        const containers = Array.from(document.querySelectorAll("div[data-pressable-container='true']"));
                        const timeRegex = /^(\d+[smhdwy]|\d{1,2}\/\d{1,2}\/\d{2,4})$/i;
                        const metricRegex = /^\d+(\.\d+)?[KkMm]?$/;
                        const uiWords = [
                            "translate", "terjemahkan", "balas", "suka", "bagikan",
                            "repost", "ikuti", "following", "follow", "show replies",
                            "lihat balasan", "top", "view activity", "berikan komentar",
                            "reply to", "postingan lain dari", "more from"
                        ];

                        let mainPost = null;
                        const replies = [];
                        let reachedOtherPosts = false;

                        containers.forEach((c, idx) => {
                            // Cek jika sudah mencapai section "Postingan lain dari..."
                            const containerText = (c.innerText || "").toLowerCase();
                            if (containerText.includes("postingan lain dari") || containerText.includes("more from")) {
                                reachedOtherPosts = true;
                                return;
                            }
                            if (reachedOtherPosts) return;

                            // Ekstrak author
                            const authorLink = c.querySelector("a[href^='/@']");
                            let author = "";
                            if (authorLink) {
                                const href = authorLink.getAttribute("href") || "";
                                const m = href.match(/\/@([^\/\?]+)/);
                                author = m ? m[1] : authorLink.innerText.trim().replace("@", "");
                            }

                            // Kumpulkan semua span teks
                            const spans = Array.from(c.querySelectorAll("span[dir='auto']"))
                                .map(s => (s.innerText || "").trim())
                                .filter(s => s.length > 0);

                            const contentSpans = spans.filter(txt => {
                                const low = txt.toLowerCase();
                                if (author && txt.toLowerCase() === author.toLowerCase()) return false;
                                if (timeRegex.test(txt)) return false;
                                if (metricRegex.test(txt)) return false;
                                if (uiWords.some(w => low.startsWith(w) || low === w)) return false;
                                return true;
                            });

                            const fullText = contentSpans.join("\n").trim();

                            // Likes
                            const metricSpans = spans.filter(txt => metricRegex.test(txt));
                            const likesStr = metricSpans.length > 0 ? metricSpans[0] : "0";

                            if (fullText.length > 0) {
                                if (!mainPost) {
                                    mainPost = {
                                        author: targetAuthor || author || "unknown",
                                        content: fullText,
                                        likes: likesStr
                                    };
                                } else {
                                    // Ini adalah reply / komentar
                                    if (fullText !== mainPost.content) {
                                        replies.push({
                                            author: author || "unknown",
                                            content: fullText
                                        });
                                    }
                                }
                            }
                        });

                        return { mainPost, replies };
                    }''', main_author)

                    main_post_info = thread_data.get("mainPost")
                    main_text = main_post_info.get("content", "") if main_post_info else ""
                    likes_val = parse_numeric_metric(main_post_info.get("likes", "0")) if main_post_info else 0

                    if not main_text:
                        # Fallback jika container pressable tidak terbaca
                        text_spans = await page.query_selector_all("span[dir='auto']")
                        for span in text_spans:
                            txt = (await span.inner_text()).strip()
                            if len(txt) > 10 and not any(w in txt.lower() for w in ["for you", "search", "activity", "following"]):
                                main_text = txt
                                break

                    # Simpan Post Utama
                    post_item = SocialMediaItem(
                        id=post_id,
                        platform=Platform.THREADS,
                        data_type=DataType.POST,
                        parent_id=None,
                        author=main_author,
                        content=main_text,
                        url=post_url,
                        engagement=EngagementMetrics(likes=likes_val),
                        hashtag_query=keyword
                    )
                    self.save_item(post_item)
                    collected_items.append(post_item)

                    # Ambil dan simpan balasan / komentar
                    comment_count = 0
                    max_comments = self.config.get("scraping", {}).get("max_comments_per_post", 30)
                    replies_list = thread_data.get("replies", [])

                    for r in replies_list[:max_comments]:
                        r_author = r.get("author") or "unknown"
                        r_text = r.get("content") or ""

                        if r_text and r_text != main_text:
                            comment_id = f"threads_comment_{uuid.uuid4().hex[:12]}"
                            comment_item = SocialMediaItem(
                                id=comment_id,
                                platform=Platform.THREADS,
                                data_type=DataType.COMMENT,
                                parent_id=post_id,
                                author=r_author,
                                content=r_text,
                                url=post_url,
                                hashtag_query=keyword
                            )
                            self.save_item(comment_item)
                            collected_items.append(comment_item)
                            comment_count += 1

                    logger.info(f"[THREADS] Berhasil mengekstrak post '{post_id}' (@{main_author}) + {comment_count} komentar.")

                except Exception as e:
                    logger.warning(f"[THREADS] Gagal mengekstrak thread {post_url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"[THREADS] Terjadi kesalahan saat scraping: {e}", exc_info=True)
        finally:
            await self.close_browser()

        return collected_items
