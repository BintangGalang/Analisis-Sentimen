"""
Scraper Facebook
Mengambil post publik dan komentar dari pencarian Facebook dengan kata kunci 'undiksha'.
"""

import uuid
from typing import List, Dict, Any
from .base import BaseScraper
from ..models.schema import Platform, DataType, EngagementMetrics, SocialMediaItem
from ..utils.logger import logger
from ..utils.helpers import random_delay, parse_numeric_metric


class FacebookScraper(BaseScraper):
    """Scraper untuk Facebook Search Posts."""

    def __init__(self, config: Dict[str, Any], headless: bool = False):
        super().__init__(platform_name="facebook", config=config, headless=headless)

    async def scrape(self, keyword: str = "undiksha", limit: int = 30) -> List[SocialMediaItem]:
        page = await self.init_browser()
        collected_items: List[SocialMediaItem] = []

        try:
            clean_kw = keyword.replace("#", "")
            search_url = f"https://www.facebook.com/search/posts/?q={clean_kw}"
            logger.info(f"[FACEBOOK] Membuka URL pencarian: {search_url}")

            await page.goto(search_url, wait_until="domcontentloaded")
            await random_delay(4.0, 6.0)

            # Tutup dialog cookie / login jika muncul
            try:
                close_btn = await page.query_selector("div[aria-label='Tutup'], div[aria-label='Close']")
                if close_btn:
                    await close_btn.click()
                    await random_delay(1.0, 2.0)
            except Exception:
                pass

            logger.info("[FACEBOOK] Melakukan scrolling untuk memuat postingan...")
            processed_posts = set()
            attempts = 0
            max_attempts = max(6, (limit // 2) + 5)

            while len(processed_posts) < limit and attempts < max_attempts:
                attempts += 1
                
                # Cari container artikel / postingan
                articles = await page.query_selector_all("div[role='feed'] > div, div[role='article']")
                if not articles:
                    articles = await page.query_selector_all("div[class*='userContentWrapper'], div[data-ad-preview='message']")

                for art in articles:
                    if len(processed_posts) >= limit:
                        break

                    try:
                        # Ekstrak author
                        author_el = await art.query_selector("h2 a, h3 a, strong span, a[role='link'] strong")
                        author = (await author_el.inner_text()).strip() if author_el else "Facebook User"

                        # Ekstrak teks konten postingan
                        text_els = await art.query_selector_all("div[dir='auto']")
                        post_text = ""
                        for el in text_els:
                            t = (await el.inner_text()).strip()
                            if len(t) > 10 and not any(k in t.lower() for k in ["komentar", "bagikan", "suka", "tanggapan"]):
                                post_text = t
                                break

                        if not post_text or post_text in processed_posts:
                            continue

                        processed_posts.add(post_text)
                        post_id = f"fb_post_{uuid.uuid4().hex[:12]}"

                        # Tautan post jika tersedia
                        post_url = page.url
                        link_el = await art.query_selector("a[href*='/posts/'], a[href*='/permalink/'], a[href*='story_fbid'], a[href*='/photo/'], a[href*='/videos/']")
                        if link_el:
                            href = await link_el.get_attribute("href")
                            if href:
                                if "/photo/" in href:
                                    post_url = href.split("&__cft__")[0].split("&__tn__")[0]
                                else:
                                    post_url = href.split("?")[0]
                                if not post_url.startswith("http"):
                                    post_url = f"https://www.facebook.com{post_url}"

                        # Ekstrak likes / reaksi
                        likes = 0
                        like_badge = await art.query_selector("span[role='toolbar'], span:has-text('suka')")
                        if like_badge:
                            likes = parse_numeric_metric(await like_badge.inner_text())

                        post_item = SocialMediaItem(
                            id=post_id,
                            platform=Platform.FACEBOOK,
                            data_type=DataType.POST,
                            parent_id=None,
                            author=author,
                            content=post_text,
                            url=post_url,
                            engagement=EngagementMetrics(likes=likes),
                            hashtag_query=keyword
                        )
                        self.save_item(post_item)
                        collected_items.append(post_item)

                        # Ekstrak komentar
                        max_comments = self.config.get("scraping", {}).get("max_comments_per_post", 40)
                        cmt_count = 0

                        # Metode 1: Buka halaman detail post jika memiliki permalink khusus
                        if self.context and post_url and post_url != page.url and any(k in post_url for k in ["/posts/", "/permalink/", "story_fbid", "/photo/", "/videos/"]):
                            try:
                                detail_page = await self.context.new_page()
                                await detail_page.goto(post_url, wait_until="domcontentloaded")
                                await random_delay(3.0, 4.5)

                                comments_data = await detail_page.evaluate('''() => {
                                    const results = [];
                                    const seen = new Set();
                                    const cmtArticles = Array.from(document.querySelectorAll("div[role='article'], div[aria-label*='Komentar oleh'], div[aria-label*='Comment by']"));

                                    cmtArticles.forEach(art => {
                                        const aria = art.getAttribute("aria-label") || "";
                                        let cAuthor = "Facebook User";
                                        const match = aria.match(/Komentar oleh ([^,]+)/) || aria.match(/Comment by ([^,]+)/);
                                        if (match) {
                                            cAuthor = match[1].trim();
                                        } else {
                                            const authorEl = art.querySelector("a[role='link'] strong, span strong, h3, h4");
                                            if (authorEl) cAuthor = authorEl.innerText.trim();
                                        }

                                        const textNodes = Array.from(art.querySelectorAll("div[dir='auto']"));
                                        let commentText = "";
                                        for (const t of textNodes) {
                                            const val = t.innerText.trim();
                                            if (val && val !== cAuthor && !['Suka', 'Balas', 'Bagikan'].includes(val)) {
                                                commentText = val;
                                                break;
                                            }
                                        }
                                        if (!commentText && art.innerText) {
                                            const lines = art.innerText.split('\\n').map(l => l.trim()).filter(l => l && l !== cAuthor && !['Suka', 'Balas', 'Bagikan'].includes(l));
                                            if (lines.length > 0) commentText = lines[0];
                                        }

                                        if (commentText && !seen.has(commentText)) {
                                            seen.add(commentText);
                                            results.push({ author: cAuthor, text: commentText });
                                        }
                                    });
                                    return results;
                                }''')

                                for cmt in comments_data[:max_comments]:
                                    c_text = cmt.get("text") or ""
                                    if c_text:
                                        cmt_id = f"fb_comment_{uuid.uuid4().hex[:12]}"
                                        cmt_item = SocialMediaItem(
                                            id=cmt_id,
                                            platform=Platform.FACEBOOK,
                                            data_type=DataType.COMMENT,
                                            parent_id=post_id,
                                            author=cmt.get("author") or "Komentator FB",
                                            content=c_text,
                                            url=post_url,
                                            hashtag_query=keyword
                                        )
                                        self.save_item(cmt_item)
                                        collected_items.append(cmt_item)
                                        cmt_count += 1

                                await detail_page.close()
                            except Exception as e:
                                logger.debug(f"[FACEBOOK] Gagal mengambil komentar detail {post_url}: {e}")

                        # Metode 2: Fallback ke komentar inline pada feed card jika cmt_count masih 0
                        if cmt_count == 0:
                            cmt_elements = await art.query_selector_all("ul li div[dir='auto'], div[aria-label*='Komentar oleh']")
                            for c_el in cmt_elements[:20]:
                                c_text = (await c_el.inner_text()).strip()
                                if c_text and c_text != post_text:
                                    cmt_id = f"fb_comment_{uuid.uuid4().hex[:12]}"
                                    cmt_item = SocialMediaItem(
                                        id=cmt_id,
                                        platform=Platform.FACEBOOK,
                                        data_type=DataType.COMMENT,
                                        parent_id=post_id,
                                        author="Komentator FB",
                                        content=c_text,
                                        url=post_url,
                                        hashtag_query=keyword
                                    )
                                    self.save_item(cmt_item)
                                    collected_items.append(cmt_item)
                                    cmt_count += 1

                        logger.info(f"[FACEBOOK] Berhasil ekstrak post ({author}) + {cmt_count} komentar.")

                    except Exception:
                        continue

                await self.human_scroll(page, times=2, distance=850)
                await random_delay(2.5, 4.0)

            logger.info(f"[FACEBOOK] Total postingan terkumpul: {len(processed_posts)}")

        except Exception as e:
            logger.error(f"[FACEBOOK] Terjadi kesalahan: {e}", exc_info=True)
        finally:
            await self.close_browser()

        return collected_items
