"""
Scraper Instagram
Mengambil postingan dan komentar dengan tagar #undiksha dari instagram.com.
Mendukung sesi login untuk bypass limit penjelajahan publik.
"""

import re
import uuid
from typing import List, Dict, Any
from .base import BaseScraper
from ..models.schema import Platform, DataType, EngagementMetrics, SocialMediaItem
from ..utils.logger import logger
from ..utils.helpers import random_delay, parse_numeric_metric


class InstagramScraper(BaseScraper):
    """Scraper untuk Instagram Explore Tags & Posts."""

    def __init__(self, config: Dict[str, Any], headless: bool = False):
        super().__init__(platform_name="instagram", config=config, headless=headless)

    async def scrape(self, keyword: str = "undiksha", limit: int = 30) -> List[SocialMediaItem]:
        page = await self.init_browser()
        collected_items: List[SocialMediaItem] = []
        post_urls = set()

        try:
            clean_kw = re.sub(r"[\s#]+", "", keyword) or "undiksha"
            tag_url = f"https://www.instagram.com/explore/tags/{clean_kw}/"
            logger.info(f"[INSTAGRAM] Membuka URL tag: {tag_url}")

            await page.goto(tag_url, wait_until="domcontentloaded")
            await random_delay(4.0, 6.0)

            # Cek jika ada popup login modal yang bisa ditutup atau login wall
            try:
                close_btn = await page.query_selector("svg[aria-label='Tutup'], svg[aria-label='Close']")
                if close_btn:
                    await close_btn.click()
                    await random_delay(1.0, 2.0)
            except Exception:
                pass

            # Kumpulkan link postingan / reels
            logger.info(f"[INSTAGRAM] Mengumpulkan URL postingan (Target: {limit})...")
            attempts = 0
            max_attempts = max(6, (limit // 3) + 5)

            while len(post_urls) < limit and attempts < max_attempts:
                attempts += 1
                links = await page.query_selector_all("a[href*='/p/'], a[href*='/reel/']")
                for link in links:
                    href = await link.get_attribute("href")
                    if href:
                        clean_href = href.split("?")[0]
                        # Normalisasi tautan reel ke format /p/ agar membuka layout standar dengan panel komentar
                        if "/reel/" in clean_href:
                            clean_href = clean_href.replace("/reel/", "/p/")
                        full_url = f"https://www.instagram.com{clean_href}" if clean_href.startswith("/") else clean_href
                        post_urls.add(full_url)
                        if len(post_urls) >= limit:
                            break

                await self.human_scroll(page, times=2, distance=800)
                await random_delay(2.5, 4.0)

            logger.info(f"[INSTAGRAM] Ditemukan {len(post_urls)} tautan post.")

            # Kunjungi setiap postingan untuk ekstrak caption dan komentar
            max_comments = self.config.get("scraping", {}).get("max_comments_per_post", 30)

            for idx, post_url in enumerate(list(post_urls)[:limit], start=1):
                logger.info(f"[INSTAGRAM] [{idx}/{min(len(post_urls), limit)}] Memproses post: {post_url}")
                try:
                    await page.goto(post_url, wait_until="domcontentloaded")
                    await random_delay(3.0, 4.5)

                    # ID Post
                    id_match = re.search(r"/(?:p|reel)/([A-Za-z0-9_-]+)", post_url)
                    shortcode = id_match.group(1) if id_match else uuid.uuid4().hex[:10]
                    post_id = f"ig_post_{shortcode}"

                    # Coba klik tombol muat lebih banyak komentar jika ada
                    try:
                        for _ in range(2):
                            more_btn = await page.query_selector("svg[aria-label*='Load more'], svg[aria-label*='Muat komentar'], div[role='button']:has-text('Lihat komentar lainnya'), div[role='button']:has-text('View more comments'), div[role='button']:has-text('Load more comments')")
                            if more_btn:
                                await more_btn.click()
                                await random_delay(1.0, 2.0)
                            else:
                                break
                    except Exception:
                        pass

                    # Ekstraksi langsung di browser DOM untuk keakuratan dan performa maksimal
                    post_data = await page.evaluate(r'''(maxC) => {
                        const url = window.location.href;
                        const shortcodeMatch = url.match(/\/(?:p|reel)\/([A-Za-z0-9_-]+)/);
                        const shortcode = shortcodeMatch ? shortcodeMatch[1] : "";

                        // 1. Author
                        let author = "";
                        if (shortcode) {
                            const authorPostLink = document.querySelector(`a[href*='/p/${shortcode}/']`);
                            if (authorPostLink) {
                                const m = authorPostLink.getAttribute("href").match(/^\/([a-zA-Z0-9._]+)\/p\//);
                                if (m && m[1]) author = m[1];
                            }
                        }
                        if (!author) {
                            const userLinks = Array.from(document.querySelectorAll("main a[href^='/'], header a[href^='/'], div[role='main'] a[href^='/']"));
                            for (const a of userLinks) {
                                const href = a.getAttribute("href") || "";
                                const m = href.match(/^\/([a-zA-Z0-9._]+)\/?$/);
                                if (m && m[1] && !['explore', 'reels', 'direct', 'stories', 'accounts'].includes(m[1])) {
                                    author = a.innerText.trim() || m[1];
                                    break;
                                }
                            }
                        }
                        if (!author) {
                            const titleMatch = document.title.match(/^([^:@•]+?)\s*(?:on Instagram|di Instagram)/i);
                            if (titleMatch) author = titleMatch[1].trim();
                        }
                        if (!author) author = "unknown";

                        // 2. Caption
                        let caption = "";
                        const h1 = document.querySelector("h1");
                        if (h1 && h1.innerText.trim().length > 3) {
                            caption = h1.innerText.trim();
                        } else {
                            const spans = Array.from(document.querySelectorAll("main span[dir='auto'], div[role='main'] span[dir='auto']"));
                            for (const s of spans) {
                                const txt = s.innerText.trim();
                                if (txt.length > 5 && 
                                    !txt.includes("See translation") && 
                                    !txt.includes("Lihat terjemahan") && 
                                    !txt.includes("View hidden") && 
                                    !txt.includes("Comments on this") && 
                                    txt !== "Reply" && txt !== "Balas" && 
                                    txt !== author) {
                                    let cleanTxt = txt;
                                    if (author && cleanTxt.startsWith(author)) {
                                        cleanTxt = cleanTxt.substring(author.length).trim();
                                        cleanTxt = cleanTxt.replace(/^\d+[smhdw]\s*/i, '').trim();
                                    }
                                    if (cleanTxt.length > 3) {
                                        caption = cleanTxt;
                                        break;
                                    }
                                }
                            }
                        }

                        // 3. Likes
                        let likes = 0;
                        const pageText = document.body.innerText;
                        const likesMatch = pageText.match(/([\d\.,]+)\s*(?:likes|suka)/i);
                        if (likesMatch) {
                            const rawNum = likesMatch[1].replace(/\./g, '').replace(/,/g, '');
                            likes = parseInt(rawNum, 10) || 0;
                        }

                        // 4. Comments
                        const comments = [];
                        const seenIds = new Set();
                        const seenTexts = new Set();

                        // Teknik A: Cari via permalink komentar (/c/)
                        const cLinks = Array.from(document.querySelectorAll("a[href*='/c/']"));
                        for (const cl of cLinks) {
                            if (comments.length >= maxC) break;
                            const href = cl.getAttribute("href") || "";
                            const m = href.match(/\/c\/(\d+)\/?/);
                            const commentId = m ? m[1] : "";
                            if (commentId && seenIds.has(commentId)) continue;

                            const timeText = cl.innerText.trim();
                            let parent = cl;
                            let cAuthor = "";
                            let cText = "";

                            for (let d = 0; d < 8 && parent; d++) {
                                parent = parent.parentElement;
                                if (!parent) break;

                                const authorLink = parent.querySelector("a[href^='/']:not([href*='/c/']):not([href*='/p/'])");
                                if (authorLink) {
                                    const uMatch = (authorLink.getAttribute("href") || "").match(/^\/([a-zA-Z0-9._]+)\/?$/);
                                    if (uMatch) cAuthor = authorLink.innerText.trim() || uMatch[1];
                                }

                                const spans = Array.from(parent.querySelectorAll("span[dir='auto'], div.html-div"));
                                for (const s of spans) {
                                    const st = s.innerText.trim();
                                    if (st && 
                                        st !== cAuthor && 
                                        st !== timeText && 
                                        st !== "Reply" && 
                                        st !== "Balas" && 
                                        !st.includes("See translation") && 
                                        !st.includes("Lihat terjemahan") && 
                                        st !== caption &&
                                        !st.includes(timeText)) {
                                        cText = st;
                                        break;
                                    }
                                }

                                if (cAuthor && cText) break;
                            }

                            if (cAuthor && cText && !seenTexts.has(cText)) {
                                if (commentId) seenIds.add(commentId);
                                seenTexts.add(cText);
                                comments.push({
                                    id: commentId || "",
                                    author: cAuthor,
                                    content: cText
                                });
                            }
                        }

                        // Teknik B: Fallback via tombol "Reply" / "Balas"
                        if (comments.length === 0) {
                            const replyBtns = Array.from(document.querySelectorAll("div[role='button'], span, button")).filter(
                                el => el.innerText && (el.innerText.trim() === "Reply" || el.innerText.trim() === "Balas")
                            );
                            for (const btn of replyBtns) {
                                if (comments.length >= maxC) break;
                                let p = btn;
                                for (let d = 0; d < 7 && p; d++) {
                                    p = p.parentElement;
                                    if (!p) break;
                                    const aLink = p.querySelector("a[href^='/']:not([href*='/p/'])");
                                    if (aLink) {
                                        const uMatch = (aLink.getAttribute("href") || "").match(/^\/([a-zA-Z0-9._]+)\/?$/);
                                        if (uMatch && uMatch[1] !== author) {
                                            const cAuthor = aLink.innerText.trim() || uMatch[1];
                                            const spans = Array.from(p.querySelectorAll("span[dir='auto']"));
                                            for (const s of spans) {
                                                const st = s.innerText.trim();
                                                if (st && st !== cAuthor && st !== "Reply" && st !== "Balas" && !st.includes("translation") && st !== caption) {
                                                    if (!seenTexts.has(st)) {
                                                        seenTexts.add(st);
                                                        comments.push({
                                                            id: "",
                                                            author: cAuthor,
                                                            content: st
                                                        });
                                                    }
                                                    break;
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        return {
                            author: author,
                            caption: caption,
                            likes: likes,
                            comments: comments
                        };
                    }''', max_comments)

                    post_author = post_data.get("author") or "unknown"
                    post_caption = post_data.get("caption") or ""
                    likes = post_data.get("likes") or 0
                    raw_comments = post_data.get("comments") or []

                    post_item = SocialMediaItem(
                        id=post_id,
                        platform=Platform.INSTAGRAM,
                        data_type=DataType.POST,
                        parent_id=None,
                        author=post_author,
                        content=post_caption,
                        url=post_url,
                        engagement=EngagementMetrics(
                            likes=likes,
                            comments_count=len(raw_comments)
                        ),
                        hashtag_query=keyword
                    )
                    self.save_item(post_item)
                    collected_items.append(post_item)

                    cmt_count = 0
                    for cmt in raw_comments:
                        c_id = cmt.get("id")
                        comment_id = f"ig_comment_{c_id}" if c_id else f"ig_comment_{uuid.uuid4().hex[:12]}"
                        c_author = cmt.get("author") or "unknown"
                        c_text = cmt.get("content") or ""

                        cmt_item = SocialMediaItem(
                            id=comment_id,
                            platform=Platform.INSTAGRAM,
                            data_type=DataType.COMMENT,
                            parent_id=post_id,
                            author=c_author,
                            content=c_text,
                            url=post_url,
                            hashtag_query=keyword
                        )
                        self.save_item(cmt_item)
                        collected_items.append(cmt_item)
                        cmt_count += 1

                    logger.info(f"[INSTAGRAM] Selesai: Post {post_id} (@{post_author}) + {cmt_count} komentar.")

                except Exception as e:
                    logger.warning(f"[INSTAGRAM] Gagal memproses {post_url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"[INSTAGRAM] Terjadi kesalahan: {e}", exc_info=True)
        finally:
            await self.close_browser()

        return collected_items
