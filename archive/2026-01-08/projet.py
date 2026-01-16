import re
import csv
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

def parse_like_count(text: str) -> int:
    if not text:
        return 0
    t = text.strip().lower().replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    # 1,2k / 1.2k / 1k / 1 k etc.
    t = t.replace(",", ".")
    m = re.match(r"^(\d+(?:\.\d+)?)([kmk]?)$", t)
    if not m:
        return 0
    num, suf = m.groups()
    x = float(num)
    if suf == "k":
        x *= 1_000
    elif suf == "m":
        x *= 1_000_000
    return int(x)

def accept_consent(page):
    candidates = [
        "Tout accepter", "Accepter tout", "J'accepte", "J’accepte",
        "I agree", "Accept all", "Agree to all"
    ]
    for label in candidates:
        try:
            page.get_by_role("button", name=re.compile(label, re.I)).click(timeout=1500)
            page.wait_for_timeout(500)
            return
        except PWTimeout:
            pass
        except Exception:
            pass

def scroll_to_load(page, target_count=None, max_scrolls=300, idle_ms=1200):
    last_h = 0
    stagnant = 0
    for i in range(max_scrolls):
        page.mouse.wheel(0, 25000)
        page.wait_for_timeout(idle_ms)
        h = page.evaluate("document.documentElement.scrollHeight")
        if h == last_h:
            stagnant += 1
            if stagnant >= 2:
                break
        else:
            stagnant = 0
        last_h = h
        if target_count:
            seen = page.locator("ytd-comment-thread-renderer").count()
            if seen >= target_count:
                break

def extract_comment_fields(root):
    # root est un ytd-comment-renderer (top-level ou reply)
    try:
        author = root.locator("#author-text span").first.inner_text(timeout=800).strip()
    except:
        author = None
    try:
        text = root.locator("#content-text").all_inner_texts()[0].strip()
    except:
        text = None
    # bouton "Lire la suite"
    try:
        root.locator("#more, yt-formatted-string[role='button']#more").first.click(timeout=300)
    except:
        pass
    try:
        likes_txt = root.locator("#vote-count-middle").first.inner_text(timeout=600).strip()
    except:
        likes_txt = ""
    try:
        published_when = root.locator("a#published-time-text, span.published-time-text a").first.inner_text(timeout=800).strip()
    except:
        published_when = None
    return author, text, parse_like_count(likes_txt), published_when

def expand_replies(thread):
    # Déplie les réponses si un bouton "Afficher les réponses" existe
    # Peut nécessiter plusieurs clics si "Plus de réponses"
    for _ in range(3):
        try:
            btn = thread.locator("#replies ytd-button-renderer button, ytd-button-renderer#more-replies button").first
            if btn.is_visible():
                btn.click(timeout=800)
                thread.page.wait_for_timeout(700)
            else:
                break
        except:
            break

def scrape(url, out_csv, max_comments=1000, include_replies=False, headless=True, lang="fr-FR"):
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(locale=lang, user_agent=None)
        page = ctx.new_page()
        page.set_default_timeout(6000)

        page.goto(url)
        accept_consent(page)

        # S'assure que la section commentaires est chargée
        try:
            page.locator("#comments").scroll_into_view_if_needed(timeout=5000)
        except:
            pass

        scroll_to_load(page, target_count=min(max_comments, 500), max_scrolls=60, idle_ms=1200)

        # Boucle de scroll jusqu'à atteindre ~max_comments (approx.)
        while True:
            count_before = page.locator("ytd-comment-thread-renderer").count()
            if count_before >= max_comments:
                break
            prev_h = page.evaluate("document.documentElement.scrollHeight")
            scroll_to_load(page, target_count=max_comments, max_scrolls=20, idle_ms=900)
            curr_h = page.evaluate("document.documentElement.scrollHeight")
            count_after = page.locator("ytd-comment-thread-renderer").count()
            if curr_h == prev_h or count_after == count_before:
                break

        threads = page.locator("ytd-comment-thread-renderer")
        n = min(threads.count(), max_comments)

        for i in range(n):
            thread = threads.nth(i)
            top = thread.locator("ytd-comment-renderer").first
            try:
                author, text, likes, when = extract_comment_fields(top)
                if text:
                    rows.append({
                        "thread_index": i,
                        "is_reply": 0,
                        "author": author,
                        "text": text,
                        "like_count": likes,
                        "published_when": when
                    })
            except Exception:
                continue

            if include_replies:
                expand_replies(thread)
                # Récupère les replies une fois ouvertes
                replies = thread.locator("ytd-comment-replies-renderer ytd-comment-renderer")
                for j in range(replies.count()):
                    r = replies.nth(j)
                    try:
                        author, text, likes, when = extract_comment_fields(r)
                        if text:
                            rows.append({
                                "thread_index": i,
                                "is_reply": 1,
                                "author": author,
                                "text": text,
                                "like_count": likes,
                                "published_when": when
                            })
                    except:
                        pass

        browser.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scraper de commentaires YouTube (sans API) via Playwright.")
    ap.add_argument("url", help="URL de la vidéo YouTube (watch, youtu.be, shorts).")
    ap.add_argument("--out", default=None, help="Chemin CSV de sortie (par défaut: youtube_comments_<videoid>.csv)")
    ap.add_argument("--max", type=int, default=1000, help="Nombre max de commentaires top-level à tenter.")
    ap.add_argument("--replies", action="store_true", help="Inclure les réponses sous chaque commentaire.")
    ap.add_argument("--headed", action="store_true", help="Afficher le navigateur (non headless).")
    args = ap.parse_args()

    # Nom de sortie par défaut
    vid = re.search(r"[?&]v=([^&]+)", args.url)
    vid = (vid.group(1) if vid else "video").strip()
    out_csv = args.out or f"youtube_comments_{vid}.csv"

    df = scrape(
        url=args.url,
        out_csv=out_csv,
        max_comments=args.max,
        include_replies=args.replies,
        headless=not args.headed
    )
    print(f"✅ Exporté {len(df)} lignes → {out_csv}")
