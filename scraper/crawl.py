from __future__ import annotations

import argparse
import json
import logging
import re
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import ParseResult, urldefrag, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from .scrape import ScrapedPage, classify_page_type, extract_meaningful_text, extract_title, fetch_html, save_json


ALLOWED_DOMAIN = "kucet.ac.in"

# Schemes we don't want to crawl
BLOCKED_SCHEMES = ("mailto", "tel", "javascript")

# Social / external domains to ignore even if they show up in pages
BLOCKED_DOMAINS = {
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "linkedin.com",
    "www.linkedin.com",
    "youtube.com",
    "www.youtube.com",
    "t.me",
    "wa.me",
    "whatsapp.com",
    "www.whatsapp.com",
}

# We do NOT crawl binary assets in Phase 2 (we only crawl HTML pages).
# PDFs may still be discovered as links, but we don't fetch them as pages here.
SKIP_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".mp4",
    ".mp3",
    ".zip",
    ".rar",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}


PDF_URL_REGEX = re.compile(r"\.pdf(?:$|[?#&])", re.IGNORECASE)


@dataclass
class CrawlSummary:
    pages_crawled: int = 0
    pages_saved: int = 0
    links_discovered: int = 0
    failed_pages: int = 0
    max_depth_reached: int = 0


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger()
    logger.setLevel(log_level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(output_dir / "crawl.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)

    logger.handlers = []
    logger.addHandler(ch)
    logger.addHandler(fh)


def _is_allowed_domain(netloc: str) -> bool:
    netloc = (netloc or "").lower()
    if not netloc:
        return False
    return netloc == ALLOWED_DOMAIN or netloc.endswith("." + ALLOWED_DOMAIN)


def _canonicalize_netloc(netloc: str) -> str:
    """Best-effort canonicalization to reduce duplicates."""
    netloc = (netloc or "").strip().lower()

    # Treat www.kucet.ac.in as the same site as kucet.ac.in
    if netloc.startswith("www."):
        netloc = netloc[len("www.") :]

    # Drop default ports when present (host:80, host:443)
    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        if port in {"80", "443"}:
            netloc = host

    return netloc


def _has_skipped_extension(path: str) -> bool:
    path_lower = (path or "").lower()
    return any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS)


def normalize_url(url: str, base_url: str, preferred_scheme: str) -> Optional[str]:
    """Normalize a possibly-relative URL into an absolute URL.

    Rules:
    - Makes it absolute using the page URL as base
    - Drops URL fragments (#section)
    - Skips mailto:, tel:, javascript:
    - Forces scheme to preferred_scheme for internal URLs (avoids http/https duplicates)
    """

    if not url:
        return None

    url = url.strip()
    if not url or url.startswith("#"):
        return None

    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme.lower() in BLOCKED_SCHEMES:
        return None

    absolute = urljoin(base_url, url)
    absolute, _frag = urldefrag(absolute)

    parsed_abs = urlparse(absolute)

    if parsed_abs.scheme.lower() not in ("http", "https"):
        return None

    netloc = _canonicalize_netloc(parsed_abs.netloc)
    if netloc in BLOCKED_DOMAINS:
        return None

    # Domain restriction (only keep kucet.ac.in)
    if not _is_allowed_domain(netloc):
        return None

    if _has_skipped_extension(parsed_abs.path):
        return None

    # Normalize scheme (keep crawling consistent)
    normalized = ParseResult(
        scheme=preferred_scheme,
        netloc=netloc,
        path=parsed_abs.path or "/",
        params="",
        query=parsed_abs.query or "",
        fragment="",
    )

    # Normalize trailing slash (avoid duplicates like /about and /about/)
    if normalized.path != "/" and normalized.path.endswith("/"):
        normalized = normalized._replace(path=normalized.path.rstrip("/"))

    return urlunparse(normalized)


def looks_like_pdf_url(url: str) -> bool:
    if not url:
        return False
    return bool(PDF_URL_REGEX.search(url.strip()))


def discover_internal_links(soup: BeautifulSoup, page_url: str, preferred_scheme: str) -> set[str]:
    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        normalized = normalize_url(a.get("href"), base_url=page_url, preferred_scheme=preferred_scheme)
        if normalized:
            links.add(normalized)
    return links


def crawl(
    start_url: str,
    output_dir: Path,
    max_pages: int = 200,
    max_depth: int = 5,
    timeout_s: int = 20,
    on_page: Optional[Callable[[str, BeautifulSoup], None]] = None,
    show_links: bool = False,
) -> CrawlSummary:
    log = logging.getLogger(__name__)

    preferred_scheme = urlparse(start_url).scheme or "http"

    visited_urls: set[str] = set()
    discovered_links: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    start = normalize_url(start_url, base_url=start_url, preferred_scheme=preferred_scheme) or start_url
    visited_urls.add(start)
    queue.append((start, 0))

    summary = CrawlSummary()

    while queue and summary.pages_crawled < max_pages:
        current_url, depth = queue.popleft()
        summary.pages_crawled += 1

        if depth > summary.max_depth_reached:
            summary.max_depth_reached = depth

        log.info(
            "Crawling (%d/%d) | depth=%d/%d | queue=%d | url=%s",
            summary.pages_crawled,
            max_pages,
            depth,
            max_depth,
            len(queue),
            current_url,
        )

        try:
            # Fetch HTML and build soup
            html = fetch_html(current_url, timeout_s=timeout_s)
            soup = BeautifulSoup(html, "html.parser")

            # IMPORTANT: Discover links BEFORE content extraction.
            # Content extraction removes nav/menu elements (noise removal), which would hide
            # department/dropdown links if we discovered links afterwards.
            discovered = discover_internal_links(soup, page_url=current_url, preferred_scheme=preferred_scheme)
            log.debug("Found %d internal links on page", len(discovered))

            if show_links:
                print(f"\n=== Links discovered on: {current_url} (count={len(discovered)}) ===")
                for link in sorted(discovered):
                    print(link)

            # Optional hook for additional discovery (e.g., PDFs)
            # Keep this before content extraction so the hook can also see nav/footer links.
            if on_page is not None:
                try:
                    on_page(current_url, soup)
                except Exception:
                    log.exception("on_page hook failed for: %s", current_url)

            # Store discovered links (dedup) and expand queue (BFS)
            new_pages_enqueued = 0
            new_links_logged = 0
            for link in discovered:
                # Count unique discovered links (including PDFs)
                if link not in discovered_links:
                    discovered_links.add(link)
                    summary.links_discovered += 1

                # Do not crawl PDFs as HTML pages in Phase 2
                if looks_like_pdf_url(link):
                    continue

                if link in visited_urls:
                    continue

                if depth >= max_depth:
                    continue

                visited_urls.add(link)
                queue.append((link, depth + 1))
                new_pages_enqueued += 1

                # Debugging: log the actual newly enqueued URLs (without spamming INFO)
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("Enqueued new URL (depth=%d): %s", depth + 1, link)
                    new_links_logged += 1

            log.info(
                "Links | found=%d | newly_enqueued=%d | queue=%d",
                len(discovered),
                new_pages_enqueued,
                len(queue),
            )

            # Extract page content AFTER link discovery (content cleaning is unchanged)
            title = extract_title(soup)
            content = extract_meaningful_text(soup)
            content_length = len(content)
            page_type = classify_page_type(url=current_url, title=title, content=content)

            scraped = ScrapedPage(
                title=title,
                content=content,
                source=current_url,
                scraped_at=_utc_iso_now(),
                content_length=content_length,
                page_type=page_type,
            )

            out_path = save_json(scraped, output_dir)
            summary.pages_saved += 1

            log.info("Saved: %s", out_path)
            log.info("Title: %s", title)
            log.info("Page type: %s", page_type)
            log.info("Content length: %d", content_length)

        except Exception:
            summary.failed_pages += 1
            log.exception("Failed to crawl: %s", current_url)

    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deep BFS crawler for kucet.ac.in (Phase 2).")
    parser.add_argument(
        "--start-url",
        default="http://kucet.ac.in/",
        help="Starting URL (default: http://kucet.ac.in/)",
    )
    parser.add_argument(
        "--out",
        default=str(Path("data") / "processed"),
        help="Output directory for JSON (default: data/processed)",
    )
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum pages to crawl")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum crawl depth")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--show-links",
        action="store_true",
        help="Print all discovered internal URLs per page (debug option)",
    )

    args = parser.parse_args(argv)

    output_dir = Path(args.out)
    _configure_logging(output_dir, verbose=args.verbose)

    log = logging.getLogger(__name__)
    log.info("Starting crawl: %s", args.start_url)
    log.info("Allowed domain: %s", ALLOWED_DOMAIN)
    log.info("Max pages: %d", args.max_pages)
    log.info("Max depth: %d", args.max_depth)

    summary = crawl(
        start_url=args.start_url,
        output_dir=output_dir,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        timeout_s=args.timeout,
        show_links=args.show_links,
    )

    summary_json = asdict(summary)

    print(json.dumps(summary_json, indent=2))
    log.info("Crawl summary: %s", summary_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
