from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class ScrapedPage:
    title: str
    content: str
    source: str
    scraped_at: str
    content_length: int
    page_type: str


UI_ONLY_TEXT = {
    "show more",
    "view profile",
    "read more",
    "click here",
    "view team",
    "read more"
}


PRESERVE_SECTION_KEYWORDS = {
    "faculty",
    "admissions",
    "admission",
    "notice",
    "notices",
    "infrastructure",
    "contact",
    "contact us",
}


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_output_filename(url: str) -> str:
    """Create a stable, filesystem-safe filename for a URL."""
    parsed = urlparse(url)

    # Human-readable slug (best effort)
    netloc = (parsed.netloc or "unknown-host").replace(":", "_")
    path = (parsed.path or "/").strip("/")
    if not path:
        path = "root"
    path = path.replace("/", "_")

    # Short hash to avoid collisions and keep filenames manageable
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]

    return f"{netloc}__{path}__{url_hash}.json"


def _collapse_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_ui_only_text(text: str) -> bool:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return False
    return normalized in UI_ONLY_TEXT


def _looks_like_navigation_container(tag) -> bool:
    tag_id = (tag.get("id") or "").lower()
    classes = " ".join(tag.get("class") or []).lower()
    haystack = f"{tag_id} {classes}"
    return any(k in haystack for k in ["nav", "navbar", "menu", "breadcrumb", "breadcrumbs"])


def _remove_noise(soup: BeautifulSoup) -> None:
    """Remove common non-content elements.

    This is heuristic-based and intentionally simple.
    """

    # Always remove these
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Often-not-useful layout/navigation blocks
    for tag in soup.select(
        "nav, form, "
        ".nav, #nav, .navbar, #navbar, "
        ".menu, #menu, .breadcrumb, #breadcrumb, "
        ".sidebar, #sidebar"
    ):
        tag.decompose()


def _remove_ui_only_elements(container) -> None:
    """Remove common UI-only elements such as "Read More" buttons."""
    for tag in container.find_all(["a", "button", "span", "div"]):
        txt = tag.get_text(strip=True)
        if _is_ui_only_text(txt):
            tag.decompose()


def _table_to_text(table) -> str:
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            cell_text = _normalize_text(cell.get_text(separator=" ", strip=True))
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _convert_tables_to_text(soup: BeautifulSoup, container) -> None:
    """Replace HTML tables with a simple text representation to preserve content."""
    for table in list(container.find_all("table")):
        table_text = _table_to_text(table)
        replacement = soup.new_tag("pre")
        replacement.string = table_text or ""
        table.replace_with(replacement)


def _pick_main_container(soup: BeautifulSoup):
    """Try to pick the most likely container with meaningful text."""
    for selector in ["main", "article", "#content", ".content", "#main", ".main"]:
        el = soup.select_one(selector)
        if el is not None:
            return el
    return soup.body or soup


def extract_title(soup: BeautifulSoup) -> str:
    """Extract the best page title using simple priority rules.

    Priority (Phase 1.5):
    1) First meaningful H1
    2) Page banner / page heading (common class/id patterns)
    3) First meaningful H2
    4) HTML <title>
    """

    def is_meaningful_heading_text(text: str) -> bool:
        normalized = _normalize_text(text)
        if not normalized:
            return False
        if _is_ui_only_text(normalized):
            return False
        # Avoid very short generic headings
        return len(normalized) >= 3

    # 1) First meaningful H1 (avoid nav/menu headings)
    for h1 in soup.find_all("h1"):
        text = h1.get_text(strip=True)
        if not is_meaningful_heading_text(text):
            continue
        if h1.find_parent("nav") is not None:
            continue
        if any(_looks_like_navigation_container(p) for p in h1.parents if getattr(p, "get", None)):
            continue
        return _normalize_text(text)

    # 2) Page banner / page heading
    banner_selectors = [
        ".page-title",
        ".page_heading",
        ".page-heading",
        ".page-header",
        "#page-title",
        "#page_heading",
        "#page-heading",
        ".banner",
        ".inner-banner",
        ".hero",
    ]
    for sel in banner_selectors:
        el = soup.select_one(sel)
        if el is None:
            continue
        for tag_name in ["h1", "h2"]:
            heading = el.find(tag_name)
            if heading is not None:
                txt = heading.get_text(strip=True)
                if is_meaningful_heading_text(txt):
                    return _normalize_text(txt)
        txt = el.get_text(strip=True)
        if is_meaningful_heading_text(txt):
            return _normalize_text(txt)

    # 3) First meaningful H2
    for h2 in soup.find_all("h2"):
        txt = h2.get_text(strip=True)
        if is_meaningful_heading_text(txt):
            return _normalize_text(txt)

    # 4) HTML <title>
    if soup.title and soup.title.get_text(strip=True):
        return _normalize_text(soup.title.get_text(strip=True))

    return ""


def extract_meaningful_text(soup: BeautifulSoup) -> str:
    _remove_noise(soup)

    container = _pick_main_container(soup)

    # Remove common UI-only button/link text
    _remove_ui_only_elements(container)

    # Preserve tables by converting them to text before extraction
    _convert_tables_to_text(soup, container)

    text = container.get_text(separator="\n", strip=True)
    text = _collapse_whitespace(text)

    # Final pass: drop leftover UI-only lines (common on CMS sites)
    filtered_lines: list[str] = []
    for line in text.splitlines():
        if _is_ui_only_text(line):
            continue
        filtered_lines.append(line)
    text = "\n".join(filtered_lines)

    return text


def fetch_html(url: str, timeout_s: int = 20, user_agent: Optional[str] = None) -> str:
    headers = {
        "User-Agent": user_agent
        or "campus-ai-agent/phase1.5-scraper (+https://github.com/GouthamA15/campus-ai-agent)"
    }

    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()

    # requests usually guesses encoding well; keep it explicit.
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def scrape_url(url: str) -> ScrapedPage:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup)
    content = extract_meaningful_text(soup)
    content_length = len(content)
    page_type = classify_page_type(url=url, title=title, content=content)

    return ScrapedPage(
        title=title,
        content=content,
        source=url,
        scraped_at=_utc_iso_now(),
        content_length=content_length,
        page_type=page_type,
    )


def classify_page_type(url: str, title: str, content: str) -> str:
    """Classify a page using simple, beginner-friendly rules.

    This is intentionally heuristic-based (no ML) for Phase 1.5.
    """
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    combined = f"{url} {title} {content[:2000]}".lower()

    if path in ("", "/") or path.endswith(("/index.html", "/index.php")):
        return "homepage"
    if any(k in combined for k in ["/department", "department", "/dept", "faculty", "school of"]):
        return "department"
    if "alumni" in combined:
        return "alumni"
    if any(k in combined for k in ["notice", "notices", "announcement", "circular"]):
        return "notice"

    return "general"


def save_json(scraped: ScrapedPage, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / _make_output_filename(scraped.source)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(scraped), f, ensure_ascii=False, indent=2)

    return out_path


def _configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger()
    logger.setLevel(log_level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)

    # File
    fh = logging.FileHandler(output_dir / "scrape.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)

    # Avoid duplicate handlers if called multiple times
    logger.handlers = []
    logger.addHandler(ch)
    logger.addHandler(fh)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape a single web page to JSON (Phase 1.5).")
    parser.add_argument("--url", required=True, help="Page URL to scrape")
    parser.add_argument(
        "--out",
        default=str(Path("data") / "processed"),
        help="Output directory for JSON (default: data/processed)",
    )
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    output_dir = Path(args.out)
    _configure_logging(output_dir, verbose=args.verbose)

    log = logging.getLogger(__name__)

    try:
        log.info("Scraping URL: %s", args.url)
        html = fetch_html(args.url, timeout_s=args.timeout)
        soup = BeautifulSoup(html, "html.parser")

        title = extract_title(soup)
        content = extract_meaningful_text(soup)
        content_length = len(content)
        page_type = classify_page_type(url=args.url, title=title, content=content)

        scraped = ScrapedPage(
            title=title,
            content=content,
            source=args.url,
            scraped_at=_utc_iso_now(),
            content_length=content_length,
            page_type=page_type,
        )

        out_path = save_json(scraped, output_dir)

        # Required extraction statistics
        print("URL:", args.url)
        print("Title:", scraped.title)
        print("Page type:", scraped.page_type)
        print("Content length:", len(scraped.content))

        log.info("Saved JSON: %s", out_path)
        log.info("Title: %s", scraped.title)
        log.info("Page type: %s", scraped.page_type)
        log.info("Content length: %d", scraped.content_length)

        return 0

    except requests.exceptions.RequestException as e:
        log.exception("Request failed")
        print("ERROR: Request failed:", str(e))
        return 2
    except Exception as e:
        log.exception("Unexpected error")
        print("ERROR: Unexpected error:", str(e))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
