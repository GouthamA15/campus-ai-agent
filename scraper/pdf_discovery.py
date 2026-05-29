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
from urllib.parse import ParseResult, urldefrag, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from .crawl import ALLOWED_DOMAIN, BLOCKED_DOMAINS, BLOCKED_SCHEMES, _is_allowed_domain, crawl


PDF_URL_REGEX = re.compile(r"\.pdf(?:$|[?#&])", re.IGNORECASE)


@dataclass
class PdfMetadata:
    title: str
    pdf_url: str
    discovered_from: str
    discovered_at: str


@dataclass
class PdfDiscoverySummary:
    pages_crawled: int = 0
    pdfs_discovered: int = 0
    pdfs_downloaded: int = 0
    pdfs_skipped: int = 0
    failed_pdfs: int = 0


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_dir / "pdf_discovery.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root.handlers = []
    root.addHandler(ch)
    root.addHandler(fh)


def looks_like_pdf_url(url: str) -> bool:
    """Detect PDF links.

    Handles:
    - direct links ending in .pdf
    - links with query params containing a .pdf filename
    """
    if not url:
        return False
    return bool(PDF_URL_REGEX.search(url.strip())) or url.strip().lower().endswith(".pdf")


def normalize_pdf_url(url: str, base_url: str, preferred_scheme: str) -> Optional[str]:
    """Normalize a possibly-relative PDF URL into an absolute URL.

    Rules:
    - Makes it absolute using the page URL as base
    - Drops fragments (#...)
    - Skips mailto:, tel:, javascript:
    - Skips social/external domains
    - Keeps only kucet.ac.in domain (same restriction as crawl)
    - Forces scheme to preferred_scheme to avoid http/https duplicates
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

    netloc = (parsed_abs.netloc or "").lower()
    if netloc in BLOCKED_DOMAINS:
        return None

    if not _is_allowed_domain(netloc):
        return None

    normalized = ParseResult(
        scheme=preferred_scheme,
        netloc=netloc,
        path=parsed_abs.path or "/",
        params="",
        query=parsed_abs.query or "",
        fragment="",
    )

    return urlunparse(normalized)


def _safe_slug(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "pdf"


def _pdf_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _pdf_metadata_filename(pdf_url: str, title: str) -> str:
    slug = _safe_slug(title)[:60]
    return f"{slug}__{_pdf_hash(pdf_url)}.json"


def _best_pdf_filename(pdf_url: str) -> str:
    """Pick a reasonable filename for the downloaded PDF."""
    # Try to find a real *.pdf name in the URL (path or query)
    match = re.search(r"([^/=?&]+\.pdf)(?:$|[?#&])", pdf_url, flags=re.IGNORECASE)
    if match:
        name = match.group(1)
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return name

    return f"document__{_pdf_hash(pdf_url)}.pdf"


def save_pdf_metadata(record: PdfMetadata, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _pdf_metadata_filename(record.pdf_url, record.title)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(record), f, ensure_ascii=False, indent=2)
    return out_path


def download_pdf(pdf_url: str, pdf_dir: Path, timeout_s: int = 30) -> tuple[Path, bool]:
    pdf_dir.mkdir(parents=True, exist_ok=True)

    target_name = _best_pdf_filename(pdf_url)
    target_path = pdf_dir / target_name

    # Avoid duplicate downloads
    if target_path.exists():
        return target_path, False

    headers = {
        "User-Agent": "campus-ai-agent/pdf-discovery (+https://github.com/GouthamA15/campus-ai-agent)"
    }

    with requests.get(pdf_url, headers=headers, timeout=timeout_s, stream=True) as resp:
        resp.raise_for_status()

        # Write to disk in chunks (beginner-friendly, safe for large PDFs)
        with target_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)

    return target_path, True


class PdfCollector:
    def __init__(
        self,
        metadata_dir: Path,
        pdf_dir: Path,
        preferred_scheme: str,
        timeout_s: int,
    ) -> None:
        self.metadata_dir = metadata_dir
        self.pdf_dir = pdf_dir
        self.preferred_scheme = preferred_scheme
        self.timeout_s = timeout_s

        self.discovered_pdfs: set[str] = set()

        self.pdfs_discovered = 0
        self.pdfs_downloaded = 0
        self.pdfs_skipped = 0
        self.failed_pdfs = 0

        self.log = logging.getLogger(self.__class__.__name__)

    def on_page(self, page_url: str, soup: BeautifulSoup) -> None:
        candidates: list[tuple[str, str]] = []

        # Most PDF links are <a href>
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href or not looks_like_pdf_url(href):
                continue
            title = a.get_text(strip=True) or "PDF"
            candidates.append((href, title))

        # Sometimes PDFs are embedded
        for tag_name, attr in [("iframe", "src"), ("embed", "src"), ("object", "data")]:
            for tag in soup.find_all(tag_name):
                raw = tag.get(attr)
                if not raw or not looks_like_pdf_url(raw):
                    continue
                title = (tag.get("title") or tag.get("aria-label") or "PDF").strip() or "PDF"
                candidates.append((raw, title))

        for raw_url, title in candidates:
            pdf_url = normalize_pdf_url(
                raw_url,
                base_url=page_url,
                preferred_scheme=self.preferred_scheme,
            )
            if not pdf_url:
                continue

            if pdf_url in self.discovered_pdfs:
                continue

            self.discovered_pdfs.add(pdf_url)
            self.pdfs_discovered += 1
            self.log.info("PDF discovered: %s (from %s)", pdf_url, page_url)

            record = PdfMetadata(
                title=title,
                pdf_url=pdf_url,
                discovered_from=page_url,
                discovered_at=_utc_iso_now(),
            )

            try:
                meta_path = save_pdf_metadata(record, self.metadata_dir)
                self.log.debug("Saved PDF metadata: %s", meta_path)

                pdf_path, downloaded = download_pdf(pdf_url, self.pdf_dir, timeout_s=self.timeout_s)
                if downloaded:
                    self.pdfs_downloaded += 1
                    self.log.info("PDF downloaded: %s", pdf_path)
                else:
                    self.pdfs_skipped += 1
                    self.log.info("PDF skipped (already exists): %s", pdf_path)

            except Exception:
                self.failed_pdfs += 1
                self.log.exception("Failed PDF download: %s", pdf_url)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: Discover and download PDFs while crawling.")
    parser.add_argument(
        "--start-url",
        default="http://kucet.ac.in/",
        help="Starting URL (default: http://kucet.ac.in/)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum HTML pages to crawl (default: 50)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum crawl depth (default: 5)",
    )
    parser.add_argument(
        "--processed-out",
        default=str(Path("data") / "processed"),
        help="Where to save scraped page JSON (default: data/processed)",
    )
    parser.add_argument(
        "--metadata-out",
        default=str(Path("data") / "pdf_metadata"),
        help="Where to save PDF metadata JSON (default: data/pdf_metadata)",
    )
    parser.add_argument(
        "--pdf-out",
        default=str(Path("data") / "pdfs"),
        help="Where to download PDFs (default: data/pdfs)",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--show-links",
        action="store_true",
        help="Print all discovered internal URLs per page (debug option)",
    )

    args = parser.parse_args(argv)

    processed_dir = Path(args.processed_out)
    metadata_dir = Path(args.metadata_out)
    pdf_dir = Path(args.pdf_out)

    _configure_logging(metadata_dir, verbose=args.verbose)

    log = logging.getLogger(__name__)
    log.info("Starting PDF discovery crawl: %s", args.start_url)
    log.info("Allowed domain: %s", ALLOWED_DOMAIN)
    log.info("Max pages: %d", args.max_pages)
    log.info("Max depth: %d", args.max_depth)

    preferred_scheme = urlparse(args.start_url).scheme or "http"

    collector = PdfCollector(
        metadata_dir=metadata_dir,
        pdf_dir=pdf_dir,
        preferred_scheme=preferred_scheme,
        timeout_s=args.timeout,
    )

    crawl_summary = crawl(
        start_url=args.start_url,
        output_dir=processed_dir,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        timeout_s=args.timeout,
        on_page=collector.on_page,
        show_links=args.show_links,
    )

    final_summary = PdfDiscoverySummary(
        pages_crawled=crawl_summary.pages_crawled,
        pdfs_discovered=collector.pdfs_discovered,
        pdfs_downloaded=collector.pdfs_downloaded,
        pdfs_skipped=collector.pdfs_skipped,
        failed_pdfs=collector.failed_pdfs,
    )

    print(json.dumps(asdict(final_summary), indent=2))
    log.info("Final summary: %s", asdict(final_summary))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
