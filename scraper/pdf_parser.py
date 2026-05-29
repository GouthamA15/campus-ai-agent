from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pdfplumber


@dataclass
class PdfPage:
    page_number: int
    content: str


@dataclass
class ExtractedPdf:
    title: str
    source_pdf: str
    page_count: int
    extracted_at: str
    content_length: int
    pages: list[PdfPage]
    full_content: str
    cleanup_applied: Optional[bool] = None
    removed_repeated_lines: Optional[list[str]] = None


@dataclass
class PdfParseSummary:
    pdfs_processed: int = 0
    pdfs_successful: int = 0
    pdfs_failed: int = 0


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(output_dir / "pdf_parse.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root.handlers = []
    root.addHandler(ch)
    root.addHandler(fh)


def _table_to_text(table: list[list[Optional[str]]]) -> str:
    """Best-effort table rendering as plain text."""
    lines: list[str] = []
    for row in table:
        if not row:
            continue
        cells = [(c or "").strip() for c in row]
        # Remove totally empty rows
        if not any(cells):
            continue
        lines.append(" | ".join(cells))
    return "\n".join(lines)


_UNIT_LINE_RE = re.compile(r"^\s*UNIT\s*[-–—]?\s*([IVXLC]+|\d+)\b", re.IGNORECASE)
# Course codes like BSC 101, CSE-302, etc. Keep digits limited so PIN codes (6 digits)
# and location lines don't get misclassified as course codes.
_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,6}\s*[-/]?\s*\d{2,4}[A-Z]{0,3}\b")
_PINCODE_RE = re.compile(r"\b\d{6}\b")
_ROMAN_NUMERAL_RE = re.compile(
    r"^(?=[IVXLCDM]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)


def _is_table_line(line: str) -> bool:
    upper = line.strip().upper()
    return upper.startswith("[TABLE") or "|" in line


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _canonicalize_for_repeated_match(line: str) -> str:
    """Canonical key for matching repeated banner lines.

    Deterministic normalization only (no fuzzy libs):
    - lowercase
    - remove punctuation/commas/dashes
    - remove extra spaces
    - remove academic years/date ranges by removing numeric tokens
    - remove postal codes (numeric tokens)
    - remove standalone roman numerals (e.g., VI)
    """

    s = line.lower().replace("&", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t]

    # Merge common degree tokens: b tech -> btech, m tech -> mtech
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] in {"b", "m"} and tokens[i + 1] == "tech":
            merged.append(tokens[i] + "tech")
            i += 2
            continue
        merged.append(tokens[i])
        i += 1

    stopwords = {
        "of",
        "the",
        "and",
        "for",
        "to",
        "in",
        "on",
        "at",
        "ku",
    }

    cleaned: list[str] = []
    for tok in merged:
        if tok in stopwords:
            continue
        if any(ch.isdigit() for ch in tok):
            # Drops years, date ranges, pin codes, etc.
            continue
        if _ROMAN_NUMERAL_RE.match(tok):
            continue
        if len(tok) <= 1:
            continue
        cleaned.append(tok)

    return " ".join(cleaned).strip()


def _should_never_remove_line(line: str) -> bool:
    """Conservative protections to avoid deleting meaningful academic content."""
    if not line.strip():
        return True

    upper = line.upper()

    # Never touch unit markers
    if _UNIT_LINE_RE.match(line):
        return True

    # Protected patterns (explicit requirement)
    protected_substrings = (
        "UNIT",
        "MODULE",
        "EXPERIMENT",
        "WEEK",
        "COURSE CODE",
        "TABLE DATA",
    )
    if any(s in upper for s in protected_substrings):
        return True

    # Never touch course codes like BSC 101, CSE-302, etc.
    if _COURSE_CODE_RE.search(upper):
        return True

    # Never touch table blocks (we add [TABLE n] markers)
    if _is_table_line(line):
        return True

    # Preserve key academic headings
    important_keywords = (
        "TIMETABLE",
        "TIME TABLE",
        "EXAM",
        "NOTIFICATION",
        "CIRCULAR",
    )
    if any(k in upper for k in important_keywords):
        return True

    return False


def _looks_like_boilerplate(line: str) -> bool:
    """Heuristic: likely to be a repeated banner/header/footer line."""
    upper = line.upper()

    # Common institution/location banners
    boilerplate_keywords = (
        "UNIVERSITY",
        "FACULTY",
        "DEPARTMENT",
        "COLLEGE",
        "INSTITUTE",
        "CAMPUS",
        "WARANGAL",
        "TELANGANA",
    )
    if any(k in upper for k in boilerplate_keywords):
        return True

    # Academic banners that are often repeated but not useful for embeddings
    academic_banner_keywords = (
        "ACADEMIC YEAR",
        "SEMESTER",
        "CURRICULUM",
        "SYLLABUS",
        "B TECH",
        "B.TECH",
        "B. TECH",
        "BTECH",
        "M TECH",
        "M.TECH",
        "M. TECH",
        "MTECH",
        "BACHELOR",
        "MASTER",
    )
    if any(k in upper for k in academic_banner_keywords):
        return True

    # PIN code / postal code lines
    if _PINCODE_RE.search(upper):
        return True

    return False


def _looks_like_course_or_subject_title(line: str) -> bool:
    """Heuristic protection: likely meaningful academic title (keep it).

    This helps avoid removing repeated course/subject titles that may appear at the
    top of many pages.
    """

    if not line.strip():
        return False

    if _looks_like_boilerplate(line):
        return False

    # Titles are usually short-ish and mostly alphabetic
    normalized = _normalize_line(line)
    if len(normalized) < 4 or len(normalized) > 80:
        return False

    if any(ch.isdigit() for ch in normalized):
        return False

    if not re.fullmatch(r"[A-Za-z&()\[\]/\-.,' ]+", normalized):
        return False

    return True


def _header_footer_window_indexes(
    content: str,
    header_scan_lines: int = 10,
    footer_scan_lines: int = 5,
) -> set[int]:
    """Indexes for the first/last N non-empty, non-table lines."""

    lines = content.splitlines()
    indexes: list[int] = []
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if _is_table_line(stripped):
            continue
        indexes.append(idx)

    header = indexes[:header_scan_lines]
    footer = indexes[-footer_scan_lines:] if footer_scan_lines > 0 else []
    return set(header) | set(footer)


def _iter_header_footer_candidates(
    content: str,
    header_scan_lines: int = 10,
    footer_scan_lines: int = 5,
) -> list[tuple[str, str]]:
    """Return (canonical_key, raw) lines from header/footer windows for counting."""

    lines = content.splitlines()
    window = _header_footer_window_indexes(
        content,
        header_scan_lines=header_scan_lines,
        footer_scan_lines=footer_scan_lines,
    )

    candidates: list[tuple[str, str]] = []
    for idx, raw in enumerate(lines):
        if idx not in window:
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        if _is_table_line(stripped):
            continue
        if _should_never_remove_line(stripped):
            continue

        key = _canonicalize_for_repeated_match(stripped)
        if not key:
            continue
        candidates.append((key, stripped))

    return candidates


def cleanup_repeated_headers_footers(
    extracted: ExtractedPdf,
    threshold: float = 0.5,
) -> ExtractedPdf:
    """Remove repeated header/footer lines that appear across many pages.

    A line is considered repeated if it appears on >= threshold of pages in the
    header/footer zones.
    """

    log = logging.getLogger(__name__)

    if not extracted.pages:
        extracted.cleanup_applied = True
        extracted.removed_repeated_lines = []
        return extracted

    page_count = extracted.page_count or len(extracted.pages)
    if page_count <= 1:
        extracted.cleanup_applied = True
        extracted.removed_repeated_lines = []
        return extracted

    # Accept 0.5 or 50 (percent)
    if threshold > 1:
        threshold = threshold / 100.0
    threshold = max(0.0, min(1.0, threshold))

    min_pages = max(2, int(math.ceil(threshold * page_count)))

    key_counts: dict[str, int] = {}
    raw_variant_counts: dict[str, dict[str, int]] = {}

    for p in extracted.pages:
        candidates = _iter_header_footer_candidates(p.content, header_scan_lines=10, footer_scan_lines=5)
        page_keys = {k for (k, _raw) in candidates}  # page-level dedupe
        for k in page_keys:
            key_counts[k] = key_counts.get(k, 0) + 1

        for k, raw in candidates:
            raw_variant_counts.setdefault(k, {})
            raw_variant_counts[k][raw] = raw_variant_counts[k].get(raw, 0) + 1

    repeated_keys = {k for k, c in key_counts.items() if c >= min_pages}

    def pick_rep(k: str) -> str:
        variants = raw_variant_counts.get(k, {})
        if not variants:
            return k

        max_count = max(variants.values())
        # Prefer a "clean" representative, but keep it reasonably common.
        min_count = max(1, int(math.ceil(max_count * 0.5)))
        pool = [raw for raw, c in variants.items() if c >= min_count]
        if not pool:
            pool = list(variants.keys())

        def score(raw: str) -> tuple[int, int, int, str]:
            digits = sum(ch.isdigit() for ch in raw)
            punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in raw)
            # Lower is better; final tie-break is lexicographic for determinism
            return (digits, punct, len(raw), raw)

        return min(pool, key=score)

    removable_keys: set[str] = set()
    for k in repeated_keys:
        rep = pick_rep(k)
        if _should_never_remove_line(rep):
            continue
        if _looks_like_course_or_subject_title(rep) and not _looks_like_boilerplate(rep):
            continue
        removable_keys.add(k)

    if not removable_keys:
        extracted.cleanup_applied = True
        extracted.removed_repeated_lines = []
        return extracted

    removed_pages: dict[str, int] = {k: 0 for k in removable_keys}
    cleaned_pages: list[PdfPage] = []

    for page in extracted.pages:
        lines = page.content.splitlines()
        window = _header_footer_window_indexes(page.content, header_scan_lines=10, footer_scan_lines=5)
        page_removed: set[str] = set()

        new_lines: list[str] = []
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            if idx in window and stripped and not _is_table_line(stripped) and not _should_never_remove_line(stripped):
                key = _canonicalize_for_repeated_match(stripped)
                if key and key in removable_keys:
                    page_removed.add(key)
                    continue
            new_lines.append(raw)

        for k in page_removed:
            removed_pages[k] += 1

        new_content = "\n".join(new_lines).strip()
        cleaned_pages.append(PdfPage(page_number=page.page_number, content=new_content))

    sorted_keys = sorted(removable_keys, key=lambda k: (-key_counts.get(k, 0), k))
    log.info("Detected repeated headers/footers (normalized, >= %d/%d pages):", min_pages, page_count)
    for k in sorted_keys:
        log.info("- %s", k)
    log.info("Removed from pages:")
    for k in sorted_keys:
        log.info("- %s: %d pages", k, removed_pages.get(k, 0))

    removed_lines = [pick_rep(k) for k in sorted_keys if removed_pages.get(k, 0) > 0]

    extracted.pages = cleaned_pages
    extracted.full_content = "\n\n".join(p.content for p in cleaned_pages).strip()
    extracted.content_length = sum(len(p.content) for p in cleaned_pages)
    extracted.cleanup_applied = True
    extracted.removed_repeated_lines = removed_lines

    return extracted


def _to_json_dict(extracted: ExtractedPdf) -> dict:
    data = asdict(extracted)
    # Preserve existing JSON shape when cleanup is not enabled
    if data.get("cleanup_applied") is None:
        data.pop("cleanup_applied", None)
    if data.get("removed_repeated_lines") is None:
        data.pop("removed_repeated_lines", None)
    return data


def extract_pdf_text(pdf_path: Path) -> ExtractedPdf:
    """Extract text from a PDF (no OCR).

    Notes:
    - Uses pdfplumber (pdfminer.six under the hood).
    - Scanned/image-only PDFs often produce empty text; we log a warning upstream.
    - Saves output even if some pages are empty.
    """

    extracted_at = _utc_iso_now()

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_count = len(pdf.pages)

        # Title: try PDF metadata first, then filename
        meta_title = None
        try:
            meta_title = (pdf.metadata or {}).get("Title")
        except Exception:
            meta_title = None

        title = (meta_title or "").strip() or pdf_path.stem

        pages: list[PdfPage] = []

        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()

            page_parts: list[str] = []
            if page_text:
                page_parts.append(page_text)

            # Tables (best effort)
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for t_index, table in enumerate(tables, start=1):
                table_text = _table_to_text(table)
                if not table_text.strip():
                    continue

                # Best effort: keep tables within the same page block
                page_parts.append(f"[TABLE {t_index}]\n{table_text}")

            page_content = "\n\n".join(page_parts).strip()
            pages.append(PdfPage(page_number=i, content=page_content))

        full_content = "\n\n".join(p.content for p in pages).strip()

        # content_length: total character count across all pages
        content_length = sum(len(p.content) for p in pages)

        # Fallback title: first non-empty line of extracted content
        if title == pdf_path.stem:
            for p in pages:
                if not p.content:
                    continue
                for line in p.content.splitlines():
                    line = line.strip()
                    if line:
                        title = line[:200]
                        break
                if title != pdf_path.stem:
                    break

        return ExtractedPdf(
            title=title,
            source_pdf=str(pdf_path.as_posix()),
            page_count=page_count,
            extracted_at=extracted_at,
            content_length=content_length,
            pages=pages,
            full_content=full_content,
        )


def save_extracted_pdf(extracted: ExtractedPdf, out_dir: Path, pdf_path: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stable filename: original_pdf_name.json (e.g., rules.pdf -> rules.json)
    out_path = out_dir / f"{pdf_path.stem}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_dict(extracted), f, ensure_ascii=False, indent=2)

    return out_path


def parse_one(
    pdf_path: Path,
    out_dir: Path,
    cleanup: bool = False,
    cleanup_threshold: float = 0.5,
) -> bool:
    log = logging.getLogger(__name__)

    try:
        log.info("Processing PDF: %s", pdf_path)
        extracted = extract_pdf_text(pdf_path)

        if cleanup:
            extracted = cleanup_repeated_headers_footers(extracted, threshold=cleanup_threshold)
        log.info("Page count: %d", extracted.page_count)
        log.info("Extracted text length: %d", extracted.content_length)

        if extracted.content_length == 0:
            log.warning(
                "No extractable text found (possibly scanned/image-only PDF, no OCR in Phase 4): %s",
                pdf_path,
            )

        out_path = save_extracted_pdf(extracted, out_dir=out_dir, pdf_path=pdf_path)
        log.info("Saved JSON: %s", out_path)
        return True

    except Exception:
        log.exception("Failed to parse PDF: %s", pdf_path)
        return False


def _iter_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4: Extract text from PDFs (no OCR).")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Parse a single PDF file")
    group.add_argument("--all", action="store_true", help="Parse all PDFs in data/pdfs/")

    parser.add_argument(
        "--pdf-dir",
        default=str(Path("data") / "pdfs"),
        help="PDF input directory (default: data/pdfs)",
    )
    parser.add_argument(
        "--out",
        default=str(Path("data") / "pdf_text"),
        help="Output directory for extracted JSON (default: data/pdf_text)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove repeated header/footer lines across pages (recommended for university PDFs)",
    )
    parser.add_argument(
        "--cleanup-threshold",
        type=float,
        default=0.5,
        help="Repeated-line threshold as a fraction (0.5) or percent (50) (default: 0.5)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out)

    _configure_logging(out_dir, verbose=args.verbose)

    summary = PdfParseSummary()

    if args.file:
        pdf_path = Path(args.file)
        summary.pdfs_processed = 1
        ok = parse_one(
            pdf_path,
            out_dir=out_dir,
            cleanup=bool(args.cleanup),
            cleanup_threshold=float(args.cleanup_threshold),
        )
        summary.pdfs_successful = 1 if ok else 0
        summary.pdfs_failed = 0 if ok else 1

    else:
        pdfs = _iter_pdfs(pdf_dir)
        log = logging.getLogger(__name__)
        log.info("Found %d PDFs in %s", len(pdfs), pdf_dir)

        for pdf_path in pdfs:
            summary.pdfs_processed += 1
            ok = parse_one(
                pdf_path,
                out_dir=out_dir,
                cleanup=bool(args.cleanup),
                cleanup_threshold=float(args.cleanup_threshold),
            )
            if ok:
                summary.pdfs_successful += 1
            else:
                summary.pdfs_failed += 1

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
