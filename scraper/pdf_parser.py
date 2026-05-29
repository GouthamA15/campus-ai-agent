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
    # Generic university-wide metadata (document-neutral)
    document_type: str = "unknown"
    text_extracted: bool = True
    needs_ocr: bool = False
    ocr_status: Optional[str] = None
    cleanup_applied: Optional[bool] = None
    removed_repeated_lines: Optional[list[str]] = None


_DOC_TYPE_UNKNOWN = "unknown"


def _sample_text_for_classification(title: str, pages: list[PdfPage], max_chars: int = 20000) -> str:
    parts: list[str] = []
    if title:
        parts.append(title)

    # First 2 pages carry the strongest cues for most university docs.
    for p in pages[:2]:
        if p.content:
            parts.append(p.content)

    text = "\n\n".join(parts)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:max_chars]


def _classify_document_type(title: str, pages: list[PdfPage], content_length: int) -> str:
    """Deterministic keyword-based document classification (no AI, no fuzzy libs)."""

    if content_length <= 0:
        return _DOC_TYPE_UNKNOWN

    text = _sample_text_for_classification(title, pages)
    if not text:
        return _DOC_TYPE_UNKNOWN

    # Rules are ordered only to break ties deterministically.
    rules: list[tuple[str, list[tuple[re.Pattern[str], int]]]] = [
        (
            "scholarship_notice",
            [
                (re.compile(r"\bscholarship\b", re.I), 6),
                (re.compile(r"\b(fresh|renewal)\b", re.I), 3),
                (re.compile(r"\b(nsp|national scholarship portal)\b", re.I), 4),
                (re.compile(r"\beligib(ility|le)\b", re.I), 2),
            ],
        ),
        (
            "fee_circular",
            [
                (re.compile(r"\bfee\b", re.I), 4),
                (re.compile(r"\b(tuition|exam fee|college fee|hostel fee)\b", re.I), 4),
                (re.compile(r"\b(last date|due date|fine)\b", re.I), 2),
                (re.compile(r"\bremit\b|\bpay(?:ment)?\b", re.I), 2),
            ],
        ),
        (
            "quotation",
            [
                (re.compile(r"\bcall\s+for\s+quotations?\b", re.I), 8),
                (re.compile(r"\bquotations?\s+are\s+invited\b", re.I), 7),
                (re.compile(r"\bsealed\s+quotations?\b", re.I), 5),
                (re.compile(r"\bquotation\b", re.I), 3),
                (re.compile(r"\b(last date|due date)\b", re.I), 1),
            ],
        ),
        (
            "tender",
            [
                (re.compile(r"\btender\b", re.I), 7),
                (re.compile(r"\b(e-?tender|bid(?:ding)?|bidder)\b", re.I), 4),
                (re.compile(r"\b(nit|emd)\b", re.I), 4),
            ],
        ),
        (
            "office_order",
            [
                (re.compile(r"\boffice\s+order\b", re.I), 8),
                (re.compile(r"\bproceedings\b", re.I), 4),
                (re.compile(r"\bmemo(?:randum)?\b", re.I), 3),
                (re.compile(r"\border\s+no\b|\bdated\b", re.I), 2),
            ],
        ),
        (
            "placement_notification",
            [
                (re.compile(r"\bplacement\b", re.I), 6),
                (re.compile(r"\b(campus\s+drive|recruitment|interview)\b", re.I), 4),
                (re.compile(r"\bpackage\b|\bctc\b", re.I), 2),
            ],
        ),
        (
            "hostel_notice",
            [
                (re.compile(r"\bhostel\b", re.I), 7),
                (re.compile(r"\bwarden\b", re.I), 3),
                (re.compile(r"\bmess\b|\broom\b", re.I), 2),
            ],
        ),
        (
            "timetable",
            [
                (re.compile(r"\btime\s*table\b|\btimetable\b", re.I), 7),
                (re.compile(r"\b(fn|an)\b|\bforenoon\b|\bafternoon\b", re.I), 2),
                (re.compile(r"\bperiods?\b|\bsession\b", re.I), 1),
            ],
        ),
        (
            "exam_schedule",
            [
                (re.compile(r"\b(exam|examination)\b", re.I), 4),
                (re.compile(r"\bcontroller\s+of\s+examinations\b", re.I), 4),
                (re.compile(r"\bhall\s*ticket\b", re.I), 3),
                (re.compile(r"\bcommencement\b|\btheory\b|\bpractical\b", re.I), 2),
            ],
        ),
        (
            "circular",
            [
                (re.compile(r"\bcircular\b", re.I), 7),
                (re.compile(r"\bsub\s*:\b", re.I), 2),
                (re.compile(r"\bref\s*:\b", re.I), 2),
            ],
        ),
        (
            "notification",
            [
                (re.compile(r"\bnotification\b", re.I), 7),
                (re.compile(r"\bhereby\s+notified\b", re.I), 3),
            ],
        ),
        (
            "administrative_document",
            [
                (re.compile(r"\badministrative\b", re.I), 4),
                (re.compile(r"\bcommittee\b", re.I), 4),
                (re.compile(r"\bminutes\b", re.I), 4),
                (re.compile(r"\bmeeting\b", re.I), 2),
                (re.compile(r"\bagenda\b", re.I), 2),
            ],
        ),
        (
            "general_announcement",
            [
                (re.compile(r"\bannouncement\b", re.I), 6),
                (re.compile(r"\bpress\s+note\b", re.I), 6),
                (re.compile(r"\bnotice\b", re.I), 4),
                (re.compile(r"\bthis\s+is\s+to\s+inform\b", re.I), 2),
            ],
        ),
        (
            "academic_regulations",
            [
                (re.compile(r"\bregulations?\b", re.I), 6),
                (re.compile(r"\bord(inance|inances)\b", re.I), 4),
                (re.compile(r"\b(attendance|grading|credits?|evaluation)\b", re.I), 2),
            ],
        ),
        (
            "syllabus",
            [
                (re.compile(r"\bsyllabus\b", re.I), 6),
                (re.compile(r"\bunit\s*[-–—]?\s*(?:[ivxlcdm]+|\d+)\b", re.I), 4),
                (re.compile(r"\b(text\s*books?|references?)\b", re.I), 3),
                (re.compile(r"\b(course\s+outcomes?|co\d+)\b", re.I), 2),
            ],
        ),
    ]

    best_type = _DOC_TYPE_UNKNOWN
    best_score = 0

    for doc_type, pats in rules:
        score = 0
        for pat, weight in pats:
            if pat.search(text):
                score += weight

        if score > best_score:
            best_score = score
            best_type = doc_type

    # Keep classification conservative: require at least one strong signal.
    if best_score < 6:
        return _DOC_TYPE_UNKNOWN
    return best_type


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
# Course codes like CSE-302, PCS-601DS, BSC 101.
# Avoid matching year-like tokens (e.g., "KU 2023") that appear in headers/footers.
_COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,6}\s*[-/]?\s*(?!(?:19|20)\d{2}\b)\d{2,4}[A-Z]{0,3}\b")
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


def _canonical_tokens(text: str) -> list[str]:
    key = _canonicalize_for_repeated_match(text)
    return [t for t in key.split() if t]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = min(len(a), len(b))
    return inter / denom if denom else 0.0


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
    # However, don't let this protection block removal of obvious boilerplate headers/footers.
    if _COURSE_CODE_RE.search(upper) and not _looks_like_boilerplate(line):
        return True

    # Never touch table blocks (we add [TABLE n] markers)
    if _is_table_line(line):
        return True

    # Preserve key business headings / document labels.
    # These may repeat across pages, but they are often meaningful context.
    important_keywords = (
        "TIMETABLE",
        "TIME TABLE",
        "EXAM",
        "EXAMINATION",
        "NOTIFICATION",
        "CIRCULAR",
        "SYLLABUS",
        "REGULATION",
        "REGULATIONS",
        "OFFICE ORDER",
        "TENDER",
        "QUOTATION",
        "SCHOLARSHIP",
        "FEE",
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


def _tokens_look_like_syllabus_content(tokens: set[str]) -> bool:
    """Tokens that strongly indicate real syllabus content (do not remove)."""

    if not tokens:
        return False

    protected = {
        "unit",
        "module",
        "experiment",
        "experiments",
        "week",
        "weeks",
        "objective",
        "objectives",
        "outcome",
        "outcomes",
        "timetable",
        "exam",
        "examination",
        "notification",
        "circular",
        "syllabus",
        "laboratory",
        "lab",
    }

    return any(t in protected for t in tokens)


def _group_words_into_lines(words: list[dict], y_tolerance: float = 3.0) -> list[str]:
    """Build human-ish lines from pdfplumber words.

    Deterministic, beginner-friendly grouping:
    - Sort by (top, x0)
    - Start a new line when the 'top' coordinate jumps more than y_tolerance
    """

    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (float(w.get("top", 0.0)), float(w.get("x0", 0.0))))

    lines: list[list[dict]] = []
    current: list[dict] = []
    current_top: Optional[float] = None

    for w in sorted_words:
        top = float(w.get("top", 0.0))
        if current_top is None:
            current_top = top
            current = [w]
            continue

        if abs(top - current_top) <= y_tolerance:
            current.append(w)
        else:
            lines.append(current)
            current_top = top
            current = [w]

    if current:
        lines.append(current)

    out_lines: list[str] = []
    for line_words in lines:
        # Keep left-to-right order
        line_words_sorted = sorted(line_words, key=lambda w: float(w.get("x0", 0.0)))
        text = " ".join((w.get("text") or "").strip() for w in line_words_sorted).strip()
        text = _normalize_line(text)
        if text:
            out_lines.append(text)

    return out_lines


def _text_header_footer_window_indexes(
    page_text: str,
    header_scan_lines: int = 10,
    footer_scan_lines: int = 5,
) -> tuple[set[int], set[int]]:
    """Line indexes for header/footer windows in extracted text.

    Uses non-empty, non-table lines as the counting basis.
    """

    lines = page_text.splitlines()
    eligible: list[int] = []
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if _is_table_line(stripped):
            continue
        eligible.append(idx)

    header = set(eligible[:header_scan_lines])
    footer = set(eligible[-footer_scan_lines:]) if footer_scan_lines > 0 else set()
    return header, footer


@dataclass
class _RemovalPattern:
    zone: str  # "header" | "footer"
    label: str
    tokens: set[str]
    rep_raw: str


@dataclass
class _RegionLineCandidate:
    page_index: int  # 0-based
    zone: str  # "header" | "footer"
    raw_text: str
    tokens: frozenset[str]


@dataclass
class _TokenCluster:
    zone: str
    pages: set[int]
    core_tokens: set[str]
    union_tokens: set[str]
    raw_variants: dict[str, int]


def _is_similar_token_set(
    a: set[str],
    b: set[str],
    jaccard_threshold: float = 0.6,
    overlap_threshold: float = 0.8,
) -> bool:
    return (_jaccard(a, b) >= jaccard_threshold) or (_overlap_coeff(a, b) >= overlap_threshold)


def _pick_representative_raw(raw_variants: dict[str, int]) -> str:
    if not raw_variants:
        return ""

    max_count = max(raw_variants.values())
    min_count = max(1, int(math.ceil(max_count * 0.5)))
    pool = [raw for raw, c in raw_variants.items() if c >= min_count]
    if not pool:
        pool = list(raw_variants.keys())

    def score(raw: str) -> tuple[int, int, int, str]:
        digits = sum(ch.isdigit() for ch in raw)
        punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in raw)
        return (digits, punct, len(raw), raw)

    return min(pool, key=score)


def _extract_region_candidates_from_pdf(
    pdf,
    top_frac: float = 0.15,
    bottom_frac: float = 0.12,
) -> list[_RegionLineCandidate]:
    """Extract header/footer line candidates using word coordinates."""

    candidates: list[_RegionLineCandidate] = []

    for page_index, page in enumerate(pdf.pages):
        try:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        except Exception:
            words = []

        if not words:
            continue

        height = float(getattr(page, "height", 0.0) or 0.0)
        if height <= 0:
            continue

        top_cut = height * float(top_frac)
        bottom_cut = height * (1.0 - float(bottom_frac))

        header_words = [w for w in words if float(w.get("top", 0.0)) <= top_cut]
        footer_words = [w for w in words if float(w.get("bottom", 0.0)) >= bottom_cut]

        for zone, zone_words in (("header", header_words), ("footer", footer_words)):
            # Slightly larger tolerance helps merge multi-part banners (e.g., 'B. Tech. ... Semester')
            for raw_line in _group_words_into_lines(zone_words, y_tolerance=5.0):
                if not raw_line.strip():
                    continue
                if _is_table_line(raw_line):
                    continue
                if _should_never_remove_line(raw_line):
                    continue

                tok_list = _canonical_tokens(raw_line)
                tok_set = set(tok_list)

                # Drop very small/noisy lines (e.g., page numbers after normalization)
                if len(tok_set) < 2:
                    continue

                # Don't even consider lines that look like real syllabus structure
                if _tokens_look_like_syllabus_content(tok_set):
                    continue

                candidates.append(
                    _RegionLineCandidate(
                        page_index=page_index,
                        zone=zone,
                        raw_text=raw_line,
                        tokens=frozenset(tok_set),
                    )
                )

    return candidates


def _cluster_region_candidates(candidates: list[_RegionLineCandidate]) -> list[_TokenCluster]:
    """Greedy, deterministic clustering of similar token sets."""

    clusters: list[_TokenCluster] = []

    for cand in candidates:
        tokens = set(cand.tokens)

        best_idx: Optional[int] = None
        best_score = 0.0

        for i, cl in enumerate(clusters):
            if cl.zone != cand.zone:
                continue
            rep = cl.core_tokens if cl.core_tokens else cl.union_tokens
            jac = _jaccard(tokens, rep)
            ov = _overlap_coeff(tokens, rep)
            score = max(jac, ov)

            if score > best_score and _is_similar_token_set(tokens, rep):
                best_score = score
                best_idx = i

        if best_idx is None:
            clusters.append(
                _TokenCluster(
                    zone=cand.zone,
                    pages={cand.page_index},
                    core_tokens=set(tokens),
                    union_tokens=set(tokens),
                    raw_variants={cand.raw_text: 1},
                )
            )
            continue

        cl = clusters[best_idx]
        cl.pages.add(cand.page_index)
        cl.core_tokens &= set(tokens)
        cl.union_tokens |= set(tokens)
        cl.raw_variants[cand.raw_text] = cl.raw_variants.get(cand.raw_text, 0) + 1

    return clusters


def _clean_page_text_using_clusters(
    page_text: str,
    header_patterns: list[_RemovalPattern],
    footer_patterns: list[_RemovalPattern],
    header_scan_lines: int = 10,
    footer_scan_lines: int = 5,
) -> tuple[str, set[str]]:
    """Remove matching repeated patterns from the header/footer text windows only."""

    if not page_text.strip() or (not header_patterns and not footer_patterns):
        return page_text.strip(), set()

    header_idx, footer_idx = _text_header_footer_window_indexes(
        page_text,
        header_scan_lines=header_scan_lines,
        footer_scan_lines=footer_scan_lines,
    )

    removed_labels: set[str] = set()
    out_lines: list[str] = []

    lines = page_text.splitlines()
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()

        # Keep empty lines as-is to avoid collapsing layout too much
        if not stripped:
            out_lines.append(raw_line)
            continue

        if _is_table_line(stripped) or _should_never_remove_line(stripped):
            out_lines.append(raw_line)
            continue

        patterns: list[_RemovalPattern] = []
        if idx in header_idx:
            patterns = header_patterns
        elif idx in footer_idx:
            patterns = footer_patterns

        if not patterns:
            out_lines.append(raw_line)
            continue

        tokens = set(_canonical_tokens(stripped))
        if len(tokens) < 2:
            out_lines.append(raw_line)
            continue

        removed = False
        for pat in patterns:
            if _is_similar_token_set(tokens, pat.tokens):
                removed_labels.add(pat.label)
                removed = True
                break

        if not removed:
            out_lines.append(raw_line)

    return "\n".join(out_lines).strip(), removed_labels


def cleanup_repeated_headers_footers(
    pdf,
    extracted_pages: list[PdfPage],
    threshold: float = 0.5,
) -> tuple[list[PdfPage], list[str]]:
    """Coordinate-based header/footer cleanup.

    Uses pdfplumber word coordinates to build header/footer line candidates, clusters them
    by token-set similarity, and removes only those repeated lines from page text.

    Returns: (cleaned_pages, removed_repeated_lines)
    """

    log = logging.getLogger(__name__)

    page_count = len(getattr(pdf, "pages", []) or [])
    if page_count <= 1:
        return extracted_pages, []

    # Accept 0.5 or 50 (percent)
    if threshold > 1:
        threshold = threshold / 100.0
    threshold = max(0.0, min(1.0, threshold))
    min_pages = max(2, int(math.ceil(threshold * page_count)))

    candidates = _extract_region_candidates_from_pdf(pdf, top_frac=0.15, bottom_frac=0.12)
    clusters = _cluster_region_candidates(candidates)

    repeated_clusters = [c for c in clusters if len(c.pages) >= min_pages and len(c.core_tokens) >= 2]
    if not repeated_clusters:
        return extracted_pages, []

    # Log detected clusters
    log.info("Detected repeated headers/footers (clusters >= %d/%d pages):", min_pages, page_count)
    ordered = sorted(repeated_clusters, key=lambda c: (-len(c.pages), c.zone, " ".join(sorted(c.core_tokens))))
    for cl in ordered:
        norm = " ".join(sorted(cl.core_tokens))
        log.info("- [%s] %s", cl.zone, norm)

    summary = "; ".join(
        [f"[{cl.zone}] {' '.join(sorted(cl.core_tokens))}" for cl in ordered[:5]]
    )
    if summary:
        log.info("Detected repeated headers (summary): %s", summary)

    # Build removal patterns (cluster-level) and then remove from the extracted text.
    # Document-neutral policy: only remove patterns that were detected in header/footer
    # regions AND repeated across many pages.
    patterns: list[_RemovalPattern] = []
    for cl in repeated_clusters:
        tokens = set(cl.core_tokens if cl.core_tokens else cl.union_tokens)
        if len(tokens) < 2:
            continue

        label = " ".join(sorted(tokens))
        rep_raw = _pick_representative_raw(cl.raw_variants)
        patterns.append(_RemovalPattern(zone=cl.zone, label=label, tokens=tokens, rep_raw=rep_raw))

    header_patterns = [p for p in patterns if p.zone == "header"]
    footer_patterns = [p for p in patterns if p.zone == "footer"]

    removed_pages_by_label: dict[str, set[int]] = {p.label: set() for p in patterns}
    cleaned: list[PdfPage] = []

    for page_index, page in enumerate(extracted_pages):
        cleaned_text, removed_labels = _clean_page_text_using_clusters(
            page.content,
            header_patterns=header_patterns,
            footer_patterns=footer_patterns,
            header_scan_lines=10,
            footer_scan_lines=5,
        )
        cleaned.append(PdfPage(page_number=page.page_number, content=cleaned_text))
        for label in removed_labels:
            removed_pages_by_label.setdefault(label, set()).add(page_index)

    log.info("Removed from pages:")
    for p in sorted(patterns, key=lambda x: (-len(removed_pages_by_label.get(x.label, set())), x.zone, x.label)):
        log.info("- [%s] %s: %d pages", p.zone, p.label, len(removed_pages_by_label.get(p.label, set())))

    removed_reps: list[str] = []
    for p in sorted(patterns, key=lambda x: (-len(removed_pages_by_label.get(x.label, set())), x.zone, x.label)):
        if len(removed_pages_by_label.get(p.label, set())) <= 0:
            continue
        if p.rep_raw:
            removed_reps.append(p.rep_raw)

    # Unique, stable order
    seen: set[str] = set()
    unique_removed_reps: list[str] = []
    for r in removed_reps:
        if r not in seen:
            seen.add(r)
            unique_removed_reps.append(r)

    return cleaned, unique_removed_reps


def _to_json_dict(extracted: ExtractedPdf) -> dict:
    data = asdict(extracted)
    # Preserve existing JSON shape when cleanup is not enabled
    if data.get("cleanup_applied") is None:
        data.pop("cleanup_applied", None)
    if data.get("removed_repeated_lines") is None:
        data.pop("removed_repeated_lines", None)
    if data.get("ocr_status") is None:
        data.pop("ocr_status", None)
    return data


def extract_pdf_text(
    pdf_path: Path,
    cleanup: bool = False,
    cleanup_threshold: float = 0.5,
) -> ExtractedPdf:
    """Extract text from a PDF (no OCR).

    Notes:
    - Uses pdfplumber (pdfminer.six under the hood).
    - Scanned/image-only PDFs often produce empty text; we log a warning upstream.
    - Saves output even if some pages are empty.
    """

    extracted_at = _utc_iso_now()

    log = logging.getLogger(__name__)

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

        removed_repeated_lines: list[str] = []
        if cleanup:
            pages, removed_repeated_lines = cleanup_repeated_headers_footers(
                pdf,
                extracted_pages=pages,
                threshold=cleanup_threshold,
            )

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

        text_extracted = content_length > 0
        needs_ocr = not text_extracted
        ocr_status = "pending" if needs_ocr else None

        document_type = _classify_document_type(title=title, pages=pages, content_length=content_length)

        log.info("Detected document type: %s", document_type)
        log.info("OCR required: %s", needs_ocr)

        return ExtractedPdf(
            title=title,
            source_pdf=str(pdf_path.as_posix()),
            page_count=page_count,
            extracted_at=extracted_at,
            content_length=content_length,
            pages=pages,
            full_content=full_content,
            document_type=document_type,
            text_extracted=text_extracted,
            needs_ocr=needs_ocr,
            ocr_status=ocr_status,
            cleanup_applied=True if cleanup else None,
            removed_repeated_lines=removed_repeated_lines if cleanup else None,
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
        extracted = extract_pdf_text(
            pdf_path,
            cleanup=cleanup,
            cleanup_threshold=cleanup_threshold,
        )
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
