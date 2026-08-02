"""
layout_analyzer.py  –  Stage 1: Universal Layout Analyzer
===========================================================

Analyses every extracted line of a parsed PDF document and assigns it a
primary structural type together with confidence scores and raw features.

This module is ANALYSIS-ONLY.  It does NOT perform chunking, classify the
document type, detect subjects, or modify parser output.

Pipeline position
-----------------
  ... Subject Tracker
        |
  Layout Analyzer   <- NEW (Stage 1)
        |
  Chunk Builder     (unchanged, still uses its own logic)
        |
  ...

Output
------
A list of LineAnalysis objects — one per extracted line — each carrying:
  * features  — raw numeric / boolean signals
  * scores    — per-type confidence (0.0–1.0)
  * line_type — the winning structural type (string)

The Chunk Builder continues to function identically; the layout analysis is
appended to the per-file output JSON as a new top-level key "layout_analysis".

Debug mode
----------
Pass debug=True to LayoutAnalyzer to write <stem>_layout_analysis.json
alongside the output chunk file when save_debug() is called.

Standalone class — lives in scraper/layout_analyzer.py.
Stage 2 (block_detector.py) will consume this output.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ===========================================================================
# Compiled patterns
# ===========================================================================

# ---- noise / structural noise -------------------------------------------

_RE_BLANK        = re.compile(r"^\s*$")
_RE_PAGE_NUMBER  = re.compile(
    r"^\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|[-\u2013]\s*\d+\s*[-\u2013])\s*$", re.I
)
_RE_LETTERHEAD   = re.compile(
    r"^\s*(?:"
    r"OFFICE\s+OF\s+THE\b"
    r"|Accredited\s+with\b"
    r"|KAKATIYA\s+UNIVERSITY\b"
    r"|JNTU(?:H|A)?\b"
    r"|Jawaharlal\s+Nehru\s+Technological"
    r"|(?:BACHELOR|MASTER)\s+OF\s+(?:ENGINEERING|TECHNOLOGY|SCIENCE)\b"
    r"|A\s+Four\s+Year\s+Degree\s+Programme"
    r"|(?:\d{4}\s*[-\u2013]\s*\d{2,4})\s*$"
    r")",
    re.I,
)
_RE_REF_DATE     = re.compile(
    r"^\s*No\.\s*[\d/A-Za-z-]+\s+Date\s*[:\-]?\s*[\d\-/]+\s*$", re.I
)
_RE_SIGNATURE    = re.compile(
    r"^\s*(?:"
    r"(?:Addl?\.?|Additional|Joint|Deputy|Asst?\.?)\s+(?:CONTROLLER|REGISTRAR|DIRECTOR)\b"
    r"|CONTROLLER\s+OF\s+EXAMINATIONS\b"
    r"|(?:sd|Sd)/-"
    r"|Signature\s+of\b"
    r"|(?:For\s+and\s+on\s+behalf\s+of)\b"
    r"|Copy\s+to\s*[:\-]?\s*$"
    r")",
    re.I,
)
_RE_KUCT_FOOTER  = re.compile(r"KUCE\s*&\s*T\s*[-\u2013]\s*Rules\s+and\s+Regulations", re.I)
_RE_TABLE_MARKER = re.compile(r"^\[TABLE\s+\d+\]$", re.I)

# ---- table-row signals --------------------------------------------------

_RE_PIPE_SEP       = re.compile(r"\s\|\s")
_RE_PIPE_START     = re.compile(r"^\s*\|")
_RE_SINGLE_LETTER  = re.compile(r"^[A-Za-z]$")
_RE_ISOLATED_NUM   = re.compile(r"^[\d.,]+$")
_RE_ROMAN_ONLY     = re.compile(
    r"^(?:M{0,4})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$", re.I
)
_RE_GRADE_CELL     = re.compile(r"^[A-Za-z][+-]?\s*\d{0,3}$")
_RE_PAREN_SHORT    = re.compile(r"^\(\s*[A-Za-z]{1,4}\s*\)$")
_RE_NUMERIC_RANGE  = re.compile(r"^\d+\s*[-\u2013]\s*\d+$")
_RE_FROM_SEMESTER  = re.compile(
    r"^From\s+(?:I{1,3}|IV|V{1,3}|VI{1,3}|VIII|IX|X|\d+)\s+Semester", re.I
)
_RE_FORMULA        = re.compile(
    r"(?:SGPA|CGPA)\s*=\s*[\d./]+\s*=\s*[\d.]+", re.I
)
_RE_GRADE_PT_LINE  = re.compile(r"^\d{1,2}\s+[A-Za-z][+-]?\s*$")

# ---- list / distribution ------------------------------------------------

_RE_BULLET       = re.compile(r"^[\u2022\u25cf\u25e6\u25aa\u25b8\-]\s+")
_RE_NUMBERED_ITM = re.compile(
    r"^(?:"
    r"\d+[.)]\s+[a-z]"
    r"|\d+[.)]\s+(?:The|All|Copy|A\s+candidate|Candidates?)\b"
    r"|[a-z][.)]\s+"
    r"|(?:\(\s*[ivxlcIVXLC]{1,5}\s*\)|\(\s*[a-z]\s*\))\s+"
    r")",
    re.I,
)
_RE_DIST_LIST    = re.compile(
    r"^\d+\.\s+(?:The|All)\s+"
    r"(?:Dean|Head|Director|Principal|Secretary|PA\s+to|Deputy|Addl?\.?|Stack\s+File)\b",
    re.I,
)

# ---- heading signals ----------------------------------------------------

_RE_UNIT_LABEL   = re.compile(
    r"^(?:UNIT|MODULE|CHAPTER)\s*[-\u2013\u2014]?\s*(?:[IVXLC]+|\d+)\b", re.I
)
_RE_NUMBERED_SEC = re.compile(r"^(\d+(?:\.\d+)*)\s*[.)]\s*(.{3,})")
_RE_SECTION_KW   = re.compile(
    r"^(?:"
    r"admission|duration|attendance|promotion|grading|grade\b|credit\b"
    r"|fee\b|fees\b|scholarship|eligibility|rules?\b|regulation|evaluation|examination"
    r"|award\s+of|degree|internship|hostel|circular|notification|workshop"
    r"|seminar|conference|placement|facilities|library|anti.?ragging"
    r"|disciplinary|code\s+of\s+conduct|grievance|general|miscellaneous"
    r"|annexure|appendix|chapter\b|section\b|unit\b|module\b"
    r"|introduction|objectives?|outcomes?|teaching\s+scheme|examination\s+scheme"
    r"|fee\s+particulars?|instructions\s+to|note\b|payment\s+of\s+fee"
    r"|result\b|results\b|rules\s+of\s+promotion"
    r")",
    re.I,
)
_RE_COLON_HDR    = re.compile(r"^.{3,60}:\s*$")

# ---- content signals ----------------------------------------------------

_RE_CURRENCY     = re.compile(r"[\u20b9\u0024\u00a3]|\bRs\.?\b", re.I)
_RE_DATE         = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2})\b"
)
_RE_URL          = re.compile(r"https?://\S+|www\.\S+", re.I)
_RE_EMAIL        = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_RE_PHONE        = re.compile(r"\b(?:\+91[- ]?)?[6-9]\d{9}\b|\b0\d{10}\b")
_RE_COURSE_CODE  = re.compile(r"\b[A-Z]{2,6}-?\d{3,5}[A-Z]{0,5}\b(?!/\s*\d{4})")
_RE_INDENT       = re.compile(r"^(\s*)")


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class LineFeatures:
    """Raw structural features of a single extracted line."""
    text: str
    page_number: int
    line_number: int
    word_count: int
    character_count: int
    uppercase_ratio: float
    lowercase_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    starts_with_number: bool
    starts_with_bullet: bool
    starts_with_letter: bool
    ends_with_colon: bool
    ends_with_period: bool
    blank_before: bool
    blank_after: bool
    indentation_level: int
    line_length: int
    position_from_top: float        # 0.0 .. 1.0 within the page
    position_from_bottom: float     # 0.0 .. 1.0 within the page
    contains_currency: bool
    contains_date: bool
    contains_url: bool
    contains_email: bool
    contains_phone: bool
    contains_course_code: bool
    contains_unit_label: bool
    contains_table_separator: bool
    repeated_on_multiple_pages: bool


@dataclass
class LineScores:
    """Per-type confidence scores, each in [0.0, 1.0]."""
    heading:   float = 0.0
    paragraph: float = 0.0
    table:     float = 0.0
    list:      float = 0.0
    header:    float = 0.0
    footer:    float = 0.0
    reference: float = 0.0
    signature: float = 0.0
    noise:     float = 0.0
    caption:   float = 0.0
    address:   float = 0.0
    unknown:   float = 0.0


@dataclass
class LineAnalysis:
    """Complete analysis record for a single extracted line."""
    page: int
    line_number: int
    text: str
    line_type: str           # winning type name (string key)
    scores: LineScores
    features: LineFeatures


@dataclass
class DocumentLayout:
    """Aggregated layout analysis for one entire document."""
    source_name: str
    total_lines: int
    total_pages: int
    lines: list[LineAnalysis] = field(default_factory=list)

    # Internal fast-lookup index
    _index: dict[int, LineAnalysis] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {la.line_number: la for la in self.lines}

    def get(self, line_number: int) -> LineAnalysis | None:
        """Return the analysis for a given global line number."""
        return self._index.get(line_number)

    def heading_lines(self) -> list[LineAnalysis]:
        return [la for la in self.lines if la.line_type == "heading"]

    def table_lines(self) -> list[LineAnalysis]:
        return [la for la in self.lines if la.line_type in ("table_row", "table_header")]

    def noise_lines(self) -> list[LineAnalysis]:
        return [la for la in self.lines
                if la.line_type in ("noise", "header", "footer", "page_number")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "total_lines": self.total_lines,
            "total_pages": self.total_pages,
            "lines": [
                {
                    "page": la.page,
                    "line_number": la.line_number,
                    "text": la.text,
                    "line_type": la.line_type,
                    "scores": asdict(la.scores),
                    "features": asdict(la.features),
                }
                for la in self.lines
            ],
        }


# ===========================================================================
# Internal helpers
# ===========================================================================

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0


def _lower_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.islower()) / len(letters) if letters else 0.0


def _digit_ratio(text: str) -> float:
    return sum(1 for c in text if c.isdigit()) / len(text) if text else 0.0


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    puncts = set('.,;:!?()[]{}"\'-/&@#$%')
    return sum(1 for c in text if c in puncts) / len(text)


def _indent_level(text: str) -> int:
    m = _RE_INDENT.match(text or "")
    if not m:
        return 0
    sp = m.group(1)
    return sp.count("\t") * 4 + sp.count(" ")


def _is_noise_token(n: str) -> bool:
    """True for single-char noise, grade labels, isolated numbers, etc."""
    if not n:
        return True
    if _RE_SINGLE_LETTER.fullmatch(n):
        return True
    if _RE_ISOLATED_NUM.fullmatch(n):
        return True
    if _RE_ROMAN_ONLY.fullmatch(n) and len(n) <= 6:
        return True
    if _RE_GRADE_CELL.fullmatch(n):
        return True
    if _RE_PAREN_SHORT.fullmatch(n):
        return True
    if _RE_NUMERIC_RANGE.fullmatch(n):
        return True
    if len(n) <= 2 and not any(c.isalpha() for c in n):
        return True
    return False


def _softmax(raw: dict[str, float]) -> dict[str, float]:
    """Normalise raw score dict to sum-to-1 probabilities via softmax."""
    if not raw:
        return raw
    vals = list(raw.values())
    max_v = max(vals)
    exp_vals = [math.exp(v - max_v) for v in vals]
    total = sum(exp_vals)
    return {k: exp_vals[i] / total for i, k in enumerate(raw)}


# ===========================================================================
# Feature extractor
# ===========================================================================

def _extract_features(
    n: str,
    raw_text: str,
    page_number: int,
    line_number: int,
    blank_before: bool,
    blank_after: bool,
    lines_in_page: int,
    position_in_page: int,
    repeated_pages: set[int],
    all_pages: set[int],
) -> LineFeatures:
    words = n.split()
    wc    = len(words)
    cc    = len(n)
    ur    = _upper_ratio(n)
    lr    = _lower_ratio(n)
    dr    = _digit_ratio(n)
    pr    = _punct_ratio(n)
    ind   = _indent_level(raw_text)

    pos_top    = position_in_page / max(1, lines_in_page - 1)
    pos_bottom = 1.0 - pos_top

    is_repeated = (
        len(repeated_pages) >= 2
        and len(repeated_pages) / max(1, len(all_pages)) >= 0.4
    )

    return LineFeatures(
        text                     = n,
        page_number              = page_number,
        line_number              = line_number,
        word_count               = wc,
        character_count          = cc,
        uppercase_ratio          = round(ur, 3),
        lowercase_ratio          = round(lr, 3),
        digit_ratio              = round(dr, 3),
        punctuation_ratio        = round(pr, 3),
        starts_with_number       = bool(re.match(r"^\d", n)),
        starts_with_bullet       = bool(_RE_BULLET.match(n)),
        starts_with_letter       = bool(re.match(r"^[A-Za-z]", n)),
        ends_with_colon          = n.endswith(":"),
        ends_with_period         = n.endswith("."),
        blank_before             = blank_before,
        blank_after              = blank_after,
        indentation_level        = ind,
        line_length              = cc,
        position_from_top        = round(pos_top, 3),
        position_from_bottom     = round(pos_bottom, 3),
        contains_currency        = bool(_RE_CURRENCY.search(n)),
        contains_date            = bool(_RE_DATE.search(n)),
        contains_url             = bool(_RE_URL.search(n)),
        contains_email           = bool(_RE_EMAIL.search(n)),
        contains_phone           = bool(_RE_PHONE.search(n)),
        contains_course_code     = bool(_RE_COURSE_CODE.search(n)),
        contains_unit_label      = bool(_RE_UNIT_LABEL.search(n)),
        contains_table_separator = bool(_RE_PIPE_SEP.search(n) or _RE_PIPE_START.match(n)),
        repeated_on_multiple_pages = is_repeated,
    )


# ===========================================================================
# Score calculator
# ===========================================================================

def _score_line(feat: LineFeatures, n: str) -> LineScores:
    """
    Calculate per-type raw scores then normalise via softmax.
    All logic is deterministic and rule-based — no ML.
    """
    raw: dict[str, float] = {
        "heading":   0.0,
        "paragraph": 0.0,
        "table":     0.0,
        "list":      0.0,
        "header":    0.0,
        "footer":    0.0,
        "reference": 0.0,
        "signature": 0.0,
        "noise":     0.0,
        "caption":   0.0,
        "address":   0.0,
        "unknown":   0.05,
    }

    wc = feat.word_count
    ur = feat.uppercase_ratio
    dr = feat.digit_ratio
    cc = feat.character_count

    words = n.split()

    # ---- NOISE ---------------------------------------------------------------
    if not n:
        raw["noise"] = 10.0
    if _RE_TABLE_MARKER.fullmatch(n):
        raw["noise"] += 8.0
    if _RE_PAGE_NUMBER.match(n):
        raw["noise"] += 8.0
        raw["footer"] += 2.0
    if _is_noise_token(n):
        raw["noise"] += 7.0
        raw["table"]  += 3.0
    if _RE_FORMULA.search(n):
        raw["noise"]  += 2.0
        raw["table"]  += 4.0
        raw["paragraph"] += 2.0

    # ---- HEADER / FOOTER / LETTERHEAD ----------------------------------------
    if _RE_LETTERHEAD.match(n):
        raw["header"] += 6.0
        raw["noise"]  += 1.0
    if _RE_KUCT_FOOTER.search(n):
        raw["footer"] += 6.0
        raw["noise"]  += 1.0
    if feat.repeated_on_multiple_pages:
        raw["header"] += 3.0
        raw["footer"] += 2.0
        raw["noise"]  += 1.0
    if feat.position_from_top <= 0.08 and feat.page_number > 1:
        raw["header"] += 1.0
    if feat.position_from_bottom <= 0.08 and feat.page_number > 1:
        raw["footer"] += 1.0

    # ---- SIGNATURE -----------------------------------------------------------
    if _RE_SIGNATURE.match(n):
        raw["signature"] += 6.0
        raw["noise"]     += 2.0
    if _RE_REF_DATE.match(n):
        raw["header"] += 4.0
        raw["noise"]  += 2.0

    # ---- TABLE ---------------------------------------------------------------
    if feat.contains_table_separator:
        raw["table"] += 5.0
    if _RE_FROM_SEMESTER.match(n):
        raw["table"] += 4.0
    if _RE_GRADE_PT_LINE.match(n):
        raw["table"] += 4.0
    if wc <= 4 and dr >= 0.3:
        raw["table"] += 2.0
    if 1 <= wc <= 3 and all(_is_noise_token(w) for w in words):
        raw["table"] += 3.0

    # ---- LIST ----------------------------------------------------------------
    if feat.starts_with_bullet:
        raw["list"] += 6.0
    if _RE_NUMBERED_ITM.match(n):
        raw["list"] += 4.0
    if _RE_DIST_LIST.match(n):
        raw["list"]  += 5.0
        raw["noise"] += 1.0

    # ---- HEADING -------------------------------------------------------------
    # Unit labels (UNIT-I, MODULE-2, CHAPTER III) are very strong headings
    if feat.contains_unit_label and not _RE_NUMBERED_ITM.match(n):
        raw["heading"] += 6.0

    # Section keyword at line start
    if _RE_SECTION_KW.match(n):
        raw["heading"] += 4.0

    # Numbered section with keyword or ALL-CAPS body: "3. ATTENDANCE"
    m = _RE_NUMBERED_SEC.match(n)
    if m:
        body = m.group(2).strip()
        if _RE_SECTION_KW.match(body) or _upper_ratio(body) >= 0.7:
            raw["heading"] += 5.0
        elif body and body[0].islower():
            raw["list"] += 3.0  # numbered list item

    # ALL-CAPS short heading
    if ur >= 0.75 and 1 <= wc <= 10 and cc <= 80:
        raw["heading"] += 3.0

    # Colon heading surrounded by whitespace
    if feat.ends_with_colon and (feat.blank_before or feat.blank_after) and 3 <= cc <= 65:
        raw["heading"] += 2.0

    # Title-case, short, isolated (blank on both sides, not digit-led)
    if (feat.blank_before and feat.blank_after
            and 1 <= wc <= 8
            and n == n.title()
            and not feat.starts_with_number):
        raw["heading"] += 1.0

    # Penalties for headings
    if _RE_FORMULA.search(n):
        raw["heading"] -= 6.0
    if dr >= 0.5 and wc <= 4:
        raw["heading"] -= 3.0

    # ---- REFERENCE / ADDRESS ------------------------------------------------
    if feat.contains_url or feat.contains_email:
        raw["reference"] += 3.0
        raw["paragraph"] += 1.0
    if feat.contains_phone:
        raw["address"]   += 3.0
        raw["reference"] += 1.0
    if feat.contains_course_code and wc <= 6:
        raw["reference"] += 2.0

    # ---- PARAGRAPH -----------------------------------------------------------
    if wc >= 8 and feat.lowercase_ratio >= 0.6:
        raw["paragraph"] += 4.0
    if feat.ends_with_period and wc >= 5:
        raw["paragraph"] += 2.0
    if feat.contains_currency:
        raw["paragraph"] += 1.0
    if feat.contains_date and wc >= 5:
        raw["paragraph"] += 1.0

    # ---- CAPTION -------------------------------------------------------------
    if n.lower().startswith(("fig", "figure", "table", "chart")) and wc <= 12:
        raw["caption"] += 5.0

    # Clamp negatives
    raw = {k: max(0.0, v) for k, v in raw.items()}

    # Softmax normalise
    norm = _softmax(raw)

    return LineScores(
        heading   = round(norm.get("heading",   0.0), 4),
        paragraph = round(norm.get("paragraph", 0.0), 4),
        table     = round(norm.get("table",     0.0), 4),
        list      = round(norm.get("list",      0.0), 4),
        header    = round(norm.get("header",    0.0), 4),
        footer    = round(norm.get("footer",    0.0), 4),
        reference = round(norm.get("reference", 0.0), 4),
        signature = round(norm.get("signature", 0.0), 4),
        noise     = round(norm.get("noise",     0.0), 4),
        caption   = round(norm.get("caption",   0.0), 4),
        address   = round(norm.get("address",   0.0), 4),
        unknown   = round(norm.get("unknown",   0.0), 4),
    )


def _winning_type(scores: LineScores, feat: LineFeatures) -> str:
    """Return the winning type name, with sanity overrides."""
    d = asdict(scores)
    winner = max(d, key=lambda k: d[k])
    # Heading too long → paragraph
    if winner == "heading" and feat.word_count > 12:
        winner = "paragraph"
    # Noise winner but line has a course code → protect as reference
    if winner in ("noise", "header", "footer") and feat.contains_course_code:
        winner = "reference"
    return winner


# ===========================================================================
# Layout Analyzer  (main class)
# ===========================================================================

class LayoutAnalyzer:
    """
    Universal Layout Analyzer — Stage 1.

    Assigns every extracted line a structural type and confidence scores.
    Does NOT modify or influence the Chunk Builder's output.

    Usage::

        from pipeline.pdf.layout_analyzer import LayoutAnalyzer

        analyzer = LayoutAnalyzer(debug=True)
        layout   = analyzer.analyze(source_name="doc.json", pages=pages)

        # Inspect
        for la in layout.heading_lines():
            print(la.page, la.text)

        # Save debug artifact
        analyzer.save_debug(layout, output_dir="data/pdf_chunks")

    Parameters
    ----------
    debug : bool
        Enable debug artifact writing.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def analyze(
        self,
        source_name: str,
        pages: list[dict[str, Any]],
    ) -> DocumentLayout:
        """
        Analyse all pages and return a DocumentLayout.

        Parameters
        ----------
        source_name :
            Identifier string (the source JSON filename, e.g. "doc.json").
        pages :
            List of page dicts, each containing:
              - "page_number" (int)
              - "content"     (str)
        """
        all_page_nums: set[int] = {
            (pg.page_number if hasattr(pg, "page_number") else pg.get("page_number", i + 1))
            for i, pg in enumerate(pages)
        }

        # ── Phase A: flatten all lines across pages ─────────────────────────
        # Each entry: (page_num, local_position, raw_text)
        raw_lines: list[tuple[int, int, str]] = []
        page_line_counts: dict[int, int] = {}
        for pg in pages:
            if hasattr(pg, "page_number"):
                pnum = pg.page_number
                content = pg.content
            else:
                pnum    = pg.get("page_number", 1)
                content = pg.get("content", "")
            pg_lines = content.splitlines()
            page_line_counts[pnum] = max(1, len(pg_lines))
            for pos, raw in enumerate(pg_lines):
                raw_lines.append((pnum, pos, raw))

        # ── Phase B: detect repeated lines (headers/footers) ────────────────
        text_to_pages: dict[str, set[int]] = {}
        for pnum, _pos, raw in raw_lines:
            n = _norm(raw)
            if n:
                text_to_pages.setdefault(n, set()).add(pnum)

        # ── Phase C: analyse each line ───────────────────────────────────────
        analyses: list[LineAnalysis] = []
        total = len(raw_lines)

        for global_idx, (pnum, local_pos, raw) in enumerate(raw_lines):
            n = _norm(raw)

            # Context: blank before / after
            prev_raw = raw_lines[global_idx - 1][2] if global_idx > 0 else ""
            next_raw = raw_lines[global_idx + 1][2] if global_idx + 1 < total else ""
            blank_before = not _norm(prev_raw)
            blank_after  = not _norm(next_raw)

            lines_in_page   = page_line_counts.get(pnum, 1)
            repeated_pages  = text_to_pages.get(n, set())

            feat = _extract_features(
                n               = n,
                raw_text        = raw,
                page_number     = pnum,
                line_number     = global_idx,
                blank_before    = blank_before,
                blank_after     = blank_after,
                lines_in_page   = lines_in_page,
                position_in_page= local_pos,
                repeated_pages  = repeated_pages,
                all_pages       = all_page_nums,
            )

            if not n:
                scores    = LineScores(noise=1.0)
                line_type = "noise"
            else:
                scores    = _score_line(feat, n)
                line_type = _winning_type(scores, feat)

            analyses.append(LineAnalysis(
                page        = pnum,
                line_number = global_idx,
                text        = n,
                line_type   = line_type,
                scores      = scores,
                features    = feat,
            ))

        return DocumentLayout(
            source_name = source_name,
            total_lines = len(analyses),
            total_pages = len(pages),
            lines       = analyses,
        )

    def save_debug(
        self,
        layout: DocumentLayout,
        output_dir: str | Path,
        *,
        basename: str | None = None,
    ) -> Path:
        """
        Write <basename>_layout_analysis.json to output_dir.
        Called only when self.debug is True.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem     = basename or Path(layout.source_name).stem
        out_path = output_dir / f"{stem}_layout_analysis.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(layout.to_dict(), fh, ensure_ascii=False, indent=2)
        return out_path


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def analyze_layout(
    source_name: str,
    pages: list[dict[str, Any]],
    debug: bool = False,
    output_dir: str | Path | None = None,
) -> DocumentLayout:
    """
    Convenience wrapper around LayoutAnalyzer.analyze().

    Parameters
    ----------
    source_name : str
        Name of the source document.
    pages : list[dict]
        Pages list as returned by the PDF extractor.
    debug : bool
        If True and output_dir is provided, write the debug JSON.
    output_dir : Path or str, optional
        Directory for debug output.
    """
    analyzer = LayoutAnalyzer(debug=debug)
    layout   = analyzer.analyze(source_name=source_name, pages=pages)
    if debug and output_dir:
        analyzer.save_debug(layout, output_dir=output_dir)
    return layout
