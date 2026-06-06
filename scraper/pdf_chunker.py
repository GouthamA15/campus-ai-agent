from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


DOC_TYPE_SYLLABUS = "syllabus"
DOC_TYPE_REGULATIONS = "academic_regulations"
DOC_TYPE_EXAM_SCHEDULE = "exam_schedule"


@dataclass
class PdfPage:
    page_number: int
    content: str


@dataclass
class Chunk:
    chunk_id: str
    chunk_type: str
    section_title: str
    page_start: int
    page_end: int
    content: str
    metadata: dict[str, Any]


@dataclass
class ChunkedDocument:
    source_pdf: str
    document_type: str
    chunk_count: int
    chunks: list[Chunk]


@dataclass(frozen=True)
class LineRecord:
    page_number: int
    text: str


@dataclass(frozen=True)
class HeadingMarker:
    index: int
    chunk_type: str
    section_title: str


@dataclass(frozen=True)
class SubjectMarker:
    index: int
    subject_name: str
    subject_slug: str
    course_code: Optional[str] = None


_SYNTHETIC_DOC_TYPES = {
    "regulation": DOC_TYPE_REGULATIONS,
    "regulations": DOC_TYPE_REGULATIONS,
    "timetable": DOC_TYPE_EXAM_SCHEDULE,
}

_GENERIC_KEYWORDS = {
    "circular",
    "notification",
    "notice",
    "scholarship",
    "fee",
    "quotation",
    "tender",
    "office order",
    "hostel",
    "placement",
    "announcement",
    "administrative",
}

_SYLLABUS_SUBJECT_ANCHOR_PATTERNS = (
    re.compile(r"\bcourse\s*code\b", re.I),
    re.compile(r"\bcourse\s*title\b", re.I),
    re.compile(r"\bscheme\s+and\s+credits\b", re.I),
    re.compile(r"\bteaching\s+scheme\b", re.I),
    re.compile(r"\bexamination\s+scheme\b", re.I),
    re.compile(r"\bmaximum\s+hours\b", re.I),
    re.compile(r"\bdetailed\s+contents\b", re.I),
    re.compile(r"\bclass\s*:\b", re.I),
)


def _configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(fmt)

    file_handler = logging.FileHandler(output_dir / "pdf_chunker.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root.handlers = []
    root.addHandler(stream)
    root.addHandler(file_handler)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "chunk"


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _is_blank(text: str) -> bool:
    return not text or not text.strip()


def _load_input_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}")
    return data


def _load_pages(data: dict[str, Any]) -> list[PdfPage]:
    pages: list[PdfPage] = []
    for index, raw_page in enumerate(data.get("pages", []) or [], start=1):
        if not isinstance(raw_page, dict):
            continue
        try:
            page_number = int(raw_page.get("page_number") or index)
        except Exception:
            page_number = index
        pages.append(PdfPage(page_number=page_number, content=str(raw_page.get("content") or "")))
    return pages


def _merged_content(pages: list[PdfPage]) -> str:
    return "\n\n".join(page.content.strip() for page in pages if page.content.strip()).strip()


def _source_pdf_name(data: dict[str, Any], fallback: str) -> str:
    raw_source = str(data.get("source_pdf") or "").strip()
    if raw_source:
        return Path(raw_source).name
    return f"{fallback}.pdf"


def _sample_text(data: dict[str, Any], pages: list[PdfPage], fallback_name: str) -> str:
    parts: list[str] = []
    title = str(data.get("title") or "").strip()
    if title:
        parts.append(title)
    parts.append(fallback_name)
    for page in pages[:2]:
        if page.content.strip():
            parts.append(page.content)
    return _normalize_whitespace(" ".join(parts)).lower()[:20000]


def _score(text: str, patterns: list[tuple[str, int]]) -> int:
    score = 0
    for pattern, weight in patterns:
        if re.search(pattern, text, re.I):
            score += weight
    return score


def _classify_document_type(data: dict[str, Any], pages: list[PdfPage], fallback_name: str) -> str:
    provided = _normalize_whitespace(str(data.get("document_type") or "")).lower().replace("-", "_")
    if provided and provided != "unknown":
        return _SYNTHETIC_DOC_TYPES.get(provided, provided)

    text = _sample_text(data, pages, fallback_name)
    if not text:
        return "generic_pdf"

    candidates: list[tuple[str, int]] = [
        (
            DOC_TYPE_SYLLABUS,
            _score(
                text,
                [
                    (r"\bsyllabus\b", 8),
                    (r"\bunit\s*[-–—]?\s*(?:[ivxlcdm]+|\d+)\b", 7),
                    (r"\btext\s*books?\b", 5),
                    (r"\breferences?\b", 5),
                    (r"\bcourse\s+outcomes?\b|\bcourse\s+objectives?\b", 4),
                ],
            ),
        ),
        (
            DOC_TYPE_REGULATIONS,
            _score(
                text,
                [
                    (r"\bregulations?\b", 8),
                    (r"\battendance\b|\bgrading\b|\bpromotion\b", 4),
                    (r"\bscheme\s+of\s+instruction\s+and\s+examination\b", 8),
                    (r"\bduration\s+and\s+program\s+of\s+study\b", 7),
                ],
            ),
        ),
        (
            DOC_TYPE_EXAM_SCHEDULE,
            _score(
                text,
                [
                    (r"\btime\s*table\b|\btimetable\b", 8),
                    (r"\bexam(schedule|ination)?\b", 6),
                    (r"\bhall\s*ticket\b", 4),
                    (r"\bbranch\b.*\bsubject\b", 3),
                ],
            ),
        ),
        ("scholarship_notice", _score(text, [(r"\bscholarship\b", 8), (r"\bfresh\b|\brenewal\b", 2)])),
        ("fee_circular", _score(text, [(r"\bfee\b", 6), (r"\blate\s+fee\b", 4), (r"\bexam\s+fee\b", 4)])),
        ("quotation", _score(text, [(r"\bcall\s+for\s+quotations?\b", 10), (r"\bquotations?\b", 6), (r"\bsealed\b", 2)])),
        ("tender", _score(text, [(r"\btender\b", 8), (r"\be-?tender\b|\bbidder\b|\bnit\b|\bemd\b", 4)])),
        ("office_order", _score(text, [(r"\boffice\s+order\b", 8), (r"\bproceedings\b|\bmemo(?:randum)?\b", 4)])),
        ("placement_notification", _score(text, [(r"\bplacement\b|\brecruitment\b|\bcampus\s+drive\b", 8)])),
        ("hostel_notice", _score(text, [(r"\bhostel\b|\bwarden\b|\bmess\b", 8)])),
        ("circular", _score(text, [(r"\bcircular\b", 8)])),
        ("notification", _score(text, [(r"\bnotification\b", 8)])),
        ("administrative_document", _score(text, [(r"\badministrative\b|\bcommittee\b|\bminutes\b", 6)])),
        ("general_announcement", _score(text, [(r"\bannouncement\b|\bpress\s+note\b", 8), (r"\bnotice\b", 3)])),
    ]

    best_type, best_score = "generic_pdf", 0
    for doc_type, score in candidates:
        if score > best_score:
            best_type = doc_type
            best_score = score

    return best_type if best_score >= 6 else "generic_pdf"


def _build_line_records(pages: list[PdfPage]) -> list[LineRecord]:
    records: list[LineRecord] = []
    for page in pages:
        for line in page.content.splitlines():
            records.append(LineRecord(page_number=page.page_number, text=line.rstrip()))
    return records


def _build_word_records(pages: list[PdfPage]) -> list[tuple[str, int]]:
    words: list[tuple[str, int]] = []
    for page in pages:
        for line in page.content.splitlines():
            for word in re.findall(r"\S+", line):
                words.append((word, page.page_number))
    return words


def _page_range_from_lines(lines: list[LineRecord]) -> tuple[int, int]:
    pages = [line.page_number for line in lines if line.text.strip()]
    if not pages:
        return 0, 0
    return min(pages), max(pages)


def _page_range_from_words(words: list[tuple[str, int]]) -> tuple[int, int]:
    pages = [page_number for _, page_number in words]
    if not pages:
        return 0, 0
    return min(pages), max(pages)


def _roman_to_int(value: str) -> Optional[int]:
    roman = value.strip().upper()
    if not roman:
        return None

    mapping = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(roman):
        current = mapping.get(char)
        if current is None:
            return None
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if total > 0 else None


def _canonical_unit_label(unit_number: str) -> str:
    if unit_number.isdigit():
        return str(int(unit_number))
    return unit_number.upper()


def _is_all_caps_heading(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized or len(normalized) > 80:
        return False
    letters = [char for char in normalized if char.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.8 and len(normalized.split()) <= 10


def _is_syllabus_subject_candidate(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized or len(normalized) > 60:
        return False

    upper = normalized.upper()
    if normalized.startswith("[TABLE"):
        return False

    excluded_substrings = (
        "UNIVERSITY",
        "COLLEGE",
        "FACULTY",
        "SEMESTER",
        "COURSE",
        "SCHEME",
        "CREDITS",
        "MARKS",
        "TEACHING",
        "EXAMINATION",
        "DETAILED CONTENTS",
        "FIRST YEAR",
        "COMMON TO ALL BRANCHES",
        "STRUCTURE OF CURRICULUM",
        "MAXIMUM HOURS",
        "CATEGORY",
        "TITLE",
        "OBJECTIVES",
        "OUTCOMES",
        "TEXT BOOKS",
        "REFERENCE BOOKS",
        "REFERENCES",
        "UNIT",
        "THEORY",
        "LAB",
        "PRACTICAL",
    )
    if any(token in upper for token in excluded_substrings):
        return False

    if _syllabus_marker(normalized):
        return False

    if not any(char.isalpha() for char in normalized):
        return False

    if not re.fullmatch(r"[A-Za-z0-9&().,'\-/\s]+", normalized):
        return False

    words = normalized.split()
    if not 1 <= len(words) <= 5:
        return False

    if normalized.endswith("."):
        return False

    return _is_all_caps_heading(normalized) or normalized == normalized.title() or bool(re.fullmatch(r"[A-Za-z]+(?:[-&][A-Za-z0-9]+)*(?:\s*[-–—]\s*[A-Za-z0-9]+)?", normalized))


def _normalize_subject_name(text: str) -> str:
    normalized = _normalize_whitespace(text)
    normalized = re.sub(r"\s*\(.*?\)\s*$", "", normalized)
    normalized = normalized.rstrip(":").strip()
    normalized = re.sub(r"\s*[-–—]\s*(?:[IVXLC]+|\d+)$", "", normalized).strip()
    normalized = re.sub(r"\s+(?:[IVXLC]+|\d+)$", "", normalized).strip()
    return normalized.title()


def _normalize_course_code(code: str) -> str:
    normalized = _normalize_whitespace(code).upper()
    normalized = normalized.lstrip("/").replace(" ", "")
    return normalized


def _extract_course_code_nearby(lines: list[LineRecord], start_index: int, end_index: int) -> Optional[str]:
    window = "\n".join(_normalize_whitespace(line.text) for line in lines[start_index:end_index] if line.text.strip())
    if not window:
        return None

    explicit = re.search(r"course\s*code\s*[:\-]?\s*([A-Z]{2,6}\s*[-/]?\s*\d{2,4}[A-Z]{0,3})", window, re.I)
    if explicit:
        return _normalize_course_code(explicit.group(1))

    fallback = re.search(r"\b([A-Z]{2,6}\s*[-/]?\s*\d{2,4}[A-Z]{0,3})\b", window)
    if fallback:
        return _normalize_course_code(fallback.group(1))

    return None


def _is_syllabus_metadata_anchor(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _SYLLABUS_SUBJECT_ANCHOR_PATTERNS)


def _find_syllabus_subject_markers(lines: list[LineRecord], unit_markers: list[HeadingMarker]) -> list[SubjectMarker]:
    if not unit_markers:
        return []

    unit_indexes = [marker.index for marker in unit_markers if marker.chunk_type == "unit"]
    if not unit_indexes:
        return []

    markers: list[SubjectMarker] = []
    last_seen_slug: Optional[str] = None
    last_seen_index = -10_000

    for anchor_index, line in enumerate(lines):
        if not _is_syllabus_metadata_anchor(line.text):
            continue

        next_unit_index = next((unit_index for unit_index in unit_indexes if unit_index > anchor_index), None)
        if next_unit_index is None:
            continue

        # Look backward from the metadata anchor for the closest course/subject heading.
        search_start = max(0, anchor_index - 8)
        candidate_index: Optional[int] = None
        candidate_text: Optional[str] = None

        for index in range(anchor_index - 1, search_start - 1, -1):
            text = _normalize_whitespace(lines[index].text)
            if not _is_syllabus_subject_candidate(text):
                continue
            candidate_index = index
            candidate_text = text
            break

        if candidate_index is None or candidate_text is None:
            continue

        # Course metadata should still be close to the first unit block.
        if next_unit_index - candidate_index > 120:
            continue

        subject_name = _normalize_subject_name(candidate_text)
        subject_slug = _slugify(subject_name)
        if last_seen_slug == subject_slug and candidate_index - last_seen_index < 80:
            continue

        course_code = _extract_course_code_nearby(lines, candidate_index, min(len(lines), anchor_index + 6))
        markers.append(
            SubjectMarker(
                index=candidate_index,
                subject_name=subject_name,
                subject_slug=subject_slug,
                course_code=course_code,
            )
        )
        last_seen_slug = subject_slug
        last_seen_index = candidate_index

    return markers


def _looks_like_heading(text: str, previous_blank: bool, next_blank: bool) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized or len(normalized) > 80:
        return False
    if normalized.startswith("[TABLE"):
        return False
    if re.fullmatch(r"\d+\s*[.)]\s*.+", normalized):
        return True
    if normalized.endswith(":") and (previous_blank or next_blank):
        return True
    if _is_all_caps_heading(normalized):
        return True
    if previous_blank and len(normalized.split()) <= 8 and normalized == normalized.title():
        return True
    return False


def _syllabus_marker(text: str) -> tuple[str, str] | None:
    normalized = _normalize_whitespace(text)
    unit_match = re.match(r"^UNIT\s*[-–—]?\s*([IVXLC]+|\d+)\b", normalized, re.I)
    if unit_match:
        return "unit", f"UNIT-{_canonical_unit_label(unit_match.group(1))}"

    section_patterns = [
        (r"^(?:SUGGESTED\s+)?TEXT\s*BOOKS?$", "textbooks"),
        (r"^(?:SUGGESTED\s+)?REFERENCE\s+BOOKS?$", "references"),
        (r"^REFERENCES?$", "references"),
        (r"^COURSE\s+OBJECTIVES?$", "course_objectives"),
        (r"^COURSE\s+OUTCOMES?$", "course_outcomes"),
        (r"^LAB\s+EXERCISES?$", "lab_exercises"),
    ]
    for pattern, chunk_type in section_patterns:
        if re.fullmatch(pattern, normalized, re.I):
            return chunk_type, normalized.upper()
    return None


def _regulation_marker(text: str) -> tuple[str, str] | None:
    normalized = _normalize_whitespace(text)
    numbered = re.match(r"^(\d+)\s*[.)]\s*(.+)$", normalized)
    if numbered:
        return "regulation_section", numbered.group(2).strip().upper()

    capsish_sections = {
        "ADMISSION",
        "DURATION AND PROGRAM OF STUDY",
        "ATTENDANCE",
        "SCHEME OF INSTRUCTION AND EXAMINATION",
        "PROMOTION",
        "GRADING",
        "AWARD OF DEGREE",
        "EVALUATION",
    }
    upper = normalized.upper()
    if upper in capsish_sections or (_is_all_caps_heading(normalized) and len(normalized.split()) <= 10):
        return "regulation_section", upper

    return None


def _generic_marker(text: str, previous_blank: bool, next_blank: bool) -> tuple[str, str] | None:
    normalized = _normalize_whitespace(text)
    if not _looks_like_heading(normalized, previous_blank, next_blank):
        return None
    title = normalized.rstrip(":").upper()
    return "section", title


def _find_markers(lines: list[LineRecord], mode: str) -> list[HeadingMarker]:
    markers: list[HeadingMarker] = []
    for index, line in enumerate(lines):
        text = _normalize_whitespace(line.text)
        if not text:
            continue
        previous_blank = index == 0 or _is_blank(lines[index - 1].text)
        next_blank = index == len(lines) - 1 or _is_blank(lines[index + 1].text)

        marker: tuple[str, str] | None = None
        if mode == "syllabus":
            marker = _syllabus_marker(text)
        elif mode == "regulations":
            marker = _regulation_marker(text)
        else:
            marker = _generic_marker(text, previous_blank, next_blank)

        if marker is None:
            continue

        chunk_type, section_title = marker
        if mode == "generic" and chunk_type == "section" and not any(keyword in section_title.lower() for keyword in _GENERIC_KEYWORDS):
            # Keep generic chunking conservative: require a heading-like pattern before
            # promoting arbitrary short prose.
            if not _looks_like_heading(text, previous_blank, next_blank):
                continue

        markers.append(HeadingMarker(index=index, chunk_type=chunk_type, section_title=section_title))

    return markers


def _section_spans(lines: list[LineRecord], markers: list[HeadingMarker]) -> list[tuple[int, int, HeadingMarker]]:
    if not markers:
        return []

    spans: list[tuple[int, int, HeadingMarker]] = []
    for idx, marker in enumerate(markers):
        end = markers[idx + 1].index if idx + 1 < len(markers) else len(lines)
        start = 0 if idx == 0 and marker.index > 0 else marker.index
        spans.append((start, end, marker))
    return spans


def _join_lines(lines: list[LineRecord]) -> str:
    return "\n".join(line.text for line in lines if line.text.strip()).strip()


def _make_chunk_id(base_name: str, suffix: str, used: set[str]) -> str:
    candidate = f"{base_name}_{_slugify(suffix)}"
    if candidate not in used:
        used.add(candidate)
        return candidate

    counter = 2
    while True:
        alternative = f"{candidate}_{counter}"
        if alternative not in used:
            used.add(alternative)
            return alternative
        counter += 1


def _make_section_chunks(
    lines: list[LineRecord],
    markers: list[HeadingMarker],
    base_name: str,
    source_pdf_name: str,
    document_type: str,
    strategy: str,
    subject_markers: Optional[list[SubjectMarker]] = None,
) -> list[Chunk]:
    spans = _section_spans(lines, markers)
    if not spans:
        return []

    used_ids: set[str] = set()
    counters: dict[tuple[str, str], int] = {}
    chunks: list[Chunk] = []
    subject_markers = sorted(subject_markers or [], key=lambda marker: marker.index)
    subject_index = 0
    active_subject: Optional[SubjectMarker] = None

    for start, end, marker in spans:
        while subject_index < len(subject_markers) and subject_markers[subject_index].index <= start:
            active_subject = subject_markers[subject_index]
            subject_index += 1

        span_lines = [line for line in lines[start:end] if not _is_blank(line.text)]
        if not span_lines:
            continue

        prefix = active_subject.subject_slug if active_subject else base_name
        counter_key = (prefix, marker.chunk_type)
        counters[counter_key] = counters.get(counter_key, 0) + 1
        index = counters[counter_key]

        metadata: dict[str, Any] = {
            "source_pdf": source_pdf_name,
            "document_type": document_type,
            "chunk_strategy": strategy,
            "word_count": _word_count(_join_lines(span_lines)),
        }
        if active_subject is not None:
            metadata["subject_name"] = active_subject.subject_name
            if active_subject.course_code:
                metadata["course_code"] = active_subject.course_code

        if marker.chunk_type == "unit":
            chunk_id = f"{prefix}_unit_{index}"
        elif marker.chunk_type == "textbooks":
            chunk_id = f"{prefix}_textbooks" if active_subject else f"{base_name}_textbooks"
        elif marker.chunk_type == "references":
            chunk_id = f"{prefix}_references" if active_subject else f"{base_name}_references"
        elif marker.chunk_type == "course_objectives":
            chunk_id = f"{prefix}_course_objectives" if active_subject else f"{base_name}_course_objectives"
        elif marker.chunk_type == "course_outcomes":
            chunk_id = f"{prefix}_course_outcomes" if active_subject else f"{base_name}_course_outcomes"
        elif marker.chunk_type == "lab_exercises":
            chunk_id = f"{prefix}_lab_exercises" if active_subject else f"{base_name}_lab_exercises"
        else:
            chunk_id = _make_chunk_id(prefix, f"section_{index}", used_ids)

        content = _join_lines(span_lines)
        page_start, page_end = _page_range_from_lines(span_lines)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chunk_type=marker.chunk_type,
                section_title=marker.section_title,
                page_start=page_start,
                page_end=page_end,
                content=content,
                metadata=metadata,
            )
        )

    return chunks


def _single_exam_schedule_chunk(pages: list[PdfPage], base_name: str, source_pdf_name: str, document_type: str) -> list[Chunk]:
    content = _merged_content(pages)
    if not content:
        return []

    page_start = min((page.page_number for page in pages), default=0)
    page_end = max((page.page_number for page in pages), default=0)
    section_title = "TIME TABLE" if re.search(r"\btime\s*table\b|\btimetable\b", content, re.I) else "EXAM SCHEDULE"

    return [
        Chunk(
            chunk_id=f"{base_name}_schedule",
            chunk_type="exam_schedule",
            section_title=section_title,
            page_start=page_start,
            page_end=page_end,
            content=content,
            metadata={
                "source_pdf": source_pdf_name,
                "document_type": document_type,
                "chunk_strategy": "single_timetable_chunk",
                "word_count": _word_count(content),
            },
        )
    ]


def _fixed_size_chunks(pages: list[PdfPage], base_name: str, source_pdf_name: str, document_type: str) -> list[Chunk]:
    words: list[tuple[str, int]] = []
    for page in pages:
        for line in page.content.splitlines():
            for word in re.findall(r"\S+", line):
                words.append((word, page.page_number))

    if not words:
        return []

    chunk_size = 800
    overlap = 100
    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    start = 0
    index = 1

    while start < len(words):
        window = words[start : start + chunk_size]
        if not window:
            break

        content = _normalize_whitespace(" ".join(word for word, _ in window))
        page_start = min(page for _, page in window)
        page_end = max(page for _, page in window)
        chunks.append(
            Chunk(
                chunk_id=f"{base_name}_chunk_{index}",
                chunk_type="chunk",
                section_title=f"Chunk {index}",
                page_start=page_start,
                page_end=page_end,
                content=content,
                metadata={
                    "source_pdf": source_pdf_name,
                    "document_type": document_type,
                    "chunk_strategy": "fixed_size",
                    "word_count": len(window),
                    "chunk_words": chunk_size,
                    "chunk_overlap": overlap,
                },
            )
        )

        if start + chunk_size >= len(words):
            break
        start += step
        index += 1

    return chunks


def _infer_chunk_mode(document_type: str) -> str:
    if document_type == DOC_TYPE_SYLLABUS:
        return DOC_TYPE_SYLLABUS
    if document_type == DOC_TYPE_REGULATIONS:
        return DOC_TYPE_REGULATIONS
    if document_type in {DOC_TYPE_EXAM_SCHEDULE, "timetable"}:
        return DOC_TYPE_EXAM_SCHEDULE
    return "generic"


def chunk_document(data: dict[str, Any], source_json: Path) -> ChunkedDocument:
    pages = _load_pages(data)
    source_pdf = _source_pdf_name(data, source_json.stem)
    base_name = Path(source_pdf).stem.lower().replace(" ", "_") or source_json.stem.lower()
    document_type = _classify_document_type(data, pages, base_name)
    source_pdf_name = Path(source_pdf).name

    mode = _infer_chunk_mode(document_type)
    if mode == DOC_TYPE_SYLLABUS:
        lines = _build_line_records(pages)
        markers = _find_markers(lines, mode="syllabus")
        subject_markers = _find_syllabus_subject_markers(lines, markers)
        chunks = _make_section_chunks(
            lines,
            markers,
            base_name,
            source_pdf_name,
            document_type,
            strategy="syllabus_sections",
            subject_markers=subject_markers,
        )
        if not chunks:
            chunks = _fixed_size_chunks(pages, base_name, source_pdf_name, document_type)
    elif mode == DOC_TYPE_REGULATIONS:
        lines = _build_line_records(pages)
        markers = _find_markers(lines, mode="regulations")
        chunks = _make_section_chunks(lines, markers, base_name, source_pdf_name, document_type, strategy="regulation_sections")
        if not chunks:
            chunks = _fixed_size_chunks(pages, base_name, source_pdf_name, document_type)
    elif mode == DOC_TYPE_EXAM_SCHEDULE:
        chunks = _single_exam_schedule_chunk(pages, base_name, source_pdf_name, document_type)
    else:
        lines = _build_line_records(pages)
        markers = _find_markers(lines, mode="generic")
        chunks = _make_section_chunks(lines, markers, base_name, source_pdf_name, document_type, strategy="heading_sections")
        if not chunks:
            chunks = _fixed_size_chunks(pages, base_name, source_pdf_name, document_type)

    content_length = len(_merged_content(pages))
    if content_length == 0:
        logging.warning("content_length == 0: %s", source_json.name)
    if pages and not chunks:
        logging.warning("document contains pages but no chunks generated: %s", source_json.name)

    return ChunkedDocument(
        source_pdf=Path(source_pdf).name,
        document_type=document_type,
        chunk_count=len(chunks),
        chunks=chunks,
    )


def _to_json_dict(chunked: ChunkedDocument) -> dict[str, Any]:
    return {
        "source_pdf": chunked.source_pdf,
        "document_type": chunked.document_type,
        "chunk_count": chunked.chunk_count,
        "chunks": [asdict(chunk) for chunk in chunked.chunks],
    }


def _save_chunked_document(chunked: ChunkedDocument, out_dir: Path, source_json: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_json.stem}_chunks.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_dict(chunked), handle, ensure_ascii=False, indent=2)
    return out_path


def _iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file() and not path.name.endswith("_chunks.json"))


def process_one(source_json: Path, out_dir: Path) -> bool:
    log = logging.getLogger(__name__)
    try:
        log.info("Processing: %s", source_json.name)
        data = _load_input_document(source_json)
        chunked = chunk_document(data, source_json)

        log.info("Document Type: %s", chunked.document_type)
        log.info("Chunks Created: %d", chunked.chunk_count)
        if chunked.chunks:
            chunk_types = _dedupe_preserve_order([chunk.chunk_type for chunk in chunked.chunks])
            log.info("Chunk Types: %s", ", ".join(chunk_types))

        if chunked.chunk_count == 0:
            log.warning("chunk_count == 0: %s", source_json.name)

        out_path = _save_chunked_document(chunked, out_dir, source_json)
        log.info("Saved JSON: %s", out_path)
        return True
    except Exception:
        log.exception("Failed to chunk JSON: %s", source_json)
        return False


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2A: Chunk parsed PDF JSON into retrieval-ready chunks.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Chunk a single JSON file")
    group.add_argument("--all", action="store_true", help="Chunk all JSON files in the input directory")

    parser.add_argument(
        "--input-dir",
        default=str(Path("data") / "pdf_text"),
        help="Input directory containing parsed PDF JSON (default: data/pdf_text)",
    )
    parser.add_argument(
        "--out",
        default=str(Path("data") / "pdf_chunks"),
        help="Output directory for chunked JSON (default: data/pdf_chunks)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    _configure_logging(out_dir, verbose=args.verbose)

    processed = 0
    successful = 0
    failed = 0

    if args.file:
        processed = 1
        ok = process_one(Path(args.file), out_dir)
        successful = 1 if ok else 0
        failed = 0 if ok else 1
    else:
        files = _iter_input_files(input_dir)
        log = logging.getLogger(__name__)
        log.info("Found %d JSON files in %s", len(files), input_dir)
        for source_json in files:
            processed += 1
            ok = process_one(source_json, out_dir)
            if ok:
                successful += 1
            else:
                failed += 1

    print(
        json.dumps(
            {
                "jsons_processed": processed,
                "jsons_successful": successful,
                "jsons_failed": failed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())