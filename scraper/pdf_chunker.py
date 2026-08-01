"""
pdf_chunker.py — Production-Ready Modular 6-Phase Document Classification & Chunking Engine
=============================================================================================

Pipeline
--------
  PDF (JSON)
    → Phase 1: FeatureExtractor      (extract rich evidence, never classify)
    → Phase 2: CandidateScorer       (per-type scorer profiles via Registry)
    → Phase 3: ConfidenceAnalyzer    (evidence-aware: strong/weak/contradictions)
    → Phase 4: DocumentValidator     (full structural validation for every type)
    → Phase 5: FinalClassification   (single authoritative result — used everywhere downstream)
    → Phase 6: ChunkerStrategy       (dispatch via Registry using FinalClassification, never classifies)
    → Phase 7: Enriched Metadata     (confidence, score, validator, feature summary)
    → Phase 8: Classification Report (classification_report.json)
    → Phase 9: DocumentTypeRegistry  (single registration point per doc type)

Design rules
------------
- Deterministic, explainable, rule-based, modular, extensible.
- No ML / AI / LLM / OCR / fuzzy libraries.
- Feature Extractor NEVER classifies.
- Chunker NEVER classifies.
- After Phase 4, FinalClassification is the ONE source of truth.
  No earlier document_type string (including data["document_type"]) is ever reused.
- Adding a new type = registering one DocumentTypeProfile.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ===========================================================================
# DOCUMENT TYPE CONSTANTS
# ===========================================================================

# Academic
DOC_TYPE_SYLLABUS           = "syllabus"
DOC_TYPE_LAB_MANUAL         = "lab_manual"
DOC_TYPE_ACADEMIC_CAL       = "academic_calendar"
DOC_TYPE_ALMANAC            = "almanac"
DOC_TYPE_EXAM_SCHEDULE      = "examination_schedule"
DOC_TYPE_CLASS_TIMETABLE    = "class_timetable"
DOC_TYPE_REGULATIONS        = "regulations"

# Administration
DOC_TYPE_NOTIFICATION       = "notification"
DOC_TYPE_CIRCULAR           = "circular"
DOC_TYPE_OFFICE_ORDER       = "office_order"
DOC_TYPE_PROCEEDINGS        = "proceedings"
DOC_TYPE_MINUTES            = "minutes"
DOC_TYPE_MEMORANDUM         = "memorandum"
DOC_TYPE_REPORT             = "report"

# Finance
DOC_TYPE_FEE_NOTICE         = "fee_notice"
DOC_TYPE_SCHOLARSHIP        = "scholarship"
DOC_TYPE_REIMBURSEMENT      = "reimbursement"
DOC_TYPE_TENDER             = "tender"
DOC_TYPE_QUOTATION          = "quotation"

# Events
DOC_TYPE_WORKSHOP           = "workshop"
DOC_TYPE_SEMINAR            = "seminar"
DOC_TYPE_CONFERENCE         = "conference"
DOC_TYPE_FDP                = "faculty_development_program"
DOC_TYPE_PLACEMENT          = "placement_drive"

# Student
DOC_TYPE_ADMISSIONS         = "admissions"
DOC_TYPE_HOSTEL             = "hostel_notice"
DOC_TYPE_EXAM_RESULTS       = "examination_results"
DOC_TYPE_INTERNSHIP         = "internships"

# Fallback
DOC_TYPE_GENERIC            = "generic_document"
DOC_TYPE_UNKNOWN            = "unknown"

# Category → subtype groupings for document_subtype field
_DOC_CATEGORY: dict[str, str] = {
    DOC_TYPE_SYLLABUS:        "academic",
    DOC_TYPE_LAB_MANUAL:      "academic",
    DOC_TYPE_ACADEMIC_CAL:    "academic",
    DOC_TYPE_ALMANAC:         "academic",
    DOC_TYPE_EXAM_SCHEDULE:   "academic",
    DOC_TYPE_CLASS_TIMETABLE: "academic",
    DOC_TYPE_REGULATIONS:     "academic",
    DOC_TYPE_NOTIFICATION:    "administration",
    DOC_TYPE_CIRCULAR:        "administration",
    DOC_TYPE_OFFICE_ORDER:    "administration",
    DOC_TYPE_PROCEEDINGS:     "administration",
    DOC_TYPE_MINUTES:         "administration",
    DOC_TYPE_MEMORANDUM:      "administration",
    DOC_TYPE_REPORT:          "administration",
    DOC_TYPE_FEE_NOTICE:      "finance",
    DOC_TYPE_SCHOLARSHIP:     "finance",
    DOC_TYPE_REIMBURSEMENT:   "finance",
    DOC_TYPE_TENDER:          "finance",
    DOC_TYPE_QUOTATION:       "finance",
    DOC_TYPE_WORKSHOP:        "event",
    DOC_TYPE_SEMINAR:         "event",
    DOC_TYPE_CONFERENCE:      "event",
    DOC_TYPE_FDP:             "event",
    DOC_TYPE_PLACEMENT:       "event",
    DOC_TYPE_ADMISSIONS:      "student",
    DOC_TYPE_HOSTEL:          "student",
    DOC_TYPE_EXAM_RESULTS:    "student",
    DOC_TYPE_INTERNSHIP:      "student",
}

_SYNTHETIC_DOC_TYPES: dict[str, str] = {
    "regulation":        DOC_TYPE_REGULATIONS,
    "regulations":       DOC_TYPE_REGULATIONS,
    "timetable":         DOC_TYPE_CLASS_TIMETABLE,
    "exam_schedule":     DOC_TYPE_EXAM_SCHEDULE,
    "academic_calendar": DOC_TYPE_ACADEMIC_CAL,
    "lab manual":        DOC_TYPE_LAB_MANUAL,
    "minutes":           DOC_TYPE_MINUTES,
    "memo":              DOC_TYPE_MEMORANDUM,
}


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

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
    document_subtype: str
    chunk_count: int
    chunks: list[Chunk]


@dataclass
class FinalClassification:
    """
    Phase 5 output — the single authoritative classification result.

    Every downstream component (chunker, metadata, output JSON, report, logging)
    must receive this object and read document_type exclusively from it.
    No earlier document_type string or data["document_type"] value may be
    consulted after this point.
    """
    document_type: str
    document_subtype: str
    confidence: str           # "HIGH" | "MEDIUM" | "LOW"
    score: int
    runner_up: str
    runner_up_score: int
    gap: int
    strong_evidence_count: int
    validator_result: str     # "passed" | "failed"
    validator_reason: str


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


@dataclass
class FeatureMap:
    """
    Phase 1 output — structured evidence only, NO classification labels.
    Uses counts (int / list) wherever possible instead of bare booleans.
    """
    # General
    page_count: int = 0
    word_count: int = 0
    line_count: int = 0
    table_count: int = 0
    image_count: int = 0

    # Academic Structure
    course_code_count: int = 0
    course_codes: list[str] = field(default_factory=list)
    unit_count: int = 0
    unit_numbers: list[str] = field(default_factory=list)
    module_count: int = 0
    chapter_count: int = 0
    chapter_numbers: list[str] = field(default_factory=list)
    subject_count: int = 0
    subject_titles: list[str] = field(default_factory=list)
    lab_count: int = 0
    course_objectives_present: bool = False
    course_outcomes_present: bool = False
    textbook_section_count: int = 0
    reference_section_count: int = 0
    bibliography_present: bool = False
    teaching_scheme_present: bool = False
    examination_scheme_present: bool = False
    credits_present: bool = False
    ltpc_table_present: bool = False
    rule_section_count: int = 0
    clause_count: int = 0
    attendance_rules: bool = False
    grading_rules: bool = False

    # Schedule / Timetable Features
    date_count: int = 0
    date_values: list[str] = field(default_factory=list)
    time_count: int = 0
    time_values: list[str] = field(default_factory=list)
    am_pm_count: int = 0
    day_names: list[str] = field(default_factory=list)
    month_names: list[str] = field(default_factory=list)
    session_labels: list[str] = field(default_factory=list)
    room_numbers: list[str] = field(default_factory=list)
    exam_time_patterns: int = 0
    holiday_patterns: int = 0
    holiday_entries: list[str] = field(default_factory=list)
    period_labels: list[str] = field(default_factory=list)
    instruction_days_present: bool = False
    odd_even_semester_present: bool = False

    # Administrative Features
    quotation_present: bool = False
    tender_present: bool = False
    terms_conditions_present: bool = False
    notification_present: bool = False
    notification_number: str = ""
    office_order_present: bool = False
    proceedings_present: bool = False
    minutes_present: bool = False
    circular_present: bool = False
    circular_number: str = ""
    memo_present: bool = False
    signatures_present: bool = False
    official_seal_present: bool = False
    authorities: list[str] = field(default_factory=list)
    annexure_count: int = 0
    appendix_count: int = 0
    item_table_count: int = 0

    # Financial Features
    fee_present: bool = False
    scholarship_present: bool = False
    reimbursement_present: bool = False
    bank_details_present: bool = False
    payment_deadline_present: bool = False
    amount_count: int = 0
    income_criterion_present: bool = False
    eligibility_present: bool = False
    documents_required_present: bool = False

    # Event Features
    workshop_present: bool = False
    seminar_present: bool = False
    conference_present: bool = False
    fdp_present: bool = False
    placement_present: bool = False
    resource_person_present: bool = False
    coordinator_present: bool = False
    registration_deadline: str = ""
    venue: str = ""
    event_name: str = ""
    organizer: str = ""

    # Student Features
    admissions_present: bool = False
    hostel_present: bool = False
    results_present: bool = False
    internship_present: bool = False

    # Report Features
    report_sections: list[str] = field(default_factory=list)

    # Contact Features
    contact_numbers: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    table_headers: list[str] = field(default_factory=list)

    # Metadata
    university_name: str = ""
    department_names: list[str] = field(default_factory=list)
    program_name: str = ""
    semester: str = ""
    academic_year: str = ""


@dataclass
class CandidateScore:
    doc_type: str
    score: int
    strong_evidence_count: int = 0
    weak_evidence_count: int = 0
    evidence: list[str] = field(default_factory=list)


@dataclass
class ConfidenceResult:
    winner: str
    winner_score: int
    runner_up: str
    runner_up_score: int
    gap: int
    level: str                          # "HIGH" | "MEDIUM" | "LOW"
    strong_evidence_count: int = 0
    weak_evidence_count: int = 0
    contradictions: list[str] = field(default_factory=list)
    structural_completeness: float = 0.0   # 0.0–1.0
    ranked: list[CandidateScore] = field(default_factory=list)


@dataclass
class ValidationResult:
    passed: bool
    final_type: str
    reason: str = ""


# ===========================================================================
# UTILITY HELPERS
# ===========================================================================

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


def _merged_content(pages: list[PdfPage]) -> str:
    return "\n\n".join(page.content.strip() for page in pages if page.content.strip()).strip()


def _canonical_unit_label(unit_number: str) -> str:
    return str(int(unit_number)) if unit_number.isdigit() else unit_number.upper()


def _is_all_caps_heading(text: str) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized or len(normalized) > 80:
        return False
    letters = [c for c in normalized if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.8 and len(normalized.split()) <= 10


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


# ===========================================================================
# I/O HELPERS
# ===========================================================================

def _configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "pdf_chunker.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.handlers = []
    root.addHandler(sh)
    root.addHandler(fh)


def _load_input_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}")
    return data


def _load_pages(data: dict[str, Any]) -> list[PdfPage]:
    pages: list[PdfPage] = []
    for index, raw in enumerate(data.get("pages", []) or [], start=1):
        if not isinstance(raw, dict):
            continue
        try:
            page_number = int(raw.get("page_number") or index)
        except Exception:
            page_number = index
        pages.append(PdfPage(page_number=page_number, content=str(raw.get("content") or "")))
    return pages


def _source_pdf_name(data: dict[str, Any], fallback: str) -> str:
    raw = str(data.get("source_pdf") or "").strip()
    return Path(raw).name if raw else f"{fallback}.pdf"


def _iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.glob("*.json")
        if p.is_file() and not p.name.endswith("_chunks.json")
    )


def _save_chunked_document(chunked: ChunkedDocument, out_dir: Path, source_json: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_json.stem}_chunks.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source_pdf":       chunked.source_pdf,
                "document_type":    chunked.document_type,
                "document_subtype": chunked.document_subtype,
                "chunk_count":      chunked.chunk_count,
                "chunks":           [asdict(c) for c in chunked.chunks],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


# ===========================================================================
# PHASE 1 – FEATURE EXTRACTOR
# ===========================================================================

class FeatureExtractor:
    """
    Extracts rich structured evidence from parsed PDF pages.

    CONTRACT:
      - Performs NO classification.
      - Returns a populated FeatureMap.
      - Uses counts (lists/ints) wherever possible, not bare booleans.
    """

    # --- Academic ---
    _COURSE_CODE    = re.compile(r"\b([A-Z]{2,6}\s*[-/]?\s*\d{3,5}[A-Z]{0,5})\b", re.I)
    _UNIT           = re.compile(r"\b(?:UNIT|MODULE|CHAPTER)\s*[-–—]?\s*([IVXLC]+|\d+)\b", re.I)
    _CHAPTER_NUM    = re.compile(r"\bCHAPTER\s*[-–—]?\s*(\d+)\b", re.I)
    _LAB            = re.compile(r"\b(?:lab|laboratory|practical)\b", re.I)
    _COURSE_OBJ     = re.compile(r"\bcourse\s+objectives?\b", re.I)
    _COURSE_OUT     = re.compile(r"\bcourse\s+outcomes?\b", re.I)
    _TEXTBOOKS      = re.compile(r"\btext\s*books?\b|\bprescribed\s+books?\b|\brecommended\s+books?\b|\blearning\s+resources?\b", re.I)
    _REFERENCES     = re.compile(r"\breference\s*books?\b|\breferences?\b|\bbibliography\b", re.I)
    _TEACHING_SCH   = re.compile(r"\bteaching\s+scheme\b", re.I)
    _EXAM_SCH       = re.compile(r"\bexamination\s+scheme\b", re.I)
    _CREDITS        = re.compile(r"\bcredits?\b", re.I)
    _LTPC           = re.compile(r"\bL\s*[-–]\s*T\s*[-–]\s*P\s*[-–]\s*C\b|\bL\s+T\s+P\s+C\b", re.I)
    _TABLE_TAG      = re.compile(r"^\[TABLE\s+\d+\]$", re.I | re.M)
    _RULE_SECTION   = re.compile(r"\b(?:rule|clause|regulation|article|section)\s+\d+\b", re.I)
    _CLAUSE         = re.compile(r"\bclause\s+\d+\b", re.I)
    _ATTENDANCE     = re.compile(r"\battendance\s+(?:rule|regulation|requirement|percentage)\b", re.I)
    _GRADING        = re.compile(r"\bgrading\s+(?:rule|system|policy|scheme)\b|\bgrade\s+point\b", re.I)

    # --- Schedule ---
    _DATE           = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b",
        re.I,
    )
    _TIME           = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|hrs?)?\b", re.I)
    _AM_PM          = re.compile(r"\b(?:AM|PM)\b", re.I)
    _DAY_NAMES      = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.I)
    _MONTH_NAMES    = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
    _SESSION        = re.compile(r"\b(FN|AN|Morning\s+Session|Afternoon\s+Session|Session\s+[IVXI]+)\b", re.I)
    _ROOM           = re.compile(r"\bRoom\s*No\.?\s*\w+\b|\bHall\s*No?\.?\s*\w*\b|\bBlock\s+[A-Z0-9]+\b", re.I)
    _EXAM_TIME_SLOT = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\s*[-–]\s*\d{1,2}:\d{2}\s*(?:AM|PM)\b", re.I)
    _HOLIDAY        = re.compile(r"\bholiday\b|\bvacation\b|\bbreak\b", re.I)
    _PERIOD_LABEL   = re.compile(r"\bPeriod\s+\d+\b|\bI\s+Period\b|\bII\s+Period\b", re.I)
    _INSTRUCTION_DAYS = re.compile(r"\binstruction\s+days?\b|\bworking\s+days?\b", re.I)
    _ODD_EVEN_SEM   = re.compile(r"\b(?:odd|even)\s+semester\b", re.I)

    # --- Administrative ---
    _QUOTATION      = re.compile(r"\bcall\s+for\s+quotations?\b|\bsealed\s+quotations?\b|\bquotations?\b", re.I)
    _TENDER         = re.compile(r"\btender\b|\be-?tender\b|\bbidder\b|\bemd\b|\bnit\b", re.I)
    _TERMS_COND     = re.compile(r"\bterms\s+(?:and|&)\s+conditions\b|\bT\s*&\s*C\b", re.I)
    _NOTIFICATION   = re.compile(r"\bnotification\b", re.I)
    _NOTIF_NUMBER   = re.compile(r"\bNo\.?\s*:?\s*([\w/\-]+)\b", re.I)
    _OFFICE_ORDER   = re.compile(r"\boffice\s+order\b", re.I)
    _PROCEEDINGS    = re.compile(r"\bproceedings\b", re.I)
    _MINUTES        = re.compile(r"\bminutes\s+of\s+(?:the\s+)?meeting\b|\bmeeting\s+minutes\b", re.I)
    _CIRCULAR       = re.compile(r"\bcircular\b", re.I)
    _CIRCULAR_NUM   = re.compile(r"\bCircular\s+No\.?\s*([\w/\-]+)\b", re.I)
    _MEMO           = re.compile(r"\bmemo(?:randum)?\b", re.I)
    _SIGNATURE      = re.compile(r"\b(?:signed|signature|sd/-|sd-|Sd/-|authorized\s+signatory)\b", re.I)
    _SEAL           = re.compile(r"\b(?:official\s+seal|stamp|seal\s+of)\b", re.I)
    _AUTHORITY      = re.compile(r"\b(?:Registrar|Principal|Director|Dean|Controller|Vice[-\s]?Chancellor|Head\s+of\s+(?:the\s+)?Department)\b", re.I)
    _ANNEXURE       = re.compile(r"\b(?:Annexure|Annex)\s*[-–]?\s*\w*\b", re.I)
    _APPENDIX       = re.compile(r"\bAppendix\s*[-–]?\s*\w*\b", re.I)
    _ITEM_TABLE     = re.compile(r"\bSl\.?\s*No\.?\b|\bItem\s+(?:No\.?|Description)\b", re.I)

    # --- Financial ---
    _FEE            = re.compile(r"\bfee\b|\bfees\b|\btuition\b|\bchallan\b", re.I)
    _SCHOLARSHIP    = re.compile(r"\bscholarship\b|\bstipend\b", re.I)
    _REIMBURSE      = re.compile(r"\breimbursement\b|\brefund\b", re.I)
    _BANK_DETAILS   = re.compile(r"\bbank\s+(?:account|details|name)\b|\bifsc\b|\biban\b", re.I)
    _PAY_DEADLINE   = re.compile(r"\blast\s+date\s+(?:of\s+)?payment\b|\bpay\s+before\b|\bdue\s+date\b", re.I)
    _AMOUNT         = re.compile(r"₹\s*\d+|\bRs\.?\s*\d+|\bINR\s*\d+", re.I)
    _INCOME         = re.compile(r"\bannual\s+income\b|\bfamily\s+income\b|\bincome\s+certificate\b", re.I)
    _ELIGIBILITY    = re.compile(r"\beligibility\s+(?:criteria|requirements?|conditions?)\b|\beligible\s+students?\b", re.I)
    _DOCS_REQUIRED  = re.compile(r"\bdocuments?\s+required\b|\bdocuments?\s+to\s+be\s+submitted\b", re.I)

    # --- Events ---
    _WORKSHOP       = re.compile(r"\bworkshop\b", re.I)
    _SEMINAR        = re.compile(r"\bseminar\b", re.I)
    _CONFERENCE     = re.compile(r"\bconference\b|\bsymposium\b", re.I)
    _FDP            = re.compile(r"\bfaculty\s+development\s+program(?:me)?\b|\bFDP\b", re.I)
    _PLACEMENT      = re.compile(r"\bplacement\b|\bcampus\s+drive\b|\brecruitment\s+drive\b", re.I)
    _RESOURCE_PERSON= re.compile(r"\bresource\s+person\b|\bguest\s+speaker\b|\bkeynote\s+speaker\b", re.I)
    _COORDINATOR    = re.compile(r"\bco-?ordinator\b|\bconvener\b|\borganizing\s+committee\b", re.I)
    _REG_DEADLINE   = re.compile(r"\bregistration\s+(?:deadline|last\s+date|closes?)\b", re.I)
    _VENUE          = re.compile(r"\bvenue\s*[:\-]?\s*([^\n,;]{3,60})", re.I)
    _ORGANIZER      = re.compile(r"\borganized\s+by\s*[:\-]?\s*([^\n,;]{3,60})", re.I)
    _EVENT_NAME     = re.compile(r"\b(?:National|International|State)\s+(?:Conference|Seminar|Workshop|Symposium)\b", re.I)
    _HANDS_ON       = re.compile(r"\bhands[\-\s]?on\b|\bpractical\s+session\b", re.I)

    # --- Student ---
    _ADMISSIONS     = re.compile(r"\badmission\b|\benrolment\b|\bapplication\s+form\b|\bprospectus\b", re.I)
    _HOSTEL         = re.compile(r"\bhostel\b|\bwarden\b|\bmess\s+(?:menu|committee|facility)\b", re.I)
    _RESULTS        = re.compile(r"\bexamination\s+results?\b|\bresult\s+(?:declared|published|notification)\b|\bmarks\s+statement\b", re.I)
    _INTERNSHIP     = re.compile(r"\binternship\b|\bindustrial\s+training\b|\bintern\b", re.I)

    # --- Report ---
    _REPORT_SECTIONS = re.compile(
        r"\b(Executive\s+Summary|Objectives?|Methodology|Findings|Observations?|Results?|Recommendations?|Conclusions?|Annexures?)\b",
        re.I,
    )

    # --- Contact ---
    _PHONE          = re.compile(r"\b(?:\+91[\-\s]?)?\d{10}\b|\b0\d{2,4}[\-\s]\d{6,8}\b")
    _EMAIL          = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
    _URL            = re.compile(r"\bhttps?://[\w./?=%&+#-]+", re.I)

    # --- Metadata ---
    _UNIVERSITY     = re.compile(r"\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)* University)\b")
    _DEPARTMENT     = re.compile(r"\bDepartment\s+of\s+([A-Za-z &]+)", re.I)
    _PROGRAM        = re.compile(r"\b(B\.?Tech\.?|M\.?Tech\.?|B\.?Sc\.?|M\.?Sc\.?|MBA|MCA|BCA|Ph\.?D\.?)\b", re.I)
    _SEMESTER       = re.compile(r"\b(?:Semester|Sem\.?)\s*[-–]?\s*([IVX]+|\d+)\b", re.I)
    _ACADEMIC_YEAR  = re.compile(r"\b(20\d{2}\s*[-–]\s*20\d{2}|20\d{2}\s*[-–]\s*\d{2})\b")
    _TABLE_HEADER   = re.compile(r"^\|?\s*(?:S\.?No\.?|Sl\.?\s*No\.?|Subject|Course|Date|Day|Time|Room|Venue|Fee|Amount)\s*[\|]", re.I | re.M)

    def extract(self, pages: list[PdfPage]) -> FeatureMap:
        """Extract all features. Never classifies."""
        fm = FeatureMap()
        fm.page_count = len(pages)
        full = "\n".join(p.content for p in pages)

        fm.line_count = len(full.splitlines())
        fm.word_count = _word_count(full)
        fm.table_count = len(self._TABLE_TAG.findall(full))

        # --- Academic ---
        raw_codes = self._COURSE_CODE.findall(full)
        fm.course_codes = list({c.strip().upper() for c in raw_codes})
        fm.course_code_count = len(fm.course_codes)

        unit_matches = self._UNIT.findall(full)
        fm.unit_numbers = [_canonical_unit_label(u) for u in unit_matches]
        fm.unit_count = len(fm.unit_numbers)
        fm.module_count = len(re.findall(r"\bMODULE\s*[-–—]?\s*\d+\b", full, re.I))
        chapter_matches = self._CHAPTER_NUM.findall(full)
        fm.chapter_numbers = list(dict.fromkeys(chapter_matches))
        fm.chapter_count = len(fm.chapter_numbers)
        fm.lab_count = len(self._LAB.findall(full))
        fm.course_objectives_present = bool(self._COURSE_OBJ.search(full))
        fm.course_outcomes_present = bool(self._COURSE_OUT.search(full))
        fm.textbook_section_count = len(self._TEXTBOOKS.findall(full))
        fm.reference_section_count = len(self._REFERENCES.findall(full))
        fm.bibliography_present = bool(re.search(r"\bbibliography\b", full, re.I))
        fm.teaching_scheme_present = bool(self._TEACHING_SCH.search(full))
        fm.examination_scheme_present = bool(self._EXAM_SCH.search(full))
        fm.credits_present = bool(self._CREDITS.search(full))
        fm.ltpc_table_present = bool(self._LTPC.search(full))
        fm.rule_section_count = len(self._RULE_SECTION.findall(full))
        fm.clause_count = len(self._CLAUSE.findall(full))
        fm.attendance_rules = bool(self._ATTENDANCE.search(full))
        fm.grading_rules = bool(self._GRADING.search(full))

        # --- Schedule ---
        date_matches = self._DATE.findall(full)
        fm.date_values = date_matches
        fm.date_count = len(date_matches)
        time_matches = self._TIME.findall(full)
        fm.time_values = time_matches
        fm.time_count = len(time_matches)
        fm.am_pm_count = len(self._AM_PM.findall(full))
        fm.day_names = list({m.title() for m in self._DAY_NAMES.findall(full)})
        fm.month_names = list({m.title() for m in self._MONTH_NAMES.findall(full)})
        fm.session_labels = list({m[0] if isinstance(m, tuple) else m for m in self._SESSION.findall(full)})
        fm.room_numbers = list({m for m in self._ROOM.findall(full)})
        fm.exam_time_patterns = len(self._EXAM_TIME_SLOT.findall(full))
        holiday_matches = self._HOLIDAY.findall(full)
        fm.holiday_patterns = len(holiday_matches)
        # Collect distinct holiday-adjacent lines as holiday entries
        fm.holiday_entries = _collect_holiday_entries(full)
        fm.period_labels = list({m if isinstance(m, str) else m[0] for m in self._PERIOD_LABEL.findall(full)})
        fm.instruction_days_present = bool(self._INSTRUCTION_DAYS.search(full))
        fm.odd_even_semester_present = bool(self._ODD_EVEN_SEM.search(full))

        # --- Administrative ---
        fm.quotation_present = bool(self._QUOTATION.search(full))
        fm.tender_present = bool(self._TENDER.search(full))
        fm.terms_conditions_present = bool(self._TERMS_COND.search(full))
        fm.notification_present = bool(self._NOTIFICATION.search(full))
        notif_num = self._NOTIF_NUMBER.search(full)
        fm.notification_number = notif_num.group(1) if notif_num else ""
        fm.office_order_present = bool(self._OFFICE_ORDER.search(full))
        fm.proceedings_present = bool(self._PROCEEDINGS.search(full))
        fm.minutes_present = bool(self._MINUTES.search(full))
        fm.circular_present = bool(self._CIRCULAR.search(full))
        circ_num = self._CIRCULAR_NUM.search(full)
        fm.circular_number = circ_num.group(1) if circ_num else ""
        fm.memo_present = bool(self._MEMO.search(full))
        fm.signatures_present = bool(self._SIGNATURE.search(full))
        fm.official_seal_present = bool(self._SEAL.search(full))
        fm.authorities = list({m if isinstance(m, str) else m[0] for m in self._AUTHORITY.findall(full)})
        fm.annexure_count = len(self._ANNEXURE.findall(full))
        fm.appendix_count = len(self._APPENDIX.findall(full))
        fm.item_table_count = len(self._ITEM_TABLE.findall(full))

        # --- Financial ---
        fm.fee_present = bool(self._FEE.search(full))
        fm.scholarship_present = bool(self._SCHOLARSHIP.search(full))
        fm.reimbursement_present = bool(self._REIMBURSE.search(full))
        fm.bank_details_present = bool(self._BANK_DETAILS.search(full))
        fm.payment_deadline_present = bool(self._PAY_DEADLINE.search(full))
        fm.amount_count = len(self._AMOUNT.findall(full))
        fm.income_criterion_present = bool(self._INCOME.search(full))
        fm.eligibility_present = bool(self._ELIGIBILITY.search(full))
        fm.documents_required_present = bool(self._DOCS_REQUIRED.search(full))

        # --- Events ---
        fm.workshop_present = bool(self._WORKSHOP.search(full))
        fm.seminar_present = bool(self._SEMINAR.search(full))
        fm.conference_present = bool(self._CONFERENCE.search(full))
        fm.fdp_present = bool(self._FDP.search(full))
        fm.placement_present = bool(self._PLACEMENT.search(full))
        fm.resource_person_present = bool(self._RESOURCE_PERSON.search(full))
        fm.coordinator_present = bool(self._COORDINATOR.search(full))
        reg_dl = self._REG_DEADLINE.search(full)
        fm.registration_deadline = reg_dl.group(0) if reg_dl else ""
        venue_m = self._VENUE.search(full)
        fm.venue = _normalize_whitespace(venue_m.group(1)) if venue_m else ""
        org_m = self._ORGANIZER.search(full)
        fm.organizer = _normalize_whitespace(org_m.group(1)) if org_m else ""
        ev_m = self._EVENT_NAME.search(full)
        fm.event_name = ev_m.group(0) if ev_m else ""

        # --- Student ---
        fm.admissions_present = bool(self._ADMISSIONS.search(full))
        fm.hostel_present = bool(self._HOSTEL.search(full))
        fm.results_present = bool(self._RESULTS.search(full))
        fm.internship_present = bool(self._INTERNSHIP.search(full))

        # --- Report ---
        fm.report_sections = list(dict.fromkeys(
            m if isinstance(m, str) else m[0]
            for m in self._REPORT_SECTIONS.findall(full)
        ))

        # --- Contact ---
        fm.contact_numbers = list(dict.fromkeys(self._PHONE.findall(full)))
        fm.emails = list(dict.fromkeys(self._EMAIL.findall(full)))
        fm.urls = list(dict.fromkeys(self._URL.findall(full)))
        fm.table_headers = list(dict.fromkeys(
            _normalize_whitespace(m) for m in self._TABLE_HEADER.findall(full)
        ))

        # --- Metadata ---
        u = self._UNIVERSITY.search(full)
        fm.university_name = u.group(1) if u else ""
        dept_matches = self._DEPARTMENT.findall(full)
        fm.department_names = list(dict.fromkeys(_normalize_whitespace(d) for d in dept_matches))
        p = self._PROGRAM.search(full)
        fm.program_name = p.group(1) if p else ""
        s = self._SEMESTER.search(full)
        fm.semester = s.group(1) if s else ""
        ay = self._ACADEMIC_YEAR.search(full)
        fm.academic_year = ay.group(1) if ay else ""

        return fm


def _collect_holiday_entries(full_text: str) -> list[str]:
    """Return up to 20 unique lines that contain the word 'holiday' or 'vacation'."""
    entries: list[str] = []
    seen: set[str] = set()
    for line in full_text.splitlines():
        normalized = _normalize_whitespace(line)
        if re.search(r"\bholiday\b|\bvacation\b", normalized, re.I):
            key = normalized.lower()[:60]
            if key not in seen and normalized:
                seen.add(key)
                entries.append(normalized)
            if len(entries) >= 20:
                break
    return entries


# ===========================================================================
# PHASE 9 – DOCUMENT TYPE REGISTRY
# (Defined early so scorers/validators/chunkers can register themselves)
# ===========================================================================

@dataclass
class DocumentTypeProfile:
    """
    One registration per document type.
    Adding a new type = instantiating one of these and calling registry.register().
    """
    doc_type: str
    category: str
    scorer: Any          # _BaseScorer
    validator: Any       # _BaseValidator
    chunker: Any         # _BaseChunker
    description: str = ""


class DocumentTypeRegistry:
    """
    Central registry: single source of truth for all supported document types.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, DocumentTypeProfile] = {}

    def register(self, profile: DocumentTypeProfile) -> None:
        self._profiles[profile.doc_type] = profile

    def get_scorer(self, doc_type: str) -> Any:
        p = self._profiles.get(doc_type)
        return p.scorer if p else None

    def get_validator(self, doc_type: str) -> Any:
        p = self._profiles.get(doc_type)
        return p.validator if p else None

    def get_chunker(self, doc_type: str) -> Any:
        p = self._profiles.get(doc_type)
        return p.chunker if p else None

    def all_scorers(self) -> list[Any]:
        return [p.scorer for p in self._profiles.values()]

    def all_doc_types(self) -> list[str]:
        return list(self._profiles.keys())

    def all_profiles(self) -> list[DocumentTypeProfile]:
        return list(self._profiles.values())


# Singleton registry — populated at module load time after classes are defined.
REGISTRY = DocumentTypeRegistry()


# ===========================================================================
# PHASE 2 – CANDIDATE SCORERS
# (Each type has one independent scorer class with strong/weak bookkeeping)
# ===========================================================================

class _BaseScorer:
    doc_type: str = ""

    def score(self, fm: FeatureMap) -> CandidateScore:
        raise NotImplementedError

    # helpers
    @staticmethod
    def _strong(cs: CandidateScore, pts: int, label: str) -> None:
        cs.score += pts
        cs.strong_evidence_count += 1
        cs.evidence.append(f"[S] {label}")

    @staticmethod
    def _weak(cs: CandidateScore, pts: int, label: str) -> None:
        cs.score += pts
        cs.weak_evidence_count += 1
        cs.evidence.append(f"[W] {label}")


class SyllabusScorer(_BaseScorer):
    doc_type = DOC_TYPE_SYLLABUS

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.course_code_count > 0:
            self._strong(cs, 8, f"course_codes={fm.course_code_count}")
        if fm.unit_count >= 3:
            self._strong(cs, 7, f"unit_count={fm.unit_count}")
        if fm.textbook_section_count > 0:
            self._strong(cs, 5, f"textbook_sections={fm.textbook_section_count}")
        if fm.reference_section_count > 0:
            self._strong(cs, 5, f"reference_sections={fm.reference_section_count}")
        if fm.course_outcomes_present:
            self._strong(cs, 4, "course_outcomes")
        if fm.course_objectives_present:
            self._strong(cs, 4, "course_objectives")
        if fm.ltpc_table_present:
            self._strong(cs, 5, "ltpc_table")
        if fm.teaching_scheme_present:
            self._strong(cs, 4, "teaching_scheme")
        # Weak
        if fm.semester:
            self._weak(cs, 1, "semester")
        if fm.department_names:
            self._weak(cs, 1, "department")
        if fm.credits_present:
            self._weak(cs, 2, "credits")
        if fm.examination_scheme_present:
            self._weak(cs, 2, "examination_scheme")
        return cs


class LabManualScorer(_BaseScorer):
    doc_type = DOC_TYPE_LAB_MANUAL

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.lab_count >= 3:
            self._strong(cs, 10, f"lab_count={fm.lab_count}")
        if fm.unit_count >= 1:
            self._strong(cs, 6, f"unit_count={fm.unit_count}")
        if fm.course_outcomes_present:
            self._strong(cs, 4, "course_outcomes")
        if fm.course_objectives_present:
            self._strong(cs, 4, "course_objectives")
        # Weak
        if fm.semester:
            self._weak(cs, 1, "semester")
        if fm.course_code_count > 0:
            self._weak(cs, 2, "course_codes")
        return cs


class AcademicCalendarScorer(_BaseScorer):
    doc_type = DOC_TYPE_ACADEMIC_CAL

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.academic_year:
            self._strong(cs, 6, f"academic_year={fm.academic_year}")
        if fm.month_names and len(fm.month_names) >= 3:
            self._strong(cs, 7, f"months={fm.month_names}")
        if fm.holiday_patterns >= 3:
            self._strong(cs, 5, f"holidays={fm.holiday_patterns}")
        if fm.date_count >= 10:
            self._strong(cs, 6, f"date_count={fm.date_count}")
        # Weak
        if fm.semester:
            self._weak(cs, 2, "semester")
        if fm.university_name:
            self._weak(cs, 1, "university")
        return cs


class AlmanacScorer(_BaseScorer):
    doc_type = DOC_TYPE_ALMANAC

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.odd_even_semester_present:
            self._strong(cs, 10, "odd/even_semester")
        if fm.instruction_days_present:
            self._strong(cs, 8, "instruction_days")
        if fm.holiday_patterns >= 5:
            self._strong(cs, 7, f"holiday_entries={fm.holiday_patterns}")
        if fm.academic_year:
            self._strong(cs, 5, f"academic_year={fm.academic_year}")
        # Weak
        if fm.semester:
            self._weak(cs, 2, "semester")
        if fm.university_name:
            self._weak(cs, 1, "university")
        return cs


class ExamScheduleScorer(_BaseScorer):
    doc_type = DOC_TYPE_EXAM_SCHEDULE

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.exam_time_patterns >= 2:
            self._strong(cs, 10, f"exam_time_slots={fm.exam_time_patterns}")
        if fm.date_count >= 5:
            self._strong(cs, 8, f"date_count={fm.date_count}")
        if fm.time_count >= 5:
            self._strong(cs, 8, f"time_count={fm.time_count}")
        if fm.session_labels:
            self._strong(cs, 7, f"session_labels={fm.session_labels}")
        if fm.room_numbers:
            self._strong(cs, 5, f"room_numbers={len(fm.room_numbers)}")
        if len(fm.day_names) >= 3:
            self._strong(cs, 5, f"day_names={fm.day_names}")
        if fm.am_pm_count >= 4:
            self._weak(cs, 3, f"am_pm={fm.am_pm_count}")
        # Penalise syllabus-specific structure
        if fm.unit_count >= 3 or fm.textbook_section_count > 0 or fm.course_outcomes_present:
            penalty = 6
            cs.score = max(0, cs.score - penalty)
            cs.evidence.append(f"[penalty] syllabus_features=-{penalty}")
        return cs


class ClassTimetableScorer(_BaseScorer):
    doc_type = DOC_TYPE_CLASS_TIMETABLE

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.period_labels:
            self._strong(cs, 10, f"period_labels={fm.period_labels}")
        if len(fm.day_names) >= 4:
            self._strong(cs, 8, f"day_names={fm.day_names}")
        if fm.room_numbers:
            self._strong(cs, 5, f"room_numbers={len(fm.room_numbers)}")
        if fm.course_code_count >= 2:
            self._strong(cs, 5, f"course_codes={fm.course_code_count}")
        if fm.time_count >= 5:
            self._strong(cs, 5, f"time_count={fm.time_count}")
        # Penalise for syllabus structure
        if fm.unit_count >= 3 or fm.textbook_section_count > 0:
            penalty = 4
            cs.score = max(0, cs.score - penalty)
            cs.evidence.append(f"[penalty] syllabus_features=-{penalty}")
        return cs


class RegulationsScorer(_BaseScorer):
    doc_type = DOC_TYPE_REGULATIONS

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.chapter_count >= 3:
            self._strong(cs, 8, f"chapters={fm.chapter_count}")
        if fm.rule_section_count >= 3:
            self._strong(cs, 7, f"rule_sections={fm.rule_section_count}")
        if fm.attendance_rules:
            self._strong(cs, 6, "attendance_rules")
        if fm.grading_rules:
            self._strong(cs, 6, "grading_rules")
        if fm.credits_present:
            self._strong(cs, 4, "credits")
        if fm.clause_count >= 3:
            self._strong(cs, 5, f"clauses={fm.clause_count}")
        if fm.course_code_count == 0 and fm.unit_count == 0:
            self._weak(cs, 3, "no_course_codes_or_units")
        return cs


class NotificationScorer(_BaseScorer):
    doc_type = DOC_TYPE_NOTIFICATION

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.notification_present:
            self._strong(cs, 10, "notification_keyword")
        if fm.notification_number:
            self._strong(cs, 5, f"notification_no={fm.notification_number}")
        if fm.date_count >= 1:
            self._weak(cs, 2, "date_present")
        if fm.department_names:
            self._weak(cs, 1, "department")
        return cs


class CircularScorer(_BaseScorer):
    doc_type = DOC_TYPE_CIRCULAR

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.circular_present:
            self._strong(cs, 10, "circular_keyword")
        if fm.circular_number:
            self._strong(cs, 5, f"circular_no={fm.circular_number}")
        if fm.authorities:
            self._strong(cs, 4, f"authorities={fm.authorities}")
        if fm.date_count >= 1:
            self._weak(cs, 2, "date_present")
        return cs


class OfficeOrderScorer(_BaseScorer):
    doc_type = DOC_TYPE_OFFICE_ORDER

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.office_order_present:
            self._strong(cs, 10, "office_order_keyword")
        if fm.memo_present:
            self._strong(cs, 4, "memo")
        if fm.proceedings_present:
            self._strong(cs, 3, "proceedings")
        if fm.signatures_present:
            self._weak(cs, 3, "signature")
        return cs


class ProceedingsScorer(_BaseScorer):
    doc_type = DOC_TYPE_PROCEEDINGS

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.proceedings_present:
            self._strong(cs, 10, "proceedings_keyword")
        if fm.authorities:
            self._strong(cs, 4, f"authorities={fm.authorities}")
        if fm.notification_number:
            self._strong(cs, 3, "ref_number")
        return cs


class MinutesScorer(_BaseScorer):
    doc_type = DOC_TYPE_MINUTES

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.minutes_present:
            self._strong(cs, 12, "minutes_of_meeting")
        if fm.date_count >= 1:
            self._weak(cs, 2, "date_present")
        if fm.authorities:
            self._weak(cs, 2, "authorities")
        return cs


class MemorandumScorer(_BaseScorer):
    doc_type = DOC_TYPE_MEMORANDUM

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.memo_present:
            self._strong(cs, 10, "memorandum_keyword")
        if fm.authorities:
            self._strong(cs, 4, f"authorities={fm.authorities}")
        if fm.signatures_present:
            self._weak(cs, 3, "signature")
        return cs


class ReportScorer(_BaseScorer):
    doc_type = DOC_TYPE_REPORT

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        strong_report_sections = {"Executive Summary", "Methodology", "Observations", "Findings", "Recommendations"}
        found_strong = [s for s in fm.report_sections if any(k in s for k in strong_report_sections)]
        if len(found_strong) >= 2:
            self._strong(cs, 10, f"report_sections={found_strong}")
        elif len(fm.report_sections) >= 3:
            self._strong(cs, 6, f"generic_sections={fm.report_sections}")
        if "Conclusions" in fm.report_sections or "Conclusion" in " ".join(fm.report_sections):
            self._strong(cs, 4, "conclusion_present")
        if "Objectives" in fm.report_sections:
            self._weak(cs, 3, "objectives")
        return cs


class FeeNoticeScorer(_BaseScorer):
    doc_type = DOC_TYPE_FEE_NOTICE

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.fee_present:
            self._strong(cs, 8, "fee_keyword")
        if fm.payment_deadline_present:
            self._strong(cs, 6, "payment_deadline")
        if fm.amount_count >= 2:
            self._strong(cs, 5, f"amounts={fm.amount_count}")
        if fm.bank_details_present:
            self._strong(cs, 4, "bank_details")
        return cs


class ScholarshipScorer(_BaseScorer):
    doc_type = DOC_TYPE_SCHOLARSHIP

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.scholarship_present:
            self._strong(cs, 12, "scholarship_keyword")
        if fm.eligibility_present:
            self._strong(cs, 6, "eligibility_criteria")
        if fm.income_criterion_present:
            self._strong(cs, 5, "income_criterion")
        if fm.documents_required_present:
            self._strong(cs, 4, "documents_required")
        if fm.payment_deadline_present:
            self._weak(cs, 3, "payment_deadline")
        if fm.amount_count >= 1:
            self._weak(cs, 2, f"amounts={fm.amount_count}")
        return cs


class ReimbursementScorer(_BaseScorer):
    doc_type = DOC_TYPE_REIMBURSEMENT

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.reimbursement_present:
            self._strong(cs, 12, "reimbursement_keyword")
        if fm.documents_required_present:
            self._strong(cs, 5, "documents_required")
        if fm.bank_details_present:
            self._strong(cs, 4, "bank_details")
        if fm.amount_count >= 1:
            self._weak(cs, 2, f"amounts={fm.amount_count}")
        return cs


class TenderScorer(_BaseScorer):
    doc_type = DOC_TYPE_TENDER

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.tender_present:
            self._strong(cs, 10, "tender_keyword")
        if fm.terms_conditions_present:
            self._strong(cs, 5, "terms_conditions")
        if fm.item_table_count >= 1:
            self._strong(cs, 5, f"item_tables={fm.item_table_count}")
        if fm.payment_deadline_present:
            self._weak(cs, 3, "payment_deadline")
        if fm.amount_count >= 1:
            self._weak(cs, 2, f"amounts={fm.amount_count}")
        return cs


class QuotationScorer(_BaseScorer):
    doc_type = DOC_TYPE_QUOTATION

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.quotation_present:
            self._strong(cs, 10, "quotation_keyword")
        if fm.terms_conditions_present:
            self._strong(cs, 6, "terms_conditions")
        if fm.item_table_count >= 1:
            self._strong(cs, 4, f"item_tables={fm.item_table_count}")
        if fm.payment_deadline_present:
            self._weak(cs, 4, "payment_deadline")
        if fm.amount_count >= 2:
            self._weak(cs, 3, f"amounts={fm.amount_count}")
        return cs


class WorkshopScorer(_BaseScorer):
    doc_type = DOC_TYPE_WORKSHOP

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.workshop_present:
            self._strong(cs, 10, "workshop_keyword")
        if fm.resource_person_present:
            self._strong(cs, 7, "resource_person")
        if fm.venue:
            self._strong(cs, 5, f"venue={fm.venue}")
        if fm.coordinator_present:
            self._strong(cs, 5, "coordinator")
        if fm.registration_deadline:
            self._strong(cs, 4, "registration_deadline")
        if fm.date_count >= 1:
            self._weak(cs, 2, "date")
        if fm.department_names:
            self._weak(cs, 1, "department")
        return cs


class SeminarScorer(_BaseScorer):
    doc_type = DOC_TYPE_SEMINAR

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.seminar_present:
            self._strong(cs, 10, "seminar_keyword")
        if fm.resource_person_present:
            self._strong(cs, 6, "resource_person")
        if fm.venue:
            self._strong(cs, 4, f"venue={fm.venue}")
        if fm.date_count >= 1:
            self._weak(cs, 2, "date")
        return cs


class ConferenceScorer(_BaseScorer):
    doc_type = DOC_TYPE_CONFERENCE

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.conference_present:
            self._strong(cs, 10, "conference_keyword")
        if fm.event_name:
            self._strong(cs, 7, f"event_name={fm.event_name}")
        if fm.registration_deadline:
            self._strong(cs, 5, "registration_deadline")
        if fm.venue:
            self._strong(cs, 4, f"venue={fm.venue}")
        if fm.organizer:
            self._weak(cs, 3, f"organizer={fm.organizer}")
        return cs


class FDPScorer(_BaseScorer):
    doc_type = DOC_TYPE_FDP

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.fdp_present:
            self._strong(cs, 12, "fdp_keyword")
        if fm.resource_person_present:
            self._strong(cs, 5, "resource_person")
        if fm.coordinator_present:
            self._strong(cs, 4, "coordinator")
        if fm.registration_deadline:
            self._weak(cs, 3, "registration_deadline")
        return cs


class PlacementScorer(_BaseScorer):
    doc_type = DOC_TYPE_PLACEMENT

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.placement_present:
            self._strong(cs, 12, "placement_keyword")
        if fm.venue:
            self._strong(cs, 4, f"venue={fm.venue}")
        if fm.eligibility_present:
            self._strong(cs, 4, "eligibility")
        if fm.date_count >= 1:
            self._weak(cs, 2, "date")
        return cs


class AdmissionsScorer(_BaseScorer):
    doc_type = DOC_TYPE_ADMISSIONS

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.admissions_present:
            self._strong(cs, 12, "admissions_keyword")
        if fm.eligibility_present:
            self._strong(cs, 5, "eligibility")
        if fm.documents_required_present:
            self._strong(cs, 4, "documents_required")
        if fm.payment_deadline_present:
            self._weak(cs, 3, "payment_deadline")
        return cs


class HostelScorer(_BaseScorer):
    doc_type = DOC_TYPE_HOSTEL

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.hostel_present:
            self._strong(cs, 12, "hostel_keyword")
        if fm.payment_deadline_present:
            self._weak(cs, 3, "payment_deadline")
        if fm.amount_count >= 1:
            self._weak(cs, 2, f"amounts={fm.amount_count}")
        return cs


class ExamResultsScorer(_BaseScorer):
    doc_type = DOC_TYPE_EXAM_RESULTS

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.results_present:
            self._strong(cs, 12, "results_keyword")
        if fm.course_code_count >= 1:
            self._strong(cs, 5, f"course_codes={fm.course_code_count}")
        if fm.semester:
            self._weak(cs, 3, "semester")
        return cs


class InternshipScorer(_BaseScorer):
    doc_type = DOC_TYPE_INTERNSHIP

    def score(self, fm: FeatureMap) -> CandidateScore:
        cs = CandidateScore(doc_type=self.doc_type, score=0)
        if fm.internship_present:
            self._strong(cs, 12, "internship_keyword")
        if fm.eligibility_present:
            self._strong(cs, 4, "eligibility")
        if fm.date_count >= 1:
            self._weak(cs, 2, "date")
        return cs


class CandidateScorer:
    """Phase 2 orchestrator — delegates to all registered scorers."""

    def score_all(self, fm: FeatureMap) -> list[CandidateScore]:
        results = [s.score(fm) for s in REGISTRY.all_scorers()]
        return sorted(results, key=lambda c: c.score, reverse=True)


# ===========================================================================
# PHASE 3 – CONFIDENCE ANALYZER (evidence-aware)
# ===========================================================================

class ConfidenceAnalyzer:
    """
    Phase 3: Compute confidence using score gap AND evidence quality.

    Evidence quality factors:
      - strong_evidence_count of the winner
      - contradictions: if runner-up also has strong evidence
      - structural_completeness: proportion of expected strong signals present

    Thresholds:
      HIGH   – gap >= 10 AND winner_score >= 8 AND strong_evidence >= 2
      MEDIUM – gap >= 5  AND winner_score >= 5 AND strong_evidence >= 1
      LOW    – everything else
    """

    HIGH_GAP        = 10
    HIGH_MIN_SCORE  = 8
    HIGH_MIN_STRONG = 2
    MEDIUM_GAP      = 5
    MEDIUM_MIN_SCORE = 5
    MEDIUM_MIN_STRONG = 1

    def analyze(self, ranked: list[CandidateScore]) -> ConfidenceResult:
        if not ranked:
            return ConfidenceResult(
                winner=DOC_TYPE_UNKNOWN, winner_score=0,
                runner_up=DOC_TYPE_UNKNOWN, runner_up_score=0,
                gap=0, level="LOW", ranked=ranked,
            )

        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else CandidateScore(doc_type=DOC_TYPE_UNKNOWN, score=0)
        gap = winner.score - runner_up.score

        contradictions: list[str] = []
        if runner_up.strong_evidence_count >= 2 and gap < 15:
            contradictions.append(
                f"{runner_up.doc_type} also has {runner_up.strong_evidence_count} strong signals"
            )

        strong = winner.strong_evidence_count
        weak = winner.weak_evidence_count
        structural_completeness = min(1.0, strong / max(1, strong + weak))

        if (gap >= self.HIGH_GAP and winner.score >= self.HIGH_MIN_SCORE
                and strong >= self.HIGH_MIN_STRONG and not contradictions):
            level = "HIGH"
        elif (gap >= self.MEDIUM_GAP and winner.score >= self.MEDIUM_MIN_SCORE
              and strong >= self.MEDIUM_MIN_STRONG):
            level = "MEDIUM"
        else:
            level = "LOW"

        return ConfidenceResult(
            winner=winner.doc_type,
            winner_score=winner.score,
            runner_up=runner_up.doc_type,
            runner_up_score=runner_up.score,
            gap=gap,
            level=level,
            strong_evidence_count=strong,
            weak_evidence_count=weak,
            contradictions=contradictions,
            structural_completeness=round(structural_completeness, 2),
            ranked=ranked,
        )


# ===========================================================================
# PHASE 5 – VALIDATORS
# (One per document type; PassthroughValidator for types without hard rules)
# ===========================================================================

class _BaseValidator:
    doc_type: str = ""

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        raise NotImplementedError


class SyllabusValidator(_BaseValidator):
    doc_type = DOC_TYPE_SYLLABUS

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if fm.course_code_count == 0 or fm.unit_count < 1:
            return ValidationResult(
                passed=False, final_type=DOC_TYPE_UNKNOWN,
                reason="Syllabus requires course codes AND unit markers",
            )
        if fm.textbook_section_count == 0 and fm.reference_section_count == 0:
            return ValidationResult(
                passed=False, final_type=DOC_TYPE_UNKNOWN,
                reason="Syllabus requires textbooks OR references section",
            )
        return ValidationResult(passed=True, final_type=DOC_TYPE_SYLLABUS)


class LabManualValidator(_BaseValidator):
    doc_type = DOC_TYPE_LAB_MANUAL

    _LAB_EX = re.compile(r"\b(?:Experiment|Expt|Lab\s+Exercise|Practical)\s+\d+\b", re.I)

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        full = "\n".join(p.content for p in pages)
        if fm.lab_count >= 2 or self._LAB_EX.search(full):
            return ValidationResult(passed=True, final_type=DOC_TYPE_LAB_MANUAL)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Lab manual requires lab/experiment section markers",
        )


class AlmanacValidator(_BaseValidator):
    doc_type = DOC_TYPE_ALMANAC

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if not fm.academic_year:
            return ValidationResult(
                passed=False, final_type=DOC_TYPE_UNKNOWN,
                reason="Almanac requires academic year",
            )
        if not fm.odd_even_semester_present and fm.holiday_patterns < 3:
            return ValidationResult(
                passed=False, final_type=DOC_TYPE_ACADEMIC_CAL,
                reason="Insufficient almanac structure; treating as academic_calendar",
            )
        return ValidationResult(passed=True, final_type=DOC_TYPE_ALMANAC)


class AcademicCalendarValidator(_BaseValidator):
    doc_type = DOC_TYPE_ACADEMIC_CAL

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if fm.academic_year and fm.date_count >= 5:
            return ValidationResult(passed=True, final_type=DOC_TYPE_ACADEMIC_CAL)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Academic calendar requires academic_year and multiple dates",
        )


class ExamScheduleValidator(_BaseValidator):
    doc_type = DOC_TYPE_EXAM_SCHEDULE

    _UNIT_MARKER = re.compile(r"\bUNIT\s*[-–—]?\s*(?:[IVXLC]+|\d+)\b", re.I)

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        full = "\n".join(p.content for p in pages)
        if self._UNIT_MARKER.search(full):
            return ValidationResult(
                passed=False, final_type=DOC_TYPE_SYLLABUS,
                reason="Contains UNIT markers — overriding to Syllabus",
            )
        if fm.textbook_section_count > 0 or fm.course_outcomes_present:
            return ValidationResult(
                passed=False, final_type=DOC_TYPE_SYLLABUS,
                reason="Contains textbooks/outcomes — overriding to Syllabus",
            )
        return ValidationResult(passed=True, final_type=DOC_TYPE_EXAM_SCHEDULE)


class ClassTimetableValidator(_BaseValidator):
    doc_type = DOC_TYPE_CLASS_TIMETABLE

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if (fm.period_labels or len(fm.day_names) >= 4) and fm.time_count >= 3:
            return ValidationResult(passed=True, final_type=DOC_TYPE_CLASS_TIMETABLE)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Class timetable requires period/day structure with times",
        )


class RegulationsValidator(_BaseValidator):
    doc_type = DOC_TYPE_REGULATIONS

    _ANCHORS = re.compile(
        r"\b(attendance|promotion|grading|award\s+of\s+degree|scheme\s+of\s+(?:instruction|examination))\b",
        re.I,
    )

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        full = "\n".join(p.content for p in pages)
        if self._ANCHORS.search(full):
            return ValidationResult(passed=True, final_type=DOC_TYPE_REGULATIONS)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Regulations require attendance/grading/promotion language",
        )


class QuotationValidator(_BaseValidator):
    doc_type = DOC_TYPE_QUOTATION

    _CALL_FOR = re.compile(r"\bcall\s+for\s+quotations?\b", re.I)

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        full = "\n".join(p.content for p in pages)
        if fm.terms_conditions_present or self._CALL_FOR.search(full):
            return ValidationResult(passed=True, final_type=DOC_TYPE_QUOTATION)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Quotation requires T&C or 'Call for Quotations'",
        )


class ScholarshipValidator(_BaseValidator):
    doc_type = DOC_TYPE_SCHOLARSHIP

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if fm.eligibility_present or fm.income_criterion_present or fm.reimbursement_present:
            return ValidationResult(passed=True, final_type=DOC_TYPE_SCHOLARSHIP)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Scholarship requires eligibility, income criteria, or reimbursement",
        )


class FeeNoticeValidator(_BaseValidator):
    doc_type = DOC_TYPE_FEE_NOTICE

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if fm.fee_present and (fm.amount_count >= 1 or fm.payment_deadline_present):
            return ValidationResult(passed=True, final_type=DOC_TYPE_FEE_NOTICE)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Fee notice requires fee keyword with amount or due date",
        )


class WorkshopValidator(_BaseValidator):
    doc_type = DOC_TYPE_WORKSHOP

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if fm.venue or fm.coordinator_present or fm.registration_deadline or fm.session_labels:
            return ValidationResult(passed=True, final_type=DOC_TYPE_WORKSHOP)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Workshop requires venue, coordinator, registration, or session schedule",
        )


class AlmanacValidatorForConf(_BaseValidator):
    """Alias used when almanac overrides to academic_calendar."""
    doc_type = DOC_TYPE_ACADEMIC_CAL

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        return ValidationResult(passed=True, final_type=DOC_TYPE_ACADEMIC_CAL)


class MinutesValidator(_BaseValidator):
    doc_type = DOC_TYPE_MINUTES

    _AGENDA = re.compile(r"\bagenda\b", re.I)
    _MEMBERS = re.compile(r"\bmembers\s+present\b|\battendees\b", re.I)
    _RESOLUTION = re.compile(r"\bresolution\b|\bdecision\b", re.I)

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        full = "\n".join(p.content for p in pages)
        hits = sum([
            bool(self._AGENDA.search(full)),
            bool(self._MEMBERS.search(full)),
            bool(self._RESOLUTION.search(full)),
        ])
        if hits >= 1:
            return ValidationResult(passed=True, final_type=DOC_TYPE_MINUTES)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Minutes require agenda, members present, or resolution",
        )


class ProceedingsValidator(_BaseValidator):
    doc_type = DOC_TYPE_PROCEEDINGS

    _REF_NO = re.compile(r"\bRef\.?\s*No\.?\b|\bRC\s+No\.?\b|\bProceedings\s+No\.?\b", re.I)

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        full = "\n".join(p.content for p in pages)
        if fm.authorities or self._REF_NO.search(full):
            return ValidationResult(passed=True, final_type=DOC_TYPE_PROCEEDINGS)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Proceedings require reference number or authority",
        )


class CircularValidator(_BaseValidator):
    doc_type = DOC_TYPE_CIRCULAR

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if fm.circular_present and (fm.authorities or fm.date_count >= 1):
            return ValidationResult(passed=True, final_type=DOC_TYPE_CIRCULAR)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Circular requires circular keyword with issued_by or date",
        )


class NotificationValidator(_BaseValidator):
    doc_type = DOC_TYPE_NOTIFICATION

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        if fm.notification_present and (fm.notification_number or fm.date_count >= 1):
            return ValidationResult(passed=True, final_type=DOC_TYPE_NOTIFICATION)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Notification requires notification number or date",
        )


class ReportValidator(_BaseValidator):
    doc_type = DOC_TYPE_REPORT

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        required = {"Objectives", "Summary", "Conclusion", "Recommendations", "Executive Summary"}
        present = required & {s for s in fm.report_sections}
        if len(present) >= 2:
            return ValidationResult(passed=True, final_type=DOC_TYPE_REPORT)
        return ValidationResult(
            passed=False, final_type=DOC_TYPE_UNKNOWN,
            reason="Report requires objectives, summary, conclusion, or recommendations",
        )


class PassthroughValidator(_BaseValidator):
    """Default — always accepts."""

    def __init__(self, doc_type: str) -> None:
        self.doc_type = doc_type

    def validate(self, fm: FeatureMap, pages: list[PdfPage]) -> ValidationResult:
        return ValidationResult(passed=True, final_type=self.doc_type)


class DocumentValidator:
    """
    Phase 4 orchestrator.
    Uses REGISTRY to find the right validator per type.
    HIGH confidence skips structural validation.
    """

    def validate(
        self,
        confidence: ConfidenceResult,
        fm: FeatureMap,
        pages: list[PdfPage],
    ) -> ValidationResult:
        candidate = confidence.winner

        if confidence.level == "HIGH":
            return ValidationResult(passed=True, final_type=candidate, reason="HIGH confidence — accepted")

        validator = REGISTRY.get_validator(candidate) or PassthroughValidator(candidate)
        result = validator.validate(fm, pages)

        # If rejected with an override suggestion, validate the override too
        if not result.passed and result.final_type not in (DOC_TYPE_UNKNOWN, candidate):
            override_v = REGISTRY.get_validator(result.final_type) or PassthroughValidator(result.final_type)
            override_r = override_v.validate(fm, pages)
            if override_r.passed:
                return ValidationResult(
                    passed=True,
                    final_type=result.final_type,
                    reason=f"Override from {candidate}: {result.reason}",
                )

        return result


# ===========================================================================
# PHASE 5 – FINAL CLASSIFICATION (single source of truth)
# ===========================================================================

def build_final_classification(
    confidence: ConfidenceResult,
    validation: ValidationResult,
) -> FinalClassification:
    """
    Produce the authoritative FinalClassification from the validator output.

    CONTRACT:
      - Reads ONLY from confidence + validation.
      - Never consults data["document_type"] or any earlier prediction.
      - After this point, every downstream component receives this object.
        No plain document_type string is passed independently.
    """
    if not validation.passed and validation.final_type in (DOC_TYPE_UNKNOWN, ""):
        doc_type = DOC_TYPE_UNKNOWN
    else:
        doc_type = validation.final_type

    subtype = _DOC_CATEGORY.get(doc_type, "")

    return FinalClassification(
        document_type=doc_type,
        document_subtype=subtype,
        confidence=confidence.level,
        score=confidence.winner_score,
        runner_up=confidence.runner_up,
        runner_up_score=confidence.runner_up_score,
        gap=confidence.gap,
        strong_evidence_count=confidence.strong_evidence_count,
        validator_result="passed" if validation.passed else "failed",
        validator_reason=validation.reason,
    )


# ===========================================================================
# PHASE 6 – BANNER DETECTION (dynamic frequency-based + pattern-based)
# ===========================================================================

# Exact tokens always considered structural noise
_BANNER_EXACT_TOKENS: frozenset[str] = frozenset({
    "l t p c", "internal marks", "external marks", "teaching scheme",
    "examination scheme", "course code", "course title", "category",
    "date:", "s.no", "sl.no", "sl. no.",
})

# Pattern-based static banners (university names, dept headers, year lines, etc.)
_BANNER_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*[A-Z][A-Za-z\s]+ University\s*$"),
    re.compile(r"^\s*(?:Faculty|College|School|Institute)\s+of\s+[A-Za-z\s&]+\s*$", re.I),
    re.compile(r"^\s*Department\s+of\s+[A-Za-z\s&]+\s*$", re.I),
    re.compile(r"^\s*B\.?\s*Tech\.?\s*(?:\(.*?\))?\s*(?:\d+(?:st|nd|rd|th)\s+Year\s*)?(?:[-–].*)?$", re.I),
    re.compile(r"^\s*(?:Odd|Even)\s+Semester\s*[-–]?\s*(?:20\d{2}[-–]\d{2,4})?\s*$", re.I),
    re.compile(r"^\s*(?:SEMESTER\s+[-–]?\s*(?:[IVXLC]+|\d+))\s*$", re.I),
    re.compile(r"^\s*\(?Common\s+to\s+All\s+Branches\)?\s*$", re.I),
    re.compile(r"^\s*Academic\s+Year\s*(?:[-:]\s*\d{4}[-–]\d{2,4})?\s*$", re.I),
    re.compile(r"^\s*(?:Structure\s+of\s+Curriculum|Scheme\s+and\s+Credits?)\s*$", re.I),
    re.compile(r"^\s*(?:20\d{2}[-–]\d{2,4})\s*$"),
    re.compile(r"^\s*\d{6}\s*$"),
    re.compile(r"^\s*Page\s+\d+\s+(?:of\s+\d+)?\s*$", re.I),
)

_TABLE_MARKER_PATTERN = re.compile(r"^\[TABLE\s+\d+\]$", re.I)

# Lines that must NEVER be removed — subject names, course codes, UNIT headings
_PROTECTED_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b(?:UNIT|MODULE|CHAPTER)\s*[-–—]?\s*(?:[IVXLC]+|\d+)\b", re.I),
    re.compile(r"\b[A-Z]{2,6}\s*[-/]?\s*\d{3,5}[A-Z]{0,5}\b"),
    re.compile(r"\bNotification\s+No\b", re.I),
    re.compile(r"\bCircular\s+No\b", re.I),
    re.compile(r"\bOfficial\s+Title\b", re.I),
)


def _is_protected(text: str) -> bool:
    return any(p.search(text) for p in _PROTECTED_PATTERNS)


def _build_dynamic_banners(pages: list[PdfPage], min_pages: int = 2) -> frozenset[str]:
    """
    Item 4: Detect repeated headers/footers dynamically.
    A normalized line that appears on >= min_pages different pages is a banner.
    Protected lines (course codes, UNIT headings, subject names) are never added.
    """
    from collections import Counter
    page_sets: dict[str, set[int]] = {}
    for page in pages:
        for raw in page.content.splitlines():
            norm = _normalize_whitespace(raw)
            if not norm or _is_protected(norm):
                continue
            if norm not in page_sets:
                page_sets[norm] = set()
            page_sets[norm].add(page.page_number)
    return frozenset(
        norm for norm, page_nos in page_sets.items()
        if len(page_nos) >= min_pages
    )


# Module-level cache — populated per document in _join_lines_filtered
_DYNAMIC_BANNERS: frozenset[str] = frozenset()


def _is_banner_or_table_noise(text: str, dynamic_banners: frozenset[str] = frozenset()) -> bool:
    """Return True if this line is noise and should be stripped from chunk content."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return True
    if _is_protected(normalized):
        return False
    if _TABLE_MARKER_PATTERN.fullmatch(normalized):
        return True
    lower = normalized.lower()
    if lower in _BANNER_EXACT_TOKENS:
        return True
    for pat in _BANNER_PATTERNS:
        if pat.match(normalized):
            return True
    if normalized in dynamic_banners:
        return True
    return False


# ===========================================================================
# PHASE 6 – LINE RECORDS + SECTION UTILITIES
# ===========================================================================

_SYLLABUS_MODIFIER_PATTERN = re.compile(r"^\s*(\(?\s*(?:LAB|THEORY|PRACTICAL)\s*\.?\s*\)?)?\s*$", re.I)

# Item 3: Strict course code — must be letter-prefix (2-6 chars) + optional separator + 3-5 digits + optional suffix.
# Rejects: "Zynq 7000", "Week 5", "Chapter 3", pure numbers, product names.
_COURSE_CODE_PATTERN = re.compile(
    r"\b([A-Z]{2,6}\s*[-/]?\s*\d{3,5}[A-Z]{0,3})\b",
    re.I,
)
# A valid course code must start with 2+ letters and contain exactly one numeric block
_VALID_COURSE_CODE = re.compile(
    r"^[A-Z]{2,6}\s*[-/]?\s*\d{3,5}[A-Z]{0,3}$"
)

_SYLLABUS_COMMON_WORDS = {
    "comparisons", "classification", "regression", "entropy", "precision", "recall",
    "introduction", "background", "conclusion", "summary", "overview", "basics",
}
_SYLLABUS_CONTEXT_MARKERS = (
    re.compile(r"^(?:UNIT|MODULE|CHAPTER)\s*[-–—]?\s*[IVXLC]+\b", re.I),
    re.compile(r"^(?:UNIT|MODULE|CHAPTER)\s+\d+\b", re.I),
    re.compile(r"^COURSE\s+(?:OBJECTIVES?|OUTCOMES?)", re.I),
    re.compile(r"^LAB\s+EXERCISES?", re.I),
    re.compile(r"^WEEK\s+\d+", re.I),
)

# Item 2: signals that confirm a line is a real subject heading (not a table row)
_TEACHING_SCHEME_PAT  = re.compile(r"\bteaching\s+scheme\b", re.I)
_EXAM_SCHEME_PAT      = re.compile(r"\bexamination\s+scheme\b", re.I)
_UNIT_ONE_PAT         = re.compile(r"\bUNIT\s*[-–—]?\s*(?:I|1)\b", re.I)

# Item 1: Curriculum table signals — presence of these near a course code means it's a table row
_CURRICULUM_TABLE_SIGNALS = re.compile(
    r"\b(L\s+T\s+P\s+C|Teaching\s+Scheme|Examination\s+Scheme|Credits?|Max\.?\s*Marks?|Sl\.?\s*No\.?)\b",
    re.I,
)


def _build_line_records(pages: list[PdfPage]) -> list[LineRecord]:
    records: list[LineRecord] = []
    for page in pages:
        for line in page.content.splitlines():
            records.append(LineRecord(page_number=page.page_number, text=line.rstrip()))
    return records


def _join_lines(
    lines: list[LineRecord],
    dynamic_banners: frozenset[str] = frozenset(),
) -> tuple[str, bool]:
    """
    Returns (content, header_removed).
    header_removed=True when at least one line was stripped as a banner.
    """
    kept: list[str] = []
    header_removed = False
    for line in lines:
        if _is_banner_or_table_noise(line.text, dynamic_banners):
            header_removed = True
        else:
            kept.append(line.text)
    return "\n".join(kept).strip(), header_removed


def _page_range_from_lines(lines: list[LineRecord]) -> tuple[int, int]:
    pages = [line.page_number for line in lines if line.text.strip()]
    if not pages:
        return 0, 0
    return min(pages), max(pages)


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


def _section_spans(
    lines: list[LineRecord], markers: list[HeadingMarker]
) -> list[tuple[int, int, HeadingMarker]]:
    if not markers:
        return []
    spans: list[tuple[int, int, HeadingMarker]] = []
    for idx, marker in enumerate(markers):
        end = markers[idx + 1].index if idx + 1 < len(markers) else len(lines)
        start = 0 if idx == 0 and marker.index > 0 else marker.index
        spans.append((start, end, marker))
    return spans


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


# ===========================================================================
# PHASE 6 – SYLLABUS-SPECIFIC HELPERS
# ===========================================================================

def _syllabus_marker(text: str) -> tuple[str, str] | None:
    normalized = _normalize_whitespace(text)

    unit_match = re.match(r"^(?:UNIT|MODULE|CHAPTER)\s*[-–—]?\s*([IVXLC]+|\d+)\b", normalized, re.I)
    if unit_match:
        return "unit", f"UNIT-{_canonical_unit_label(unit_match.group(1))}"

    for pattern, chunk_type in [
        (r"^(?:SUGGESTED\s+)?TEXT\s*BOOKS?[:]?$", "textbooks"),
        (r"^(?:SUGGESTED\s+)?TEXTBOOKS?[:]?$", "textbooks"),
        (r"^PRESCRIBED\s+BOOKS?[:]?$", "textbooks"),
        (r"^RECOMMENDED\s+BOOKS?[:]?$", "textbooks"),
        (r"^LEARNING\s+RESOURCES[:]?$", "textbooks"),
    ]:
        if re.fullmatch(pattern, normalized, re.I):
            return chunk_type, "TEXT BOOKS"

    for pattern, chunk_type in [
        (r"^(?:SUGGESTED\s+)?REFERENCE\s*BOOKS?[:]?$", "references"),
        (r"^(?:SUGGESTED\s+)?REFERENCES?[:]?$", "references"),
        (r"^BIBLIOGRAPHY[:]?$", "references"),
        (r"^SUGGESTED\s+READINGS?[:]?$", "references"),
    ]:
        if re.fullmatch(pattern, normalized, re.I):
            return chunk_type, "REFERENCES"

    for pattern, chunk_type in [
        (r"^COURSE\s+OBJECTIVES?[:]?$", "course_objectives"),
        (r"^COURSE\s+OUTCOMES?[:]?$", "course_outcomes"),
        (r"^LAB\s+EXERCISES?[:]?$", "lab_exercises"),
        (r"^DETAILED\s+CONTENTS?[:]?$", "section"),
    ]:
        if re.fullmatch(pattern, normalized, re.I):
            return chunk_type, normalized.upper().rstrip(":")

    return None


def _is_valid_course_code(raw: str) -> bool:
    """Item 3: Strictly validate a matched course code token."""
    cleaned = _normalize_whitespace(raw).replace(" ", "").replace("-", "").replace("/", "")
    return bool(_VALID_COURSE_CODE.match(cleaned))


def _in_curriculum_table(lines: list[LineRecord], index: int, window: int = 8) -> bool:
    """
    Item 1: Return True when the line at `index` sits inside a curriculum overview table.
    A curriculum table is identified by L T P C / Teaching Scheme / Credits signals
    appearing within `window` lines above or below.
    """
    lo = max(0, index - window)
    hi = min(len(lines), index + window + 1)
    for i in range(lo, hi):
        if i == index:
            continue
        if _CURRICULUM_TABLE_SIGNALS.search(lines[i].text):
            return True
    return False


def _subject_has_scheme_or_unit_context(lines: list[LineRecord], start_index: int, current_course_code: Optional[str] = None) -> bool:
    """
    Item 2: Check if the next 20 lines contain Teaching Scheme, Exam Scheme, or UNIT-I.
    Used to confirm that a heading is a real subject start, not a table row.
    """
    end = min(len(lines), start_index + 20)
    for i in range(start_index + 1, end):
        text = lines[i].text
        if _TEACHING_SCHEME_PAT.search(text):
            return True
        if _EXAM_SCHEME_PAT.search(text):
            return True
        if _UNIT_ONE_PAT.search(text):
            return True
            
        if current_course_code:
            codes = _COURSE_CODE_PATTERN.findall(text)
            valid_codes = [c for c in codes if _is_valid_course_code(c)]
            if any(c.strip().upper().replace(" ", "") != current_course_code.upper().replace(" ", "") for c in valid_codes):
                return False
    return False


def _has_syllabus_context(lines: list[LineRecord], start_index: int, current_course_code: Optional[str] = None) -> bool:
    end_search = min(len(lines), start_index + 61)
    for i in range(start_index + 1, end_search):
        text = _normalize_whitespace(lines[i].text)
        if any(marker.search(text) for marker in _SYLLABUS_CONTEXT_MARKERS):
            return True
        if _syllabus_marker(text):
            return True
            
        if current_course_code:
            codes = _COURSE_CODE_PATTERN.findall(text)
            valid_codes = [c for c in codes if _is_valid_course_code(c)]
            if any(c.strip().upper().replace(" ", "") != current_course_code.upper().replace(" ", "") for c in valid_codes):
                return False
    return False


def _normalize_course_code(code: str) -> str:
    return _normalize_whitespace(code).upper().lstrip("/").replace(" ", "")


def _normalize_subject_name(text: str) -> str:
    normalized = _normalize_whitespace(text)
    is_lab = bool(re.search(r"\b(LAB|PRACTICAL|LABORATORY)\b", normalized, re.I))
    normalized = re.sub(r"\s+\d+(?:\s+[\d.]+)*\s*$", "", normalized).strip()
    normalized = re.sub(r"[–—\-*:\s]+$", "", normalized).strip()
    normalized = re.sub(r"\s*\(.*?\)\s*$", "", normalized)
    normalized = normalized.rstrip(":").strip()
    name = normalized.title()
    if is_lab and all(term not in name for term in ("Lab", "Practical", "Laboratory")):
        name += " Lab"
    return name


def _is_syllabus_subject_candidate(text: str, has_course_code: bool = False) -> bool:
    normalized = _normalize_whitespace(text)
    if not normalized or len(normalized) > 80:
        return False
    upper = normalized.upper()
    if normalized.startswith("[TABLE"):
        return False
    excluded = (
        "UNIVERSITY", "COLLEGE", "FACULTY", "SEMESTER", "COURSE", "SCHEME",
        "CREDITS", "MARKS", "TEACHING", "EXAMINATION", "DETAILED CONTENTS",
        "FIRST YEAR", "COMMON TO ALL BRANCHES", "STRUCTURE OF CURRICULUM",
        "MAXIMUM HOURS", "CATEGORY", "TITLE", "OBJECTIVES", "OUTCOMES",
        "TEXT BOOKS", "REFERENCE BOOKS", "REFERENCES", "UNIT", "THEORY",
    )
    if any(token in upper for token in excluded):
        return False
    if _syllabus_marker(normalized):
        return False
    if _SYLLABUS_MODIFIER_PATTERN.fullmatch(normalized):
        return False
    if not any(c.isalpha() for c in normalized):
        return False
    if not has_course_code:
        if len(normalized) < 3:
            return False
        words = normalized.lower().split()
        if len(words) == 1 and words[0] in _SYLLABUS_COMMON_WORDS:
            return False
    words = normalized.split()
    if not 1 <= len(words) <= 10:
        return False
    if normalized.endswith("."):
        return False
    return (
        _is_all_caps_heading(normalized)
        or normalized == normalized.title()
        or bool(re.fullmatch(r"[A-Za-z]+(?:[-&][A-Za-z0-9]+)*(?:\s*[-–—]\s*[A-Za-z0-9]+)?", normalized))
    )


def _is_false_subject(text: str) -> bool:
    upper = text.upper()
    false_indicators = [
        "FACULTY", "ACADEMIC YEAR", "SCHEME", "ELECTIVE", "COURSE LIST",
        "TEXT BOOK", "REFERENCE", "AUTHOR", "PUBLISHER", "SEMESTER",
        "INTERNSHIP", "CREDIT", "TOTAL", "MARKS", "L T P C",
        "UNIVERSITY", "DEPARTMENT", "EVALUATION"
    ]
    return any(ind in upper for ind in false_indicators)


def _extract_subject_title(lines: list[LineRecord], line_index: int, course_code: str) -> Optional[str]:
    line_text = lines[line_index].text
    clean_text = _normalize_whitespace(line_text.replace(course_code, ""))
    candidate = re.sub(r"^[–—\-:\s]+", "", clean_text).strip()
    candidate = re.sub(r"\s*\(.*?\)\s*$", "", candidate).strip()
    if len(candidate) >= 3 and not any(
        token in candidate.upper() for token in ("COURSE CODE", "COURSE TITLE", "CATEGORY")
    ) and not _is_false_subject(candidate):
        return candidate
    # Search nearby lines (above and below)
    search_indices = [
        line_index - 1, line_index + 1,
        line_index - 2, line_index + 2,
        line_index - 3, line_index + 3
    ]
    for idx in search_indices:
        if 0 <= idx < len(lines):
            text = _normalize_whitespace(lines[idx].text)
            if not text or _SYLLABUS_MODIFIER_PATTERN.fullmatch(text):
                continue
            if _is_banner_or_table_noise(text):
                continue
            if 3 <= len(text) <= 80 and not _is_false_subject(text):
                return text
    return None


def _find_syllabus_subject_markers(
    lines: list[LineRecord], unit_markers: list[HeadingMarker]
) -> list[SubjectMarker]:
    """
    Items 1, 2, 3: Find real subject headings — not curriculum table rows.

    A candidate is accepted only when:
      - It has a strictly valid course code (Item 3), AND
      - It is NOT inside a curriculum table (Item 1), AND
      - It is confirmed by Teaching Scheme / Exam Scheme / UNIT-I nearby,
        OR has broader syllabus context (Item 2).
      - The line does not contain multiple different course codes (Item 2).
    """
    markers: list[SubjectMarker] = []
    seen_slugs: set[str] = set()

    for i, line in enumerate(lines):
        text = line.text
        if not text.strip():
            continue
        if text.startswith("|") or " | " in text:
            continue

        all_codes = _COURSE_CODE_PATTERN.findall(text)
        valid_codes = [c for c in all_codes if _is_valid_course_code(c)]

        if valid_codes:
            # Item 2: reject if multiple distinct codes appear on one line
            unique_codes = list(dict.fromkeys(c.upper().replace(" ", "") for c in valid_codes))
            if len(unique_codes) > 1:
                continue

            course_code = valid_codes[0].strip()

            # Item 1: skip lines inside a curriculum overview table
            if _in_curriculum_table(lines, i):
                continue

            # Item 2: must be confirmed by scheme/unit context or broad syllabus context
            confirmed = (
                _subject_has_scheme_or_unit_context(lines, i, course_code)
                or _has_syllabus_context(lines, i, course_code)
            )
            if not confirmed:
                continue

            extracted_name = _extract_subject_title(lines, i, course_code)
            subject_name = _normalize_subject_name(extracted_name or text)
            
            if _is_false_subject(subject_name):
                continue
            if not _is_syllabus_subject_candidate(subject_name, has_course_code=True):
                continue
                
            subject_slug = _slugify(subject_name)
            
            if subject_slug in seen_slugs:
                if not any(m.course_code == course_code for m in markers if m.subject_slug == subject_slug):
                    pass # Different course code for same subject name is allowed
                else:
                    continue

            markers.append(SubjectMarker(
                index=i, subject_name=subject_name,
                subject_slug=subject_slug, course_code=course_code,
            ))
            seen_slugs.add(subject_slug)
            continue

        # Lab/Practical subject without a course code
        upper_text = text.upper()
        if any(term in upper_text for term in (" LAB", " PRACTICAL", " LABORATORY")):
            if (_is_syllabus_subject_candidate(text)
                    and not _in_curriculum_table(lines, i)
                    and _has_syllabus_context(lines, i)
                    and not _is_false_subject(text)):
                subject_name = _normalize_subject_name(text)
                subject_slug = _slugify(subject_name)
                if subject_slug not in seen_slugs and len(subject_name) >= 3:
                    markers.append(SubjectMarker(
                        index=i, subject_name=subject_name,
                        subject_slug=subject_slug, course_code=None,
                    ))
                    seen_slugs.add(subject_slug)

    return sorted(markers, key=lambda m: m.index)


def _find_markers(lines: list[LineRecord], mode: str) -> list[HeadingMarker]:
    markers: list[HeadingMarker] = []
    _GENERIC_KEYWORDS = {
        "circular", "notification", "notice", "scholarship", "fee",
        "quotation", "tender", "office order", "hostel", "placement",
        "announcement", "administrative", "workshop", "seminar",
        "internship", "admissions", "results",
    }
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
            m = re.match(r"^(\d+)\s*[.)]\s*(.+)$", text)
            if m:
                marker = "regulation_section", m.group(2).strip().upper()
            elif _is_all_caps_heading(text) and len(text.split()) <= 10:
                marker = "regulation_section", text.upper()
        else:
            if _looks_like_heading(text, previous_blank, next_blank):
                title = text.rstrip(":").upper()
                marker = "section", title
        if marker is None:
            continue
        chunk_type, section_title = marker
        if mode == "generic" and chunk_type == "section":
            if not any(kw in section_title.lower() for kw in _GENERIC_KEYWORDS):
                if not _looks_like_heading(text, previous_blank, next_blank):
                    continue
        markers.append(HeadingMarker(index=index, chunk_type=chunk_type, section_title=section_title))
    return markers


# ===========================================================================
# PHASE 6 – CHUNKERS
# ===========================================================================

class _BaseChunker:
    doc_type: str = ""

    def chunk(
        self,
        pages: list[PdfPage],
        source_pdf_name: str,
        document_type: str,
        base_name: str,
        chunk_meta: dict[str, Any],
    ) -> list[Chunk]:
        raise NotImplementedError


def _enrich_metadata(
    base: dict[str, Any],
    chunk_meta: dict[str, Any],
    parsing_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Phase 7 / Item 8: Inject classification metadata and parsing metadata into
    every chunk. parsing_meta carries boundary_source, course_code_valid,
    header_removed, parsing_confidence.
    """
    merged = {**base, **chunk_meta}
    if parsing_meta:
        merged.update(parsing_meta)
    return merged


def _compute_parsing_confidence(
    boundary_source: str,
    course_code_valid: bool,
    content: str,
) -> float:
    """
    Item 8: Heuristic parsing confidence score [0.0–1.0].
    - Starts at 1.0 and applies small deductions for uncertain signals.
    """
    score = 1.0
    if boundary_source == "fixed_size":
        score -= 0.30
    elif boundary_source == "heading":
        score -= 0.05
    elif boundary_source == "subject_heading":
        pass  # best case
    if not course_code_valid:
        score -= 0.05
    word_count = _word_count(content)
    if word_count < 20:
        score -= 0.10
    elif word_count > 2000:
        score -= 0.05
    return round(max(0.0, min(1.0, score)), 2)


# Signals that terminate a references section (Item 5)
_REFERENCE_BOUNDARY_SIGNALS = (
    _TEACHING_SCHEME_PAT,
    _EXAM_SCHEME_PAT,
    _UNIT_ONE_PAT,
    _COURSE_CODE_PATTERN,
)


def _references_boundary_hit(lines: list[LineRecord], index: int) -> bool:
    """Item 5: True when the line is a natural boundary that ends a references section."""
    text = lines[index].text
    for pat in _REFERENCE_BOUNDARY_SIGNALS:
        if pat.search(text):
            if pat is _COURSE_CODE_PATTERN:
                # Only trigger boundary for *valid* course codes
                match = pat.search(text)
                if match and _is_valid_course_code(match.group(1)):
                    return True
            else:
                return True
    return False


# Lab section markers for LabManualChunker (Item 6)
_LAB_SECTION_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^(?:Experiment|Expt\.?)\s+\d+", re.I),         "experiment",    "Experiment"),
    (re.compile(r"^(?:Practical|Exercise)\s+\d+", re.I),          "practical",     "Practical"),
    (re.compile(r"^(?:Week|Program)\s+\d+", re.I),                "lab_week",      "Week"),
    (re.compile(r"^Lab(?:oratory)?\s+(?:Overview|Introduction)", re.I), "lab_overview", "Lab Overview"),
    (re.compile(r"^(?:Evaluation|Assessment|Viva)\b", re.I),      "evaluation",    "Evaluation"),
    (re.compile(r"^(?:SUGGESTED\s+)?REFERENCES?[:]?$", re.I),     "references",    "References"),
]


class SyllabusChunker(_BaseChunker):
    doc_type = DOC_TYPE_SYLLABUS

    def chunk(self, pages, source_pdf_name, document_type, base_name, chunk_meta):
        dynamic_banners = _build_dynamic_banners(pages)
        lines = _build_line_records(pages)
        markers = _find_markers(lines, mode="syllabus")
        subject_markers = _find_syllabus_subject_markers(lines, markers)
        chunks = self._make_chunks(
            lines, markers, subject_markers,
            base_name, source_pdf_name, document_type,
            chunk_meta, dynamic_banners,
        )
        if not chunks:
            chunks = _fixed_size_chunks(pages, base_name, source_pdf_name, document_type, chunk_meta)
            
        chunks = self._validate_and_split_subjects(chunks)
        return chunks

    def _validate_and_split_subjects(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Validation: Group chunks by subject_name. Verify course_code remains constant.
        If another course code appears, split automatically to prevent metadata leakage.
        """
        validated_chunks: list[Chunk] = []
        for chunk in chunks:
            subject_name = chunk.metadata.get("subject_name")
            course_code = chunk.metadata.get("course_code")
            if subject_name and course_code:
                # If course code changes for the same subject, make the chunk ID unique
                safe_code = _slugify(course_code)
                if safe_code and safe_code not in chunk.chunk_id:
                    # Inject course code into ID to split the group
                    parts = chunk.chunk_id.split("_", 1)
                    if len(parts) == 2:
                        chunk.chunk_id = f"{parts[0]}_{safe_code}_{parts[1]}"
            validated_chunks.append(chunk)
        return validated_chunks

    def _make_chunks(
        self, lines, markers, subject_markers,
        base_name, source_pdf_name, document_type,
        chunk_meta, dynamic_banners,
    ):
        all_markers = sorted(
            [(m, "subject") for m in subject_markers] + [(m, "heading") for m in markers],
            key=lambda x: (x[0].index, 0 if x[1] == "subject" else 1),
        )
        spans: list[Any] = []
        if all_markers and all_markers[0][0].index > 0:
            spans.append((0, all_markers[0][0].index, None))
        for i, (marker, mtype) in enumerate(all_markers):
            end = all_markers[i + 1][0].index if i + 1 < len(all_markers) else len(lines)
            spans.append((marker.index, end, (marker, mtype)))

        chunks: list[Chunk] = []
        active_subject: Optional[SubjectMarker] = None
        # Item 1: track whether we've seen the first real UNIT section
        first_unit_seen = any(m.chunk_type == "unit" for m in markers)
        in_curriculum_table_phase = not first_unit_seen
        curriculum_overview_lines: list[LineRecord] = []
        counters: dict[tuple[str, str], int] = {}
        used_ids: set[str] = set()

        for start, end, marker_info in spans:
            # Item 1: accumulate pre-UNIT subject_overview lines into one curriculum_overview
            if in_curriculum_table_phase:
                if marker_info and marker_info[1] == "subject":
                    # still in table phase — collect but don't emit subject chunk
                    curriculum_overview_lines.extend(
                        ln for ln in lines[start:end] if not _is_blank(ln.text)
                    )
                    continue
                elif marker_info and marker_info[1] == "heading" and marker_info[0].chunk_type == "unit":
                    # First UNIT seen — emit accumulated curriculum_overview then proceed
                    in_curriculum_table_phase = False
                    if curriculum_overview_lines:
                        content, hdr_removed = _join_lines(curriculum_overview_lines, dynamic_banners)
                        if content:
                            pg_s, pg_e = _page_range_from_lines(curriculum_overview_lines)
                            parsing_meta = {
                                "parsing_confidence": _compute_parsing_confidence("subject_heading", False, content),
                                "boundary_source": "curriculum_table",
                                "course_code_valid": False,
                                "header_removed": hdr_removed,
                            }
                            chunks.append(Chunk(
                                chunk_id=_make_chunk_id(base_name, "curriculum", used_ids),
                                chunk_type="curriculum_overview",
                                section_title="Curriculum Overview",
                                page_start=pg_s, page_end=pg_e,
                                content=content,
                                metadata=_enrich_metadata(
                                    {"source_pdf": source_pdf_name, "document_type": document_type,
                                     "chunk_strategy": "syllabus_sections",
                                     "word_count": _word_count(content)},
                                    chunk_meta, parsing_meta,
                                ),
                            ))
                        curriculum_overview_lines = []
                elif marker_info is None:
                    # preamble before first marker — collect
                    curriculum_overview_lines.extend(
                        ln for ln in lines[start:end] if not _is_blank(ln.text)
                    )
                    continue

            if marker_info and marker_info[1] == "subject":
                new_subject = marker_info[0]
                if active_subject is None:
                    print(f"Subject detected\n{new_subject.subject_name}\nCourse Code\n{new_subject.course_code}\nPage\n{lines[new_subject.index].page_number}\nReason\nCourse heading")
                elif active_subject.subject_slug != new_subject.subject_slug or active_subject.course_code != new_subject.course_code:
                    print(f"Subject switched\n{new_subject.subject_name}\nCourse Code\n{new_subject.course_code}\nPage\n{lines[new_subject.index].page_number}\nReason\nNew heading detected")
                active_subject = new_subject

            span_lines = [ln for ln in lines[start:end] if not _is_blank(ln.text)]

            # Item 5: for references chunks, enforce boundary at next subject/unit
            if (marker_info and marker_info[1] == "heading"
                    and marker_info[0].chunk_type == "references"):
                trimmed = []
                for ln in span_lines:
                    if _references_boundary_hit(lines, lines.index(ln) if ln in lines else -1):
                        # Stop at the boundary signal line (exclusive)
                        break
                    trimmed.append(ln)
                span_lines = trimmed if trimmed else span_lines

            if not span_lines:
                continue

            prefix = active_subject.subject_slug if active_subject else base_name
            course_code_valid = bool(
                active_subject and active_subject.course_code
                and _is_valid_course_code(active_subject.course_code)
            )
            content, hdr_removed = _join_lines(span_lines, dynamic_banners)

            base_meta: dict[str, Any] = {
                "source_pdf": source_pdf_name,
                "document_type": document_type,
                "chunk_strategy": "syllabus_sections",
                "word_count": _word_count(content),
            }
            if active_subject:
                base_meta["subject_name"] = active_subject.subject_name
                if active_subject.course_code:
                    base_meta["course_code"] = active_subject.course_code

            if marker_info is None:
                chunk_type = "curriculum_overview"
                section_title = "Curriculum Overview"
                chunk_id_base = "curriculum"
                boundary_source = "preamble"
            elif marker_info[1] == "subject":
                chunk_type = "subject_overview"
                section_title = active_subject.subject_name
                chunk_id_base = "overview"
                boundary_source = "subject_heading"
            else:
                m = marker_info[0]
                chunk_type = m.chunk_type
                section_title = m.section_title
                counter_key = (prefix, chunk_type)
                counters[counter_key] = counters.get(counter_key, 0) + 1
                idx = counters[counter_key]
                chunk_id_base = (
                    f"unit_{idx}" if chunk_type == "unit"
                    else chunk_type if chunk_type in (
                        "textbooks", "references", "course_objectives",
                        "course_outcomes", "lab_exercises",
                    )
                    else f"section_{idx}"
                )
                boundary_source = "section_heading"

            # Item 8: parsing metadata
            parsing_meta = {
                "parsing_confidence": _compute_parsing_confidence(boundary_source, course_code_valid, content),
                "boundary_source": boundary_source,
                "course_code_valid": course_code_valid,
                "header_removed": hdr_removed,
            }

            chunk_id = _make_chunk_id(prefix, chunk_id_base, used_ids)
            page_start, page_end = _page_range_from_lines(span_lines)
            chunks.append(Chunk(
                chunk_id=chunk_id, chunk_type=chunk_type,
                section_title=section_title, page_start=page_start,
                page_end=page_end, content=content,
                metadata=_enrich_metadata(base_meta, chunk_meta, parsing_meta),
            ))
        return chunks


class LabManualChunker(_BaseChunker):
    """
    Item 6: Lab manual chunker.
    Splits by Experiment / Practical / Exercise / Week / Program headings into
    logical sections: lab_overview, experiment, evaluation, references.
    Falls back to generic heading chunker if no lab section markers are found.
    """
    doc_type = DOC_TYPE_LAB_MANUAL

    def chunk(self, pages, source_pdf_name, document_type, base_name, chunk_meta):
        dynamic_banners = _build_dynamic_banners(pages)
        lines = _build_line_records(pages)
        markers = self._find_lab_markers(lines)
        if not markers:
            # Fallback: generic heading split
            return _generic_heading_chunks(
                pages, source_pdf_name, document_type, base_name,
                strategy="lab_sections", chunk_meta=chunk_meta,
            )
        spans = _section_spans(lines, markers)
        used_ids: set[str] = set()
        chunks: list[Chunk] = []
        for start, end, marker in spans:
            span_lines = [ln for ln in lines[start:end] if not _is_blank(ln.text)]
            if not span_lines:
                continue
            content, hdr_removed = _join_lines(span_lines, dynamic_banners)
            page_start, page_end = _page_range_from_lines(span_lines)
            chunk_id = _make_chunk_id(base_name, marker.section_title, used_ids)
            parsing_meta = {
                "parsing_confidence": _compute_parsing_confidence("section_heading", False, content),
                "boundary_source": "lab_section_heading",
                "course_code_valid": False,
                "header_removed": hdr_removed,
            }
            chunks.append(Chunk(
                chunk_id=chunk_id, chunk_type=marker.chunk_type,
                section_title=marker.section_title,
                page_start=page_start, page_end=page_end,
                content=content,
                metadata=_enrich_metadata(
                    {"source_pdf": source_pdf_name, "document_type": document_type,
                     "chunk_strategy": "lab_sections", "word_count": _word_count(content)},
                    chunk_meta, parsing_meta,
                ),
            ))
        return chunks

    @staticmethod
    def _find_lab_markers(lines: list[LineRecord]) -> list[HeadingMarker]:
        markers: list[HeadingMarker] = []
        for index, line in enumerate(lines):
            text = _normalize_whitespace(line.text)
            if not text:
                continue
            for pat, chunk_type, label in _LAB_SECTION_PATTERNS:
                if pat.match(text):
                    markers.append(HeadingMarker(
                        index=index,
                        chunk_type=chunk_type,
                        section_title=text,
                    ))
                    break
        return markers


class ExamScheduleChunker(_BaseChunker):
    doc_type = DOC_TYPE_EXAM_SCHEDULE

    def chunk(self, pages, source_pdf_name, document_type, base_name, chunk_meta):
        dynamic_banners = _build_dynamic_banners(pages)
        content_raw = _merged_content(pages)
        if not content_raw:
            return []
        # Strip dynamic banners from merged content
        lines = _build_line_records(pages)
        content, hdr_removed = _join_lines(lines, dynamic_banners)
        if not content:
            content = content_raw
            hdr_removed = False
        page_start = min((p.page_number for p in pages), default=0)
        page_end = max((p.page_number for p in pages), default=0)
        section_title = "TIME TABLE" if re.search(r"\btime\s*table\b|\btimetable\b", content, re.I) else "EXAM SCHEDULE"
        parsing_meta = {
            "parsing_confidence": _compute_parsing_confidence("section_heading", False, content),
            "boundary_source": "single_document",
            "course_code_valid": False,
            "header_removed": hdr_removed,
        }
        meta = _enrich_metadata({
            "source_pdf": source_pdf_name,
            "document_type": document_type,
            "chunk_strategy": "single_timetable_chunk",
            "word_count": _word_count(content),
        }, chunk_meta, parsing_meta)
        return [Chunk(
            chunk_id=f"{base_name}_schedule", chunk_type="exam_schedule",
            section_title=section_title, page_start=page_start,
            page_end=page_end, content=content, metadata=meta,
        )]


class RegulationChunker(_BaseChunker):
    doc_type = DOC_TYPE_REGULATIONS

    def chunk(self, pages, source_pdf_name, document_type, base_name, chunk_meta):
        dynamic_banners = _build_dynamic_banners(pages)
        lines = _build_line_records(pages)
        markers = _find_markers(lines, mode="regulations")
        chunks = self._make_chunks(lines, markers, base_name, source_pdf_name, document_type, chunk_meta, dynamic_banners)
        if not chunks:
            chunks = _fixed_size_chunks(pages, base_name, source_pdf_name, document_type, chunk_meta)
        return chunks

    def _make_chunks(self, lines, markers, base_name, source_pdf_name, document_type, chunk_meta, dynamic_banners=frozenset()):
        spans = _section_spans(lines, markers)
        if not spans:
            return []
        used_ids: set[str] = set()
        chunks: list[Chunk] = []
        for start, end, marker in spans:
            span_lines = [ln for ln in lines[start:end] if not _is_blank(ln.text)]
            if not span_lines:
                continue
            content, hdr_removed = _join_lines(span_lines, dynamic_banners)
            page_start, page_end = _page_range_from_lines(span_lines)
            chunk_id = _make_chunk_id(base_name, marker.section_title, used_ids)
            parsing_meta = {
                "parsing_confidence": _compute_parsing_confidence("section_heading", False, content),
                "boundary_source": "section_heading",
                "course_code_valid": False,
                "header_removed": hdr_removed,
            }
            meta = _enrich_metadata({
                "source_pdf": source_pdf_name,
                "document_type": document_type,
                "chunk_strategy": "regulation_sections",
                "word_count": _word_count(content),
            }, chunk_meta, parsing_meta)
            chunks.append(Chunk(
                chunk_id=chunk_id, chunk_type=marker.chunk_type,
                section_title=marker.section_title, page_start=page_start,
                page_end=page_end, content=content, metadata=meta,
            ))
        return chunks


class NotificationChunker(_BaseChunker):
    doc_type = DOC_TYPE_NOTIFICATION

    def chunk(self, pages, source_pdf_name, document_type, base_name, chunk_meta):
        return _generic_heading_chunks(pages, source_pdf_name, document_type, base_name,
                                       strategy="notification_sections", chunk_meta=chunk_meta)


class GenericHeadingChunker(_BaseChunker):
    doc_type = DOC_TYPE_UNKNOWN

    def chunk(self, pages, source_pdf_name, document_type, base_name, chunk_meta):
        return _generic_heading_chunks(pages, source_pdf_name, document_type, base_name,
                                       strategy="heading_sections", chunk_meta=chunk_meta)


def _generic_heading_chunks(
    pages: list[PdfPage],
    source_pdf_name: str,
    document_type: str,
    base_name: str,
    strategy: str,
    chunk_meta: dict[str, Any],
) -> list[Chunk]:
    dynamic_banners = _build_dynamic_banners(pages)
    lines = _build_line_records(pages)
    markers = _find_markers(lines, mode="generic")
    if not markers:
        return _fixed_size_chunks(pages, base_name, source_pdf_name, document_type, chunk_meta)
    spans = _section_spans(lines, markers)
    used_ids: set[str] = set()
    chunks: list[Chunk] = []
    for start, end, marker in spans:
        span_lines = [ln for ln in lines[start:end] if not _is_blank(ln.text)]
        if not span_lines:
            continue
        content, hdr_removed = _join_lines(span_lines, dynamic_banners)
        page_start, page_end = _page_range_from_lines(span_lines)
        chunk_id = _make_chunk_id(base_name, marker.section_title, used_ids)
        parsing_meta = {
            "parsing_confidence": _compute_parsing_confidence("heading", False, content),
            "boundary_source": "section_heading",
            "course_code_valid": False,
            "header_removed": hdr_removed,
        }
        meta = _enrich_metadata({
            "source_pdf": source_pdf_name,
            "document_type": document_type,
            "chunk_strategy": strategy,
            "word_count": _word_count(content),
        }, chunk_meta, parsing_meta)
        chunks.append(Chunk(
            chunk_id=chunk_id, chunk_type=marker.chunk_type,
            section_title=marker.section_title, page_start=page_start,
            page_end=page_end, content=content, metadata=meta,
        ))
    return chunks or _fixed_size_chunks(pages, base_name, source_pdf_name, document_type, chunk_meta)


class QuotationChunker(_BaseChunker):
    doc_type = DOC_TYPE_QUOTATION

    def chunk(self, pages, source_pdf_name, document_type, base_name, chunk_meta):
        return _generic_heading_chunks(pages, source_pdf_name, document_type, base_name,
                                       strategy="quotation_sections", chunk_meta=chunk_meta)


def _fixed_size_chunks(
    pages: list[PdfPage],
    base_name: str,
    source_pdf_name: str,
    document_type: str,
    chunk_meta: dict[str, Any],
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    words: list[tuple[str, int]] = []
    for page in pages:
        for line in page.content.splitlines():
            for word in re.findall(r"\S+", line):
                words.append((word, page.page_number))
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    start = 0
    index = 1
    while start < len(words):
        window = words[start: start + chunk_size]
        if not window:
            break
        content = _normalize_whitespace(" ".join(w for w, _ in window))
        page_start = min(pg for _, pg in window)
        page_end = max(pg for _, pg in window)
        parsing_meta = {
            "parsing_confidence": _compute_parsing_confidence("fixed_size", False, content),
            "boundary_source": "fixed_size",
            "course_code_valid": False,
            "header_removed": False,
        }
        meta = _enrich_metadata({
            "source_pdf": source_pdf_name,
            "document_type": document_type,
            "chunk_strategy": "fixed_size",
            "word_count": len(window),
            "chunk_words": chunk_size,
            "chunk_overlap": overlap,
        }, chunk_meta, parsing_meta)
        chunks.append(Chunk(
            chunk_id=f"{base_name}_chunk_{index}",
            chunk_type="chunk",
            section_title=f"Chunk {index}",
            page_start=page_start,
            page_end=page_end,
            content=content,
            metadata=meta,
        ))
        if start + chunk_size >= len(words):
            break
        start += step
        index += 1
    return chunks


class ChunkerStrategy:
    """
    Phase 6 dispatcher — uses REGISTRY, NEVER classifies.
    Receives FinalClassification as the sole source of document_type.
    Falls back to GenericHeadingChunker for unrecognised types.
    """
    _FALLBACK = GenericHeadingChunker()

    def chunk(
        self,
        classification: FinalClassification,
        pages: list[PdfPage],
        source_pdf_name: str,
        base_name: str,
        chunk_meta: dict[str, Any],
    ) -> list[Chunk]:
        doc_type = classification.document_type
        chunker = REGISTRY.get_chunker(doc_type) or self._FALLBACK
        return chunker.chunk(pages, source_pdf_name, doc_type, base_name, chunk_meta)


# ===========================================================================
# REGISTRY POPULATION
# (One DocumentTypeProfile per type — single registration point)
# ===========================================================================

def _build_registry() -> None:
    """Populate the global REGISTRY with all known document type profiles."""
    _generic_chunker      = GenericHeadingChunker()
    _notif_chunker        = NotificationChunker()
    _quotation_chunker    = QuotationChunker()
    _syllabus_chunker     = SyllabusChunker()
    _exam_chunker         = ExamScheduleChunker()
    _reg_chunker          = RegulationChunker()

    profiles: list[tuple[str, str, _BaseScorer, _BaseValidator, _BaseChunker, str]] = [
        # (doc_type, category, scorer, validator, chunker, description)
        (DOC_TYPE_SYLLABUS,        "academic",       SyllabusScorer(),      SyllabusValidator(),      _syllabus_chunker,  "Course syllabus with units and textbooks"),
        (DOC_TYPE_LAB_MANUAL,      "academic",       LabManualScorer(),     LabManualValidator(),     LabManualChunker(), "Laboratory manual with experiments"),
        (DOC_TYPE_ACADEMIC_CAL,    "academic",       AcademicCalendarScorer(), AcademicCalendarValidator(), _generic_chunker, "Academic calendar with dates"),
        (DOC_TYPE_ALMANAC,         "academic",       AlmanacScorer(),       AlmanacValidator(),       _generic_chunker,   "University almanac with semester schedule"),
        (DOC_TYPE_EXAM_SCHEDULE,   "academic",       ExamScheduleScorer(),  ExamScheduleValidator(),  _exam_chunker,      "Examination schedule/timetable"),
        (DOC_TYPE_CLASS_TIMETABLE, "academic",       ClassTimetableScorer(),ClassTimetableValidator(),_exam_chunker,      "Class timetable with periods and days"),
        (DOC_TYPE_REGULATIONS,     "academic",       RegulationsScorer(),   RegulationsValidator(),   _reg_chunker,       "Academic regulations"),
        (DOC_TYPE_NOTIFICATION,    "administration", NotificationScorer(),  NotificationValidator(),  _notif_chunker,     "Official notification"),
        (DOC_TYPE_CIRCULAR,        "administration", CircularScorer(),      CircularValidator(),      _notif_chunker,     "Administrative circular"),
        (DOC_TYPE_OFFICE_ORDER,    "administration", OfficeOrderScorer(),   PassthroughValidator(DOC_TYPE_OFFICE_ORDER), _notif_chunker, "Office order"),
        (DOC_TYPE_PROCEEDINGS,     "administration", ProceedingsScorer(),   ProceedingsValidator(),   _notif_chunker,     "Official proceedings"),
        (DOC_TYPE_MINUTES,         "administration", MinutesScorer(),       MinutesValidator(),       _notif_chunker,     "Minutes of meeting"),
        (DOC_TYPE_MEMORANDUM,      "administration", MemorandumScorer(),    PassthroughValidator(DOC_TYPE_MEMORANDUM),  _notif_chunker, "Memorandum"),
        (DOC_TYPE_REPORT,          "administration", ReportScorer(),        ReportValidator(),        _generic_chunker,   "Formal report"),
        (DOC_TYPE_FEE_NOTICE,      "finance",        FeeNoticeScorer(),     FeeNoticeValidator(),     _notif_chunker,     "Fee payment notice"),
        (DOC_TYPE_SCHOLARSHIP,     "finance",        ScholarshipScorer(),   ScholarshipValidator(),   _notif_chunker,     "Scholarship notice"),
        (DOC_TYPE_REIMBURSEMENT,   "finance",        ReimbursementScorer(), PassthroughValidator(DOC_TYPE_REIMBURSEMENT), _notif_chunker, "Reimbursement notice"),
        (DOC_TYPE_TENDER,          "finance",        TenderScorer(),        PassthroughValidator(DOC_TYPE_TENDER), _quotation_chunker, "Tender document"),
        (DOC_TYPE_QUOTATION,       "finance",        QuotationScorer(),     QuotationValidator(),     _quotation_chunker, "Call for quotation"),
        (DOC_TYPE_WORKSHOP,        "event",          WorkshopScorer(),      WorkshopValidator(),      _notif_chunker,     "Workshop announcement/brochure"),
        (DOC_TYPE_SEMINAR,         "event",          SeminarScorer(),       PassthroughValidator(DOC_TYPE_SEMINAR), _notif_chunker, "Seminar notice"),
        (DOC_TYPE_CONFERENCE,      "event",          ConferenceScorer(),    PassthroughValidator(DOC_TYPE_CONFERENCE), _notif_chunker, "Conference notice"),
        (DOC_TYPE_FDP,             "event",          FDPScorer(),           PassthroughValidator(DOC_TYPE_FDP), _notif_chunker, "Faculty Development Programme"),
        (DOC_TYPE_PLACEMENT,       "event",          PlacementScorer(),     PassthroughValidator(DOC_TYPE_PLACEMENT), _notif_chunker, "Placement drive notice"),
        (DOC_TYPE_ADMISSIONS,      "student",        AdmissionsScorer(),    PassthroughValidator(DOC_TYPE_ADMISSIONS), _notif_chunker, "Admissions notification"),
        (DOC_TYPE_HOSTEL,          "student",        HostelScorer(),        PassthroughValidator(DOC_TYPE_HOSTEL), _notif_chunker, "Hostel notice"),
        (DOC_TYPE_EXAM_RESULTS,    "student",        ExamResultsScorer(),   PassthroughValidator(DOC_TYPE_EXAM_RESULTS), _notif_chunker, "Examination results"),
        (DOC_TYPE_INTERNSHIP,      "student",        InternshipScorer(),    PassthroughValidator(DOC_TYPE_INTERNSHIP), _notif_chunker, "Internship notice"),
    ]

    for (doc_type, category, scorer, validator, chunker, desc) in profiles:
        REGISTRY.register(DocumentTypeProfile(
            doc_type=doc_type,
            category=category,
            scorer=scorer,
            validator=validator,
            chunker=chunker,
            description=desc,
        ))


_build_registry()


# ===========================================================================
# LOGGING – Pipeline Stage Logs
# ===========================================================================

def _log_feature_map(fm: FeatureMap) -> None:
    log = logging.getLogger(__name__)
    log.info("=" * 40 + " Feature Map " + "=" * 40)
    log.info("  page_count              = %d", fm.page_count)
    log.info("  word_count              = %d", fm.word_count)
    log.info("  course_code_count       = %d", fm.course_code_count)
    log.info("  unit_count              = %d", fm.unit_count)
    log.info("  textbook_sections       = %d", fm.textbook_section_count)
    log.info("  reference_sections      = %d", fm.reference_section_count)
    log.info("  course_objectives       = %s", fm.course_objectives_present)
    log.info("  course_outcomes         = %s", fm.course_outcomes_present)
    log.info("  teaching_scheme         = %s", fm.teaching_scheme_present)
    log.info("  examination_scheme      = %s", fm.examination_scheme_present)
    log.info("  ltpc_table              = %s", fm.ltpc_table_present)
    log.info("  date_count              = %d", fm.date_count)
    log.info("  time_count              = %d", fm.time_count)
    log.info("  exam_time_patterns      = %d", fm.exam_time_patterns)
    log.info("  session_labels          = %s", fm.session_labels)
    log.info("  period_labels           = %s", fm.period_labels)
    log.info("  holiday_patterns        = %d", fm.holiday_patterns)
    log.info("  instruction_days        = %s", fm.instruction_days_present)
    log.info("  odd_even_semester       = %s", fm.odd_even_semester_present)
    log.info("  quotation_present       = %s", fm.quotation_present)
    log.info("  tender_present          = %s", fm.tender_present)
    log.info("  fee_present             = %s", fm.fee_present)
    log.info("  scholarship_present     = %s", fm.scholarship_present)
    log.info("  reimbursement_present   = %s", fm.reimbursement_present)
    log.info("  workshop_present        = %s", fm.workshop_present)
    log.info("  seminar_present         = %s", fm.seminar_present)
    log.info("  conference_present      = %s", fm.conference_present)
    log.info("  fdp_present             = %s", fm.fdp_present)
    log.info("  placement_present       = %s", fm.placement_present)
    log.info("  admissions_present      = %s", fm.admissions_present)
    log.info("  hostel_present          = %s", fm.hostel_present)
    log.info("  chapter_count           = %d", fm.chapter_count)
    log.info("  rule_section_count      = %d", fm.rule_section_count)
    log.info("  clause_count            = %d", fm.clause_count)
    log.info("  report_sections         = %s", fm.report_sections)
    log.info("  signatures_present      = %s", fm.signatures_present)
    log.info("  annexure_count          = %d", fm.annexure_count)
    log.info("  amount_count            = %d", fm.amount_count)
    log.info("  venue                   = %s", fm.venue)
    log.info("  academic_year           = %s", fm.academic_year)
    log.info("=" * 93)


def _log_candidate_scores(ranked: list[CandidateScore]) -> None:
    log = logging.getLogger(__name__)
    log.info("=" * 40 + " Candidate Scores " + "=" * 35)
    for cs in ranked[:8]:   # show top 8 to avoid noise
        log.info("  %-25s = %3d  [S:%d W:%d]  %s",
                 cs.doc_type, cs.score,
                 cs.strong_evidence_count, cs.weak_evidence_count,
                 ", ".join(cs.evidence[:4]))
    log.info("=" * 93)


def _log_confidence(cr: ConfidenceResult) -> None:
    log = logging.getLogger(__name__)
    log.info("=" * 40 + " Confidence " + "=" * 41)
    log.info("  Winner          : %-25s score=%d", cr.winner, cr.winner_score)
    log.info("  Runner-up       : %-25s score=%d", cr.runner_up, cr.runner_up_score)
    log.info("  Gap             : %d", cr.gap)
    log.info("  Level           : %s", cr.level)
    log.info("  Strong Evidence : %d", cr.strong_evidence_count)
    log.info("  Weak Evidence   : %d", cr.weak_evidence_count)
    log.info("  Completeness    : %.2f", cr.structural_completeness)
    if cr.contradictions:
        log.info("  Contradictions  : %s", cr.contradictions)
    log.info("=" * 93)


def _log_validation(vr: ValidationResult) -> None:
    log = logging.getLogger(__name__)
    log.info("=" * 40 + " Validator " + "=" * 42)
    log.info("  Passed     : %s", vr.passed)
    log.info("  Final Type : %s", vr.final_type)
    if vr.reason:
        log.info("  Reason     : %s", vr.reason)
    log.info("=" * 93)


# ===========================================================================
# PHASE 7 – CHUNK METADATA BUILDER
# ===========================================================================

def _build_chunk_meta(
    classification: FinalClassification,
    fm: FeatureMap,
) -> dict[str, Any]:
    """
    Returns metadata dict injected into every chunk.
    Reads exclusively from FinalClassification — no separate confidence/validation
    objects needed, eliminating any risk of stale document_type values.
    """
    feature_summary: dict[str, Any] = {}
    if fm.course_code_count:
        feature_summary["course_code_count"] = fm.course_code_count
    if fm.unit_count:
        feature_summary["unit_count"] = fm.unit_count
    if fm.textbook_section_count:
        feature_summary["textbook_sections"] = fm.textbook_section_count
    if fm.date_count:
        feature_summary["date_count"] = fm.date_count
    if fm.exam_time_patterns:
        feature_summary["exam_time_patterns"] = fm.exam_time_patterns
    if fm.amount_count:
        feature_summary["amount_count"] = fm.amount_count
    if fm.scholarship_present:
        feature_summary["scholarship"] = True
    if fm.workshop_present:
        feature_summary["workshop"] = True
    if fm.academic_year:
        feature_summary["academic_year"] = fm.academic_year

    return {
        "classification_confidence": classification.confidence,
        "classifier_score":          classification.score,
        "runner_up_score":           classification.runner_up_score,
        "confidence_gap":            classification.gap,
        "strong_evidence_count":     classification.strong_evidence_count,
        "validator_result":          classification.validator_result,
        "validator_reason":          classification.validator_reason,
        "feature_summary":           feature_summary,
    }


# ===========================================================================
# PHASE 8 – CLASSIFICATION REPORT
# ===========================================================================

def _save_classification_report(
    report_entries: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Write classification_report.json summarising all processed PDFs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "total": len(report_entries),
                "documents": report_entries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    return report_path


def _make_report_entry(
    source_json: Path,
    chunked: ChunkedDocument,
    classification: FinalClassification,
    confidence: ConfidenceResult,
    fm: FeatureMap,
) -> dict[str, Any]:
    """
    Build a classification report entry.
    document_type is always read from FinalClassification, not from chunked or
    any earlier variable, guaranteeing the report reflects the validated result.
    """
    feature_summary: dict[str, Any] = {
        "course_code_count": fm.course_code_count,
        "unit_count": fm.unit_count,
        "date_count": fm.date_count,
        "exam_time_patterns": fm.exam_time_patterns,
        "scholarship": fm.scholarship_present,
        "workshop": fm.workshop_present,
        "academic_year": fm.academic_year,
    }
    return {
        "file_name":        source_json.name,
        "document_type":    classification.document_type,
        "document_subtype": classification.document_subtype,
        "confidence":       classification.confidence,
        "winner_score":     classification.score,
        "runner_up":        classification.runner_up,
        "runner_up_score":  classification.runner_up_score,
        "gap":              classification.gap,
        "strong_evidence":  classification.strong_evidence_count,
        "contradictions":   confidence.contradictions,
        "validator_result": classification.validator_result,
        "validator_reason": classification.validator_reason,
        "chunk_count":      chunked.chunk_count,
        "feature_summary":  feature_summary,
    }


# ===========================================================================
# TOP-LEVEL PIPELINE
# ===========================================================================

def chunk_document(
    data: dict[str, Any],
    source_json: Path,
) -> tuple[ChunkedDocument, FinalClassification, ConfidenceResult, FeatureMap]:
    """
    Full pipeline: Feature → Score → Confidence → Validate → FinalClassification → Chunk.

    FinalClassification is built once after Phase 4 and is the sole source of
    document_type for every downstream step: chunker selection, output JSON,
    chunk metadata, classification report, and logging.

    data["document_type"] (parser metadata) is intentionally ignored after this
    point to prevent stale values from overriding the validated result.
    """
    log = logging.getLogger(__name__)

    pages = _load_pages(data)
    source_pdf = _source_pdf_name(data, source_json.stem)
    source_pdf_name = Path(source_pdf).name
    base_name = Path(source_pdf).stem.lower().replace(" ", "_") or source_json.stem.lower()

    # Phase 1 — Feature Extraction
    extractor = FeatureExtractor()
    fm = extractor.extract(pages)
    _log_feature_map(fm)

    # Phase 2 — Candidate Scoring
    scorer = CandidateScorer()
    ranked = scorer.score_all(fm)
    _log_candidate_scores(ranked)

    # Phase 3 — Confidence Analysis
    analyzer = ConfidenceAnalyzer()
    confidence = analyzer.analyze(ranked)
    _log_confidence(confidence)

    # Phase 4 — Document Validation
    validator = DocumentValidator()
    validation = validator.validate(confidence, fm, pages)
    _log_validation(validation)

    # Phase 5 — FinalClassification (single source of truth from here on)
    classification = build_final_classification(confidence, validation)
    log.info("=" * 40 + " Final Classification " + "=" * 32)
    log.info("  document_type    = %s", classification.document_type.upper())
    log.info("  document_subtype = %s", classification.document_subtype)
    log.info("  confidence       = %s  (score=%d  gap=%d)",
             classification.confidence, classification.score, classification.gap)
    log.info("  validator        = %s — %s",
             classification.validator_result, classification.validator_reason)
    log.info("=" * 93)

    # Phase 7 — Chunk metadata (reads from FinalClassification only)
    chunk_meta = _build_chunk_meta(classification, fm)

    # Phase 6 — Chunker Strategy (receives FinalClassification, never classifies)
    strategy = ChunkerStrategy()
    chunks = strategy.chunk(classification, pages, source_pdf_name, base_name, chunk_meta)

    if not chunks:
        log.warning("No chunks produced; falling back to fixed_size: %s", source_json.name)
        chunks = _fixed_size_chunks(
            pages, base_name, source_pdf_name,
            classification.document_type, chunk_meta,
        )
    if not pages:
        log.warning("No pages found: %s", source_json.name)
    if not chunks:
        log.warning("No chunks generated: %s", source_json.name)

    chunked = ChunkedDocument(
        source_pdf=source_pdf_name,
        document_type=classification.document_type,
        document_subtype=classification.document_subtype,
        chunk_count=len(chunks),
        chunks=chunks,
    )
    return chunked, classification, confidence, fm


# ===========================================================================
# CLI INTERFACE
# ===========================================================================

def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def process_one(
    source_json: Path,
    out_dir: Path,
) -> Optional[dict[str, Any]]:
    """Process a single JSON file. Returns a report entry dict or None on failure."""
    log = logging.getLogger(__name__)
    try:
        log.info("Processing: %s", source_json.name)
        data = _load_input_document(source_json)
        # chunk_document returns FinalClassification as the second element;
        # confidence is kept only for the contradictions field in the report.
        chunked, classification, confidence, fm = chunk_document(data, source_json)
        log.info("Document Type  : %s (%s)",
                 classification.document_type, classification.document_subtype)
        log.info("Chunks Created : %d", chunked.chunk_count)
        if chunked.chunks:
            log.info("Chunk Types    : %s",
                     ", ".join(_dedupe_preserve_order([c.chunk_type for c in chunked.chunks])))
        if chunked.chunk_count == 0:
            log.warning("chunk_count == 0: %s", source_json.name)
        out_path = _save_chunked_document(chunked, out_dir, source_json)
        log.info("Saved JSON     : %s", out_path)
        return _make_report_entry(source_json, chunked, classification, confidence, fm)
    except Exception:
        log.exception("Failed to chunk JSON: %s", source_json)
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PDF Chunker — Production-Ready Modular 6-Phase Classification & Chunking Engine"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Chunk a single JSON file")
    group.add_argument("--all", action="store_true", help="Chunk all JSON files in input directory")
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

    processed = successful = failed = 0
    report_entries: list[dict[str, Any]] = []

    if args.file:
        processed = 1
        entry = process_one(Path(args.file), out_dir)
        if entry:
            successful = 1
            report_entries.append(entry)
        else:
            failed = 1
    else:
        files = _iter_input_files(input_dir)
        log = logging.getLogger(__name__)
        log.info("Found %d JSON files in %s", len(files), input_dir)
        for source_json in files:
            processed += 1
            entry = process_one(source_json, out_dir)
            if entry:
                successful += 1
                report_entries.append(entry)
            else:
                failed += 1

    # Phase 8 — write classification report
    if report_entries:
        report_path = _save_classification_report(report_entries, out_dir)
        logging.getLogger(__name__).info("Classification report: %s", report_path)

    print(json.dumps(
        {"jsons_processed": processed, "jsons_successful": successful, "jsons_failed": failed},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())