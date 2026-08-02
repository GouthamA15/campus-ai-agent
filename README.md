# 🎓 Interactive Campus Info AI Agent

An AI-powered Retrieval-Augmented Generation (RAG) system designed to help students, faculty, and staff quickly access information from official KUCET resources.

The system automatically crawls university websites, discovers academic documents, extracts content, processes and deeply understands document structure, and prepares a semantically rich knowledge base that powers an AI chatbot.

---

## 🚀 Project Overview

Students often struggle to find information scattered across multiple university pages and PDF documents.

Important information such as:

- Syllabus
- Examination schedules
- Circulars and notifications
- Fee notices and scholarship information
- Department and administrative notices
- Rules and regulations

is spread across various pages and documents.

This project centralizes university knowledge and provides a conversational AI interface that answers questions using official KUCET data.

---

## 🎯 Project Goal

Build a scalable university information assistant capable of answering:

- Who is the Principal of KUCET?
- What subjects are offered in B.Tech CSE Semester VI?
- What is the latest examination notification?
- What is the fee payment deadline?
- What scholarships are available?
- Where can I find a specific syllabus?

using official university resources instead of relying solely on generic LLM knowledge.

---

## 🏗 System Architecture

```text
PDF / Website
      ↓
Phase 1 — Knowledge Ingestion
      ↓
Phase 2 — PDF Intelligence Pipeline   ← COMPLETED
      ↓
Phase 3 — Embedding Generation        ← IN PROGRESS
      ↓
Phase 4 — Vector Database
      ↓
Phase 5 — Retrieval Pipeline
      ↓
Phase 6 — AI Chatbot (RAG)
```

---

## 📊 Current Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Knowledge Ingestion Pipeline | ✅ Complete |
| **Phase 2** | PDF Intelligence Pipeline | ✅ Complete |
| **Phase 3** | Embedding Generation | 🔄 In Progress |
| Phase 4 | ChromaDB Vector Storage | 🔜 Planned |
| Phase 5 | Retrieval Pipeline | 🔜 Planned |
| Phase 6 | AI Chatbot (RAG) | 🔜 Planned |
| Phase 7 | Production Automation | 🔜 Planned |

---

## ✅ Phase 1 — Knowledge Ingestion Pipeline

**Status: COMPLETE**

The system successfully:

- Crawls KUCET webpages using a recursive BFS crawler
- Discovers internal links, dropdowns, and nested pages
- Discovers and downloads PDF resources
- Parses multi-page PDFs (text + tables) using `pdfplumber`
- Cleans extracted content (banners, headers, footers)
- Generates structured per-page JSON documents

### 1.1 Website Scraper

- Static page scraping, HTTP fetching, HTML retrieval

### 1.2 HTML Content Cleaning

- Navigation, footer, sidebar, script, style and boilerplate removal

### 1.3 Metadata Extraction

- Title, URL, page classification, content statistics

### 1.4 Deep BFS Crawler

- Breadth-First Search, recursive internal link discovery, duplicate prevention, domain restriction

### 1.5 Navigation Discovery

- Navbar, dropdown, sidebar, nested page traversal — discovers resource pages hidden inside menus

### 1.6 PDF Discovery

- PDF URL detection, recursive PDF discovery across all crawled pages

### 1.7 PDF Download Manager

- Automatic downloading, duplicate prevention, download tracking

Output directory: `data/pdfs/`

### 1.8 PDF Parsing

Technology: `pdfplumber`

- Multi-page text extraction, embedded table extraction, robust error handling

### 1.9 PDF Cleanup

- Header/footer removal, university and department banner stripping, duplicate content filtering

### 1.10 Structured JSON Generation

Output per document:

```json
{
  "title": "",
  "source_pdf": "",
  "page_count": 0,
  "pages": [
    { "page_number": 1, "content": "" }
  ]
}
```

Output directory: `data/pdf_text/`

---

## ✅ Phase 2 — PDF Intelligence Pipeline

**Status: COMPLETE**

Phase 2 introduces a production-grade 8-stage pipeline that transforms raw extracted text into deeply understood, semantically structured documents ready for embedding.

### Full Pipeline

```text
JSON (raw extracted text)
        ↓
[Phase 2-1]  Feature Extractor         → FeatureMap
        ↓
[Phase 2-2]  Candidate Scorer          → RankedCandidates
        ↓
[Phase 2-3]  Confidence Analyzer       → ConfidenceResult
        ↓
[Phase 2-4]  Document Validator        → ValidationResult
        ↓
[Phase 2-5]  FinalClassification       → document_type (single source of truth)
        ↓
[Phase 2-6]  ChunkerStrategy           → Chunks
        ↓
[Stage 1]    Layout Analyzer           → DocumentLayout (line-level metadata)
        ↓
[Stage 2]    Block Detector            → DocumentBlocks (semantic blocks)
        ↓
Output JSON  (chunks + layout_analysis + document_blocks)
```

---

### 2.1 Feature Extractor (`pdf_chunker.py`)

Extracts over 30 structured signals from raw page text:

- Course code counts, unit counts, LTPC tables
- Date/time/exam pattern counts
- Fee, scholarship, notification, workshop keyword presence
- Academic year, venue, semester labels
- Signature presence, annexure count, authority mentions

---

### 2.2 Candidate Scorer (`pdf_chunker.py`)

Scores every known document type (15+ types) against the feature map using weighted evidence:

| Document Type | Examples |
|---|---|
| `syllabus` | B.Tech course syllabus PDFs |
| `fee_notice` | Examination fee notifications |
| `notification` | General university circulars |
| `examination_schedule` | Timetables |
| `quotation` | Purchase/tender notices |
| `regulations` | Rules and regulations documents |
| `scholarship` | Scholarship notices |
| `circular` | Official circulars |
| ... | and 8 more types |

Produces per-type scores with strong-evidence and weak-evidence breakdowns.

---

### 2.3 Confidence Analyzer (`pdf_chunker.py`)

- Computes gap between winner and runner-up
- Assigns confidence level: `HIGH`, `MEDIUM`, `LOW`
- Detects contradictions (when runner-up also has strong signals)

---

### 2.4 Document Validator (`pdf_chunker.py`)

Hard-validates the classification winner against type-specific rules:

- A `regulations` document must contain attendance/grading language
- A `syllabus` must contain unit count > 0 or course codes
- A `fee_notice` must contain payment amounts

Overrides classification to `UNKNOWN` if rules fail.

---

### 2.5 FinalClassification (`pdf_chunker.py`)

Single authoritative classification output used by all downstream stages. Every stage reads `document_type` exclusively from here.

---

### 2.6 ChunkerStrategy (`pdf_chunker.py`)

Document-type-aware chunking. Each type gets its own `_BaseChunker` subclass:

- `SyllabusChunker` — chunks by UNIT/subject boundaries
- `FeeNoticeChunker` — chunks into notification + fee particulars sections
- `NotificationChunker` — sections by heading markers
- `QuotationChunker` — sections by document structure
- `RegulationsChunker` — sections by numbered rule clauses
- + fallback fixed-size chunker for `UNKNOWN` types

Chunk content is cleaned before saving:
- Letterhead lines stripped
- Ref-number/date banner lines stripped
- Signature blocks stripped
- Pipe-delimited table artifacts removed
- Small chunks (< 15 words) merged into neighbours

Output per chunk:

```json
{
  "chunk_id": "fee_notice_001_s001",
  "chunk_type": "section",
  "section_title": "NOTIFICATION",
  "page_start": 1,
  "page_end": 1,
  "content": "...",
  "metadata": {
    "document_type": "fee_notice",
    "word_count": 83
  }
}
```

---

### 2.7 Layout Analyzer (`scraper/layout_analyzer.py`) — Stage 1

A pre-chunking universal document analysis module.

**Input:** Raw page text (list of `PdfPage` objects)

**Output:** `DocumentLayout` — one `LineAnalysis` per line, each containing:

```json
{
  "page": 1,
  "line_number": 14,
  "text": "3. ATTENDANCE",
  "line_type": "heading",
  "scores": {
    "heading": 0.96,
    "paragraph": 0.03,
    "table": 0.01,
    "noise": 0.00
  },
  "features": {
    "word_count": 2,
    "uppercase_ratio": 0.83,
    "blank_before": true,
    "blank_after": true,
    "starts_with_number": true,
    "contains_course_code": false
  }
}
```

**Structural types classified per line:**

`heading` · `paragraph` · `table` · `list` · `header` · `footer` · `signature` · `reference` · `address` · `caption` · `noise` · `unknown`

**Features extracted per line (29 signals):**

- Textual: word count, character count, uppercase/lowercase/digit/punctuation ratios
- Structural: starts_with_number, starts_with_bullet, ends_with_colon, ends_with_period
- Contextual: blank_before, blank_after, indentation_level, position_from_top/bottom
- Semantic: contains_currency, contains_date, contains_url, contains_course_code, contains_unit_label, contains_table_separator
- Document-level: repeated_on_multiple_pages (header/footer detection)

Scoring is deterministic and rule-based — no ML, no LLM.

---

### 2.8 Block Detector (`scraper/block_detector.py`) — Stage 2

Groups individual analysed lines into semantic document blocks.

**Input:** `DocumentLayout.lines` (from Stage 1)

**Output:** `DocumentBlocks` — a list of `Block` objects

**Block types supported:**

`heading` · `paragraph` · `table` · `list` · `reference` · `signature` · `address` · `metadata` · `caption` · `unknown`

**Each block contains:**

```json
{
  "block_id": 12,
  "block_type": "paragraph",
  "page_start": 3,
  "page_end": 4,
  "line_start": 18,
  "line_end": 31,
  "heading_level": 0,
  "confidence": 0.97,
  "text": "3.1 Candidates admitted to a particular programme...",
  "children": [{ "line_number": 18, "text": "...", "line_type": "paragraph" }],
  "previous_block": 11,
  "next_block": 13,
  "parent_heading": 10
}
```

**Key capabilities:**

- **Paragraph continuation**: Absorbs mid-sentence OCR line-wraps into a single block. `unknown`-typed lines that start lowercase or follow an open sentence are merged into the preceding paragraph.
- **Table grouping**: All consecutive `table`-typed lines merged into one Table Block.
- **List merging**: Numbered, bulleted, alphabetical, and Roman-numeral list items grouped into one List Block.
- **Heading hierarchy**: Estimates `heading_level` (1–3) from numbering depth (`1.`, `3.1`, `3.1.1`), UNIT labels, and capitalisation. Uses a heading stack to track `parent_heading` for every non-heading block.
- **Metadata isolation**: Repeated header/footer lines grouped into dedicated Metadata Blocks, separated from content.
- **Junk rejection**: Blocks containing only noise tokens, single characters, or page numbers are discarded.

**Example results across corpus:**

| Document | Lines | Blocks | Compression |
|---|---|---|---|
| rules.json (12-page regulations) | 480 | 122 | 74% |
| btech_year1_sem1.json (syllabus) | 447 | 153 | 66% |
| B.TECH fee notification | 68 | 32 | 53% |

---

### Phase 2 Output JSON Structure

Each processed document produces a `*_chunks.json` file containing:

```json
{
  "source_pdf": "document.pdf",
  "document_type": "fee_notice",
  "document_subtype": "finance",
  "chunk_count": 2,
  "chunks": [ ... ],
  "layout_analysis": {
    "source_name": "document.json",
    "total_lines": 68,
    "total_pages": 1,
    "lines": [ ... ]
  },
  "document_blocks": {
    "source_name": "document.json",
    "total_lines": 68,
    "total_pages": 1,
    "total_blocks": 32,
    "blocks": [ ... ]
  }
}
```

Output directory: `data/pdf_chunks/`

---

## 🔄 Phase 3 — Embedding Generation (In Progress)

```text
document_blocks (Stage 2 output)
        ↓
Block-aware text assembly
        ↓
Embedding Model
        ↓
Vectors + Metadata
        ↓
Vector Store
```

**Planned tasks:**

- Consume `document_blocks` instead of raw chunk text
- Use block hierarchy (parent heading, block type) to enrich embedding context
- Generate dense vector embeddings per semantic block
- Preserve all metadata (document type, page range, heading context, confidence)
- Support batch processing of all documents

---

## 📂 Project Structure

```text
campus-ai-agent/
│
├── api/
│   └── main.py                    # FastAPI backend
│
├── chatbot/
│   ├── index.html                 # Chat UI
│   └── script.js
│
├── scraper/
│   ├── scrape.py                  # Website scraper
│   ├── crawl.py                   # BFS crawler
│   ├── pdf_discovery.py           # PDF link discovery
│   ├── pdf_parser.py              # PDF text extraction
│   ├── pdf_chunker.py             # Phase 2 main pipeline (8 stages)
│   ├── layout_analyzer.py         # Stage 1: Line-level layout analysis
│   └── block_detector.py          # Stage 2: Semantic block grouping
│
├── data/
│   ├── pdfs/                      # Downloaded PDF files
│   ├── pdf_text/                  # Phase 1 output: raw extracted JSON
│   └── pdf_chunks/                # Phase 2 output: chunks + layout + blocks
│
├── assets/
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone <repository-url>
cd campus-ai-agent
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate — Windows:

```bash
.venv\Scripts\activate
```

Activate — Linux / Mac:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

### 5. Run the Phase 2 Pipeline

Process a single document:

```bash
python -m scraper pdf_chunker --file data/pdf_text/your_document.json --out data/pdf_chunks
```

Process all documents:

```bash
python -m scraper pdf_chunker --all --input-dir data/pdf_text --out data/pdf_chunks
```

### 6. Run Backend

```bash
uvicorn api.main:app --reload
```

Backend: `http://127.0.0.1:8000`

### 7. Run Frontend

Open `chatbot/index.html` in a browser, or use Live Server.

---

## 🛠 Tech Stack

### Backend
- Python 3.11+
- FastAPI
- Uvicorn

### Frontend
- HTML / CSS / JavaScript

### Data Collection
- `requests`
- `BeautifulSoup4`

### PDF Processing
- `pdfplumber`

### Phase 2 — PDF Intelligence
- Pure Python (stdlib only) — deterministic, rule-based
- `re` module for structural pattern matching
- No ML, no LLM, no fuzzy matching

### AI
- Groq API (LLM inference)

### Planned (Phase 3+)
- `sentence-transformers` — dense embedding generation
- `ChromaDB` — local vector database
- `Supabase` — optional cloud vector store

---

## ⚠️ Current Limitations

- Embedding generation not yet implemented (Phase 3 in progress)
- Semantic vector retrieval not yet implemented
- Chatbot currently responds without retrieval context
- No scheduled auto-update of crawled content
- Block Detector confidence scores use softmax (multi-label independence planned for v2)
- Geometry features (bounding boxes, font sizes) not yet available — pipeline operates on extracted text only

---

## 📐 Architecture Decisions

| Decision | Rationale |
|---|---|
| Rule-based pipeline | Fully deterministic, debuggable, no GPU required |
| Document-type-aware chunking | Avoids one-size-fits-all splitting that breaks semantic boundaries |
| Line → Block two-stage analysis | Lines are not semantic units; blocks are the minimum meaningful unit |
| Layout + Block as separate stages | Separation of concerns — each stage has one responsibility |
| Softmax scoring in Layout Analyzer | Simple v1 implementation; planned upgrade to independent sigmoid scores |
| `document_blocks` in output JSON | Makes Stage 3 (Embeddings) independent of raw text re-parsing |

---

## 🔜 Upcoming Phases

### Phase 4 — ChromaDB Integration
- Collection creation, vector storage, similarity search

### Phase 5 — Retrieval Pipeline
- Semantic question embedding, cosine similarity search, top-k chunk retrieval

### Phase 6 — AI Chatbot (Full RAG)
- Context-aware responses, source-grounded answers, citation support

### Phase 7 — Production Automation
- Scheduled crawling, automatic PDF updates, automatic embedding refresh, deployment pipeline

---

## 📝 Notes

- Developed as a Mini Project.
- Designed with scalability and clean architecture in mind.
- Data quality and structural accuracy are prioritized before AI integration.
- Retrieval quality is directly dependent on the ingestion and intelligence pipeline quality.
- Phase 2 architecture was designed to match principles used by enterprise Document AI systems (Azure Document Intelligence, Amazon Textract).

---

## 👨‍💻 Author

Developed as part of the Interactive Campus Info AI Agent Mini Project.
