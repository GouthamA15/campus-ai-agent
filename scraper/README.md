# Scraper (Phase 1.5) + Crawler (Phase 2)

This module scrapes **one HTML page**, cleans the raw HTML (removing scripts, styles, sidebars, and navigation), and saves the plain text result as **one JSON file** in `data/processed/`.

Out of scope in this repo stage:
- Embeddings / vector DB
- RAG retrieval

## Run

From the repo root:

```bash
python -m scraper --url "https://example.com"
```

## Output JSON

Each URL produces one JSON file:

```json
{
	"title": "...",
	"content": "...",
	"source": "...",
	"scraped_at": "...",
	"content_length": 1234,
	"page_type": "general"
}
```

`page_type` is classified with simple rules (homepage/department/alumni/notice/general).

Optional:

```bash
python -m scraper --url "https://example.com" --out data/processed --verbose
```

## Phase 2: Controlled Crawling (kucet.ac.in only)

This crawls internal links starting from `http://kucet.ac.in/` using a queue-based BFS.

Note: link discovery happens **before** content cleaning, so navbar/dropdown links are not lost.

Limits:
- `--max-pages` (default: 200)
- `--max-depth` (default: 5)

```bash
python -m scraper.crawl --start-url "http://kucet.ac.in/" --max-pages 200 --max-depth 5
```

Debug (prints all discovered URLs per page):

```bash
python -m scraper.crawl --start-url "http://kucet.ac.in/" --max-pages 5 --max-depth 1 --show-links --verbose
```

It prints a crawl summary like:

```json
{
	"pages_crawled": 0,
	"pages_saved": 0,
	"links_discovered": 0,
	"failed_pages": 0
	"max_depth_reached": 0
}
```

## Phase 3: PDF Discovery + Download

While crawling HTML pages, this discovers PDF links and downloads them.

Outputs:
- `data/pdf_metadata/` (one JSON metadata file per discovered PDF)
- `data/pdfs/` (downloaded PDF files)

Run:

```bash
python -m scraper.pdf_discovery --start-url "http://kucet.ac.in/" --max-pages 50 --max-depth 5
```

Debug (prints all discovered URLs per page):

```bash
python -m scraper.pdf_discovery --start-url "http://kucet.ac.in/" --max-pages 5 --max-depth 1 --show-links --verbose
```

It prints a final summary like:

```json
{
	"pages_crawled": 0,
	"pdfs_discovered": 0,
	"pdfs_downloaded": 0,
	"pdfs_skipped": 0,
	"failed_pdfs": 0
}
```

## Phase 4: PDF Text Extraction (No OCR)

This reads PDFs from `data/pdfs/` and writes extracted text JSON into `data/pdf_text/`.

Output format is page-structured (one entry per PDF page) and also includes `full_content` (concatenated) for debugging/backward compatibility.

Example output JSON:

```json
{
	"title": "...",
	"source_pdf": "data/pdfs/example.pdf",
	"page_count": 2,
	"extracted_at": "2026-01-01T00:00:00+00:00",
	"content_length": 1234,
	"document_type": "circular",
	"text_extracted": true,
	"needs_ocr": false,
	"pages": [
		{"page_number": 1, "content": "Page 1 text..."},
		{"page_number": 2, "content": "Page 2 text..."}
	],
	"full_content": "Page 1 text...\n\nPage 2 text..."
}
```

Migration note: older outputs may contain a single `content` string. Re-run `python -m scraper.pdf_parser --all` (or `--file ...`) to regenerate JSONs with `pages` + `full_content`.

Additional metadata:
- `document_type` is classified using deterministic keyword rules (no AI/LLM).
- `text_extracted` is `false` when `content_length == 0`.
- If `needs_ocr == true`, the JSON includes `ocr_status: "pending"` (Phase 4 does not run OCR; it only marks readiness).

Parse a single PDF:

```bash
python -m scraper.pdf_parser --file data/pdfs/sample.pdf
```

Parse a single PDF and remove repeated headers/footers:

```bash
python -m scraper.pdf_parser --file data/pdfs/sample.pdf --cleanup
```

Cleanup details:
- Uses pdfplumber word coordinates to collect candidate header/footer *lines* from:
	- Top zone: first ~15% of page height
	- Bottom zone: last ~10–15% of page height
- Normalizes aggressively (lowercase, strips punctuation/dashes, removes years/date-ranges/postal codes/standalone numbers)
- Clusters similar candidates using token-set similarity (Jaccard/overlap), so formatting variants still match
- Removes only the detected repeated header/footer lines (does not blindly crop regions)

Tune repeated-line threshold (fraction or percent):

```bash
python -m scraper.pdf_parser --file data/pdfs/sample.pdf --cleanup --cleanup-threshold 0.5
python -m scraper.pdf_parser --file data/pdfs/sample.pdf --cleanup --cleanup-threshold 50
```

Parse all PDFs:

```bash
python -m scraper.pdf_parser --all
```

It prints a summary like:

```json
{
  "pdfs_processed": 0,
  "pdfs_successful": 0,
  "pdfs_failed": 0
}
```

When cleanup is enabled, the output JSON includes:

```json
{
	"cleanup_applied": true,
	"removed_repeated_lines": [
		"KAKATIYA UNIVERSITY",
		"WARANGAL-506009"
	]
}
```

## Phase 2A: PDF Chunking Engine

This reads only the parsed PDF JSON from `data/pdf_text/` and writes retrieval-ready chunks to `data/pdf_chunks/`.

Run on one file:

```bash
python -m pipeline.pdf_chunker --file data/pdf_text/example.json
```

Run on all parsed PDF JSON files:

```bash
python -m pipeline.pdf.pdf_chunker --all
```

Chunking modes:
- Syllabus: unit/textbooks/references/course outcome sections
- Academic regulations: numbered or titled regulation sections
- Exam schedules and timetables: one complete timetable chunk
- Generic university documents: heading-based chunks, then fixed-size windows if no headings are found

For syllabus documents, unit chunks inherit optional subject metadata (`subject_name`, `course_code`) and use subject-prefixed chunk IDs when a subject heading is detected, for example `physics_unit_1`.

Output schema:

```json
{
	"source_pdf": "example.pdf",
	"document_type": "circular",
	"chunk_count": 2,
	"chunks": [
		{
			"chunk_id": "example_section_1",
			"chunk_type": "section",
			"section_title": "NOTICE",
			"page_start": 1,
			"page_end": 1,
			"content": "...",
			"metadata": {
				"source_pdf": "example.pdf",
				"document_type": "circular",
				"chunk_strategy": "heading_sections",
				"word_count": 142
			}
		}
	]
}
```

## Phase 2.5: Web Chunking Engine

Once web pages have been scraped and flattened into plain text in `data/processed/`, they must be chunked for retrieval. This is handled by the **Web Chunker** located in the `pipeline/web/` package.

The Web Chunker reads the scraped JSON files and converts them into semantic chunks using a deterministic, rule-based approach (No ML/LLM).

Run the web chunker:

```bash
python -m pipeline.web.web_chunker --input-dir data/processed --output-dir data/web_chunks
```

### How Web Chunking Works:

1. **Junk Filtering:** The chunker explicitly ignores lines matching known useless patterns like "Copyrights", "Developed by", "Navigation", "Quick Links", and repeating college banners.
2. **Heading Detection:** It uses heuristics to detect structural boundaries. Lines that are short, Title Cased, or ALL CAPS are treated as section headings.
3. **Semantic Grouping:** Text is grouped under its closest heading. It respects a `MAX_TOKENS` limit (~400) to prevent oversized chunks.
4. **Boundary Respect:** The chunker never splits inside a paragraph, table row, or list item. Splits only occur at natural line break boundaries.

### Output JSON Format

Chunks are saved in `data/web_chunks/` (one file per webpage):

```json
{
  "document_id": "college.php.json",
  "title": "KU COLLEGE OF ENGINEERING AND TECHNOLOGY",
  "source": "http://kucet.ac.in/college.php",
  "page_type": "department",
  "scraped_at": "2026-05-29T14:20:10.831602+00:00",
  "chunk_count": 2,
  "chunks": [
    {
      "chunk_id": "college.php.json_chunk_0",
      "chunk_index": 0,
      "heading": "KUWL",
      "heading_level": 1,
      "chunk_type": "paragraph",
      "text": "It was established in the year 2009 with the mission and vision...",
      "token_count": 416
    }
  ]
}
```
