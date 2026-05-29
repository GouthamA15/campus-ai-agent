# Scraper (Phase 1.5) + Crawler (Phase 2)

This module scrapes **one HTML page** and saves the result as **one JSON file** in `data/processed/`.

Out of scope in this repo stage:
- PDFs
- chunking
- embeddings / vector DB
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
	"pages": [
		{"page_number": 1, "content": "Page 1 text..."},
		{"page_number": 2, "content": "Page 2 text..."}
	],
	"full_content": "Page 1 text...\n\nPage 2 text..."
}
```

Migration note: older outputs may contain a single `content` string. Re-run `python -m scraper.pdf_parser --all` (or `--file ...`) to regenerate JSONs with `pages` + `full_content`.

Parse a single PDF:

```bash
python -m scraper.pdf_parser --file data/pdfs/sample.pdf
```

Parse a single PDF and remove repeated headers/footers:

```bash
python -m scraper.pdf_parser --file data/pdfs/sample.pdf --cleanup
```

Cleanup details:
- Scans the first 10 and last 5 non-empty lines of each page
- Normalizes lines (lowercase, strips punctuation/dashes, removes years/date-ranges/postal codes/standalone numbers)
- Removes lines that appear on >= `--cleanup-threshold` fraction of pages

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
