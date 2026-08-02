"""
block_detector.py — Stage 2: Universal Block Detector
======================================================

Consumes the output of the Layout Analyzer (Stage 1) and groups
individual analysed lines into semantic document blocks.

Why blocks matter
-----------------
A line is a physical artifact of the PDF page width and font size.
A BLOCK is the minimum meaningful semantic unit:

  - A Paragraph Block is one wrapped idea, possibly spanning many lines
    or crossing a page boundary.
  - A Table Block is all rows of one table, merged.
  - A List Block is a complete numbered or bulleted list.
  - A Heading Block anchors the hierarchy.

Pipeline position
-----------------
  Layout Analyzer (Stage 1)   →   Block Detector (Stage 2)   →   Chunk Builder (Stage 3)

Input
-----
  DocumentLayout  (from pipeline.pdf.layout_analyzer)
  - OR -
  A raw list of line-dicts exactly as stored in the chunks JSON under
  the "layout_analysis" → "lines" key.

Output
------
  DocumentBlocks  — a list of Block objects with relationships wired up.

Design constraints
------------------
  * Deterministic, rule-based.  No AI / ML / fuzzy matching.
  * Document-independent: no PDF-specific or university-specific logic.
  * Never re-analyses raw text; uses only the line_type + features provided
    by Stage 1.
  * The existing Chunk Builder is UNCHANGED; Stage 2 adds a new higher-level
    representation alongside it.
  * Saves block_analysis.json when debug=True.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ===========================================================================
# Block type constants
# ===========================================================================

class BlockType:
    HEADING   = "heading"
    PARAGRAPH = "paragraph"
    TABLE     = "table"
    LIST      = "list"
    REFERENCE = "reference"
    SIGNATURE = "signature"
    ADDRESS   = "address"
    METADATA  = "metadata"
    CAPTION   = "caption"
    UNKNOWN   = "unknown"


# ===========================================================================
# Block data class
# ===========================================================================

@dataclass
class Block:
    """One semantic document block (group of compatible adjacent lines)."""
    block_id:      int
    block_type:    str
    page_start:    int
    page_end:      int
    line_start:    int
    line_end:      int
    heading_level: int              # 0 = not a heading, 1 = top, 2 = sub, ...
    confidence:    float
    text:          str              # joined text of all child lines
    children:      list[dict]       # list of {line_number, text, line_type}

    # Relationship fields (filled after all blocks are built)
    previous_block: Optional[int] = None   # block_id of predecessor
    next_block:     Optional[int] = None   # block_id of successor
    parent_heading: Optional[int] = None   # block_id of nearest enclosing heading


@dataclass
class DocumentBlocks:
    """Complete block-level analysis for one document."""
    source_name:   str
    total_lines:   int
    total_pages:   int
    total_blocks:  int
    blocks:        list[Block] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name":  self.source_name,
            "total_lines":  self.total_lines,
            "total_pages":  self.total_pages,
            "total_blocks": self.total_blocks,
            "blocks": [asdict(b) for b in self.blocks],
        }


# ===========================================================================
# Heading-level estimator
# ===========================================================================

# Numbered section patterns  (match from left):
_RE_LEVEL_1 = re.compile(r"^\d+\s*[.)]\s+", re.I)              # "1. TITLE"
_RE_LEVEL_2 = re.compile(r"^\d+\.\d+\s+", re.I)                # "3.1 Sub"
_RE_LEVEL_3 = re.compile(r"^\d+\.\d+\.\d+\s+", re.I)           # "3.1.1 Deep"
_RE_UNIT_LABEL = re.compile(r"^(?:UNIT|MODULE|CHAPTER)\s*[-–]?\s*(?:[IVXLCDM]+|\d+)\b", re.I)

def _heading_level(text: str, uppercase_ratio: float) -> int:
    """
    Estimate heading depth 1–4 from numbering pattern and capitalisation.

    Returns 0 for non-heading lines (but this function is only called on
    lines already confirmed as heading-type).
    """
    if _RE_LEVEL_3.match(text):
        return 3
    if _RE_LEVEL_2.match(text):
        return 2
    if _RE_LEVEL_1.match(text):
        return 1
    if _RE_UNIT_LABEL.match(text):
        return 2
    # Fully uppercase with no numbering → treat as level 1
    if uppercase_ratio >= 0.75:
        return 1
    return 1   # fallback for any other detected heading


# ===========================================================================
# Line compatibility rules
# ===========================================================================

# "Continuation-compatible" types: a line of type B can be added to a
# running block of type A.  Key = running block type, Value = set of line
# types that may join it.
_COMPAT: dict[str, frozenset[str]] = {
    BlockType.HEADING:   frozenset({"heading"}),
    BlockType.PARAGRAPH: frozenset({"paragraph", "unknown", "caption"}),
    BlockType.TABLE:     frozenset({"table", "noise", "unknown"}),
    BlockType.LIST:      frozenset({"list", "unknown"}),
    BlockType.REFERENCE: frozenset({"reference", "unknown"}),
    BlockType.SIGNATURE: frozenset({"signature", "unknown"}),
    BlockType.ADDRESS:   frozenset({"address", "unknown"}),
    BlockType.METADATA:  frozenset({"header", "footer", "noise", "unknown", "metadata"}),
    BlockType.CAPTION:   frozenset({"caption", "unknown"}),
    BlockType.UNKNOWN:   frozenset({"unknown"}),
}

# Line types that open a *new* block (never continue an existing one)
_HARD_BREAKERS = frozenset({
    "heading",
    "header",
    "footer",
    "signature",
})

# Line types that are silently discarded as standalone blocks (too small)
_DISCARD_SOLO_TYPES = frozenset({
    "noise",
})

# Mapping from Layout Analyzer line_type → Block type used as the block opener
_LINE_TYPE_TO_BLOCK_TYPE: dict[str, str] = {
    "heading":   BlockType.HEADING,
    "paragraph": BlockType.PARAGRAPH,
    "table":     BlockType.TABLE,
    "list":      BlockType.LIST,
    "reference": BlockType.REFERENCE,
    "signature": BlockType.SIGNATURE,
    "address":   BlockType.ADDRESS,
    "header":    BlockType.METADATA,
    "footer":    BlockType.METADATA,
    "noise":     BlockType.UNKNOWN,
    "caption":   BlockType.CAPTION,
    "unknown":   BlockType.UNKNOWN,
}


def _is_junk_line(line: dict) -> bool:
    """True for lines that carry no semantic content and should be skipped."""
    text = (line.get("text") or "").strip()
    lt   = line.get("line_type", "")
    if not text:
        return True
    if lt == "noise" and len(text) <= 3:
        return True
    return False


def _can_continue(block_type: str, line_type: str) -> bool:
    """Return True if a line of line_type may be added to an open block_type."""
    return line_type in _COMPAT.get(block_type, frozenset())


def _is_paragraph_continuation(prev_line: dict, curr_line: dict) -> bool:
    """
    Heuristic: does the current line look like a continuation of the
    previous paragraph line?  Uses the fact that OCR-extracted text often
    wraps mid-sentence, so continuation lines:
      - lack an initial capital (or start with lower/mid-sentence words)
      - 'unknown' classified (insufficient signals for Stage 1)
      - previous line does NOT end with sentence-final punctuation
    """
    prev_text = (prev_line.get("text") or "").strip()
    curr_text = (curr_line.get("text") or "").strip()
    curr_lt   = curr_line.get("line_type", "")

    if curr_lt not in ("unknown", "paragraph"):
        return False
    if not prev_text or not curr_text:
        return False

    # Previous line ends mid-sentence (no period / semicolon / colon)
    prev_ends_open = prev_text[-1] not in ".;:!?"

    # Current line starts with lowercase (definite continuation)
    starts_lower = curr_text[0].islower()

    # Current line is classified 'unknown' (weak signals → likely continuation)
    is_unknown = curr_lt == "unknown"

    return prev_ends_open and (starts_lower or is_unknown)


# ===========================================================================
# Block builder helpers
# ===========================================================================

def _open_block(block_id: int, line: dict) -> Block:
    """Start a new block from a single seed line."""
    lt        = line.get("line_type", "unknown")
    block_type = _LINE_TYPE_TO_BLOCK_TYPE.get(lt, BlockType.UNKNOWN)
    feat       = line.get("features", {})
    text       = (line.get("text") or "").strip()
    page       = line.get("page", 0)
    line_no    = line.get("line_number", 0)
    score_val  = max(line.get("scores", {}).values(), default=0.0)

    heading_level = 0
    if block_type == BlockType.HEADING:
        heading_level = _heading_level(text, feat.get("uppercase_ratio", 0.0))

    return Block(
        block_id      = block_id,
        block_type    = block_type,
        page_start    = page,
        page_end      = page,
        line_start    = line_no,
        line_end      = line_no,
        heading_level = heading_level,
        confidence    = round(score_val, 4),
        text          = text,
        children      = [{"line_number": line_no,
                          "text": text,
                          "line_type": lt}],
    )


def _grow_block(block: Block, line: dict) -> None:
    """Append a compatible line to an existing block (mutates block in place)."""
    lt   = line.get("line_type", "unknown")
    text = (line.get("text") or "").strip()
    page = line.get("page", 0)
    line_no = line.get("line_number", 0)

    block.page_end = max(block.page_end, page)
    block.line_end = line_no
    block.text = (block.text + " " + text).strip()
    block.children.append({"line_number": line_no,
                            "text": text,
                            "line_type": lt})


def _is_valid_block(block: Block) -> bool:
    """
    Return False for blocks that contain only garbage content.
    These will be dropped or merged into neighbours.
    """
    text = block.text.strip()
    if not text:
        return False
    # Must have at least one alphabetic character
    if not any(c.isalpha() for c in text):
        return False
    # Single char or single digit
    if len(text) <= 2:
        return False
    # Pure page-number-like content
    if re.fullmatch(r"[-–]?\s*\d+\s*[-–]?", text):
        return False
    return True


# ===========================================================================
# Block Detector  (main class)
# ===========================================================================

class BlockDetector:
    """
    Stage 2 — Universal Block Detector.

    Groups individual analysed lines (from Stage 1) into semantic blocks.

    Usage::

        from pipeline.pdf.block_detector import BlockDetector

        detector = BlockDetector(debug=True)
        doc_blocks = detector.detect(
            source_name="rules.json",
            layout_lines=layout.to_dict()["lines"],    # from DocumentLayout
        )

        for block in doc_blocks.blocks:
            print(block.block_id, block.block_type, block.text[:60])

        detector.save_debug(doc_blocks, output_dir="data/pdf_chunks")

    Parameters
    ----------
    debug : bool
        If True, saving block_analysis.json is performed by save_debug().
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    # ------------------------------------------------------------------
    def detect(
        self,
        source_name: str,
        layout_lines: list[dict[str, Any]],
    ) -> DocumentBlocks:
        """
        Main entry point.

        Parameters
        ----------
        source_name : str
            Identifier for this document (e.g. the filename).
        layout_lines : list[dict]
            The ``lines`` list from a ``DocumentLayout.to_dict()`` call,
            or from the ``layout_analysis.lines`` key in the chunk JSON.
            Each dict must have: text, line_type, page, line_number, features.
        """
        if not layout_lines:
            return DocumentBlocks(
                source_name=source_name,
                total_lines=0,
                total_pages=0,
                total_blocks=0,
            )

        total_pages = max(l.get("page", 0) for l in layout_lines)

        # ── Pass 1: Segment lines into raw blocks ───────────────────────────
        raw_blocks = self._segment(layout_lines)

        # ── Pass 2: Validate – drop pure-garbage blocks ─────────────────────
        valid_blocks = [b for b in raw_blocks if _is_valid_block(b)]

        # ── Pass 3: Re-number block IDs sequentially ────────────────────────
        for i, b in enumerate(valid_blocks):
            b.block_id = i

        # ── Pass 4: Wire relationships (prev/next/parent_heading) ───────────
        self._wire_relationships(valid_blocks)

        return DocumentBlocks(
            source_name  = source_name,
            total_lines  = len(layout_lines),
            total_pages  = total_pages,
            total_blocks = len(valid_blocks),
            blocks       = valid_blocks,
        )

    # ------------------------------------------------------------------
    def _segment(self, lines: list[dict]) -> list[Block]:
        """
        Core segmentation loop.

        State machine:
          - We keep an *open_block* that accumulates compatible lines.
          - When a line is incompatible (different semantic type, hard-breaker,
            or explicit paragraph terminator), we close the open block and
            start a new one.
          - 'unknown' typed lines are treated as paragraph continuations
            if the heuristic detects a mid-sentence wrap.
        """
        blocks: list[Block] = []
        open_block: Optional[Block] = None
        block_counter = 0

        for i, line in enumerate(lines):
            lt   = line.get("line_type", "unknown")
            text = (line.get("text") or "").strip()

            # ---- skip junk lines (no content, single-char noise) -----------
            if _is_junk_line(line):
                continue

            # ---- METADATA block: group header/footer/noise together --------
            if lt in ("header", "footer", "noise"):
                if open_block and open_block.block_type == BlockType.METADATA:
                    _grow_block(open_block, line)
                else:
                    if open_block:
                        blocks.append(open_block)
                    open_block = _open_block(block_counter, line)
                    block_counter += 1
                continue

            # ---- Hard-breaker types always start a fresh block -------------
            if lt in _HARD_BREAKERS:
                if open_block:
                    blocks.append(open_block)
                open_block = _open_block(block_counter, line)
                block_counter += 1
                continue

            # ---- Paragraph continuation check ------------------------------
            # If the current line is 'unknown' or 'paragraph' and looks like a
            # wrapped sentence from the previous line, absorb it.
            if open_block and lt in ("unknown", "paragraph"):
                prev_child = open_block.children[-1] if open_block.children else None
                if prev_child and _is_paragraph_continuation(prev_child, line):
                    # Promote block type to paragraph if it was unknown
                    if open_block.block_type == BlockType.UNKNOWN:
                        open_block.block_type = BlockType.PARAGRAPH
                    _grow_block(open_block, line)
                    continue

            # ---- Normal compatibility check --------------------------------
            if open_block and _can_continue(open_block.block_type, lt):
                _grow_block(open_block, line)
                continue

            # ---- Incompatible → close current block, open new one ----------
            if open_block:
                blocks.append(open_block)
            open_block = _open_block(block_counter, line)
            block_counter += 1

        # Close any remaining open block
        if open_block:
            blocks.append(open_block)

        return blocks

    # ------------------------------------------------------------------
    def _wire_relationships(self, blocks: list[Block]) -> None:
        """
        Set previous_block, next_block, and parent_heading for every block.
        """
        # Build heading stack: list of (block_id, heading_level)
        heading_stack: list[tuple[int, int]] = []

        for i, block in enumerate(blocks):
            # ---- prev / next -----------------------------------------------
            block.previous_block = blocks[i - 1].block_id if i > 0 else None
            block.next_block     = blocks[i + 1].block_id if i + 1 < len(blocks) else None

            # ---- parent heading --------------------------------------------
            if block.block_type == BlockType.HEADING:
                level = block.heading_level
                # Pop anything at same or deeper level from stack
                while heading_stack and heading_stack[-1][1] >= level:
                    heading_stack.pop()
                # Parent heading is now whatever is on top (or None)
                block.parent_heading = heading_stack[-1][0] if heading_stack else None
                # Push this heading onto the stack
                heading_stack.append((block.block_id, level))
            else:
                # Non-heading block: parent is the most recent heading on stack
                block.parent_heading = heading_stack[-1][0] if heading_stack else None

    # ------------------------------------------------------------------
    def save_debug(
        self,
        doc_blocks: DocumentBlocks,
        output_dir: str | Path,
        *,
        basename: str | None = None,
    ) -> Path:
        """Write <basename>_block_analysis.json to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem     = basename or Path(doc_blocks.source_name).stem
        out_path = output_dir / f"{stem}_block_analysis.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(doc_blocks.to_dict(), fh, ensure_ascii=False, indent=2)
        return out_path


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def detect_blocks(
    source_name: str,
    layout_lines: list[dict[str, Any]],
    debug: bool = False,
    output_dir: str | Path | None = None,
) -> DocumentBlocks:
    """
    Convenience wrapper around BlockDetector.detect().

    Parameters
    ----------
    source_name : str
        Name of the source document.
    layout_lines : list[dict]
        Lines from a DocumentLayout.to_dict()["lines"] call.
    debug : bool
        If True and output_dir is given, write the debug JSON.
    output_dir : Path or str, optional
        Directory for debug block_analysis.json output.
    """
    detector  = BlockDetector(debug=debug)
    doc_blocks = detector.detect(source_name=source_name, layout_lines=layout_lines)
    if debug and output_dir:
        detector.save_debug(doc_blocks, output_dir=output_dir)
    return doc_blocks
