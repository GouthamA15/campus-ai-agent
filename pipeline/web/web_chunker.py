"""
Web Chunker Pipeline
====================
Converts processed JSON webpages into semantic retrieval chunks.
"""

import json
import re
from pathlib import Path
from typing import Any, List, Dict
from datetime import datetime

# ===========================================================================
# Configuration
# ===========================================================================

IGNORE_PATTERNS = [
    re.compile(r"Copyrights?.*?©", re.IGNORECASE),
    re.compile(r"All Rights Reserved", re.IGNORECASE),
    re.compile(r"Developed by", re.IGNORECASE),
    re.compile(r"KU COLLEGE OF ENGINEERING (AND|&) TECHNOLOGY", re.IGNORECASE),
    re.compile(r"KAKATIYA UNIVERSITY", re.IGNORECASE),
    re.compile(r"Warangal - 506009", re.IGNORECASE),
    re.compile(r"PGECET:\s*KUWL1?", re.IGNORECASE),
    re.compile(r"EAPCET:\s*KUWL", re.IGNORECASE),
    re.compile(r"ECET:\s*KUWL", re.IGNORECASE),
    re.compile(r"^KUWL1?$", re.IGNORECASE),
    re.compile(r"☎️?\s*Contact:\s*\d+", re.IGNORECASE),
    re.compile(r"^Quick Links$", re.IGNORECASE),
    re.compile(r"^Navigation$", re.IGNORECASE),
]

MAX_TOKENS = 400

# ===========================================================================
# Helpers
# ===========================================================================

def _is_junk_line(line: str) -> bool:
    """Return True if line matches any ignored patterns."""
    if not line.strip():
        return True
    for p in IGNORE_PATTERNS:
        if p.search(line):
            return True
    return False

def _is_heading(line: str) -> bool:
    """Heuristic to detect a heading line."""
    line = line.strip()
    if len(line) < 5 or len(line) > 80:
        return False
    if line.isdigit():
        return False
    # A heading usually has characters, not just symbols/numbers
    if not any(c.isalpha() for c in line):
        return False
    words = line.split()
    # Require at least 2 words unless it ends with colon
    if len(words) < 2 and not line.endswith(':'):
        return False
    if line.isupper():
        return True
    if line.istitle() and len(words) >= 2:
        return True
    if line.endswith(':'):
        return True
    return False

def _estimate_tokens(text: str) -> int:
    """Rough token estimation (words + punctuation)."""
    return len(re.findall(r'\w+|[^\w\s]', text))

# ===========================================================================
# Main Chunker
# ===========================================================================

class WebChunker:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def chunk_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single processed webpage JSON into a chunked document JSON."""
        doc_id = doc.get('source', '').split('/')[-1] or "index"
        if not doc_id.endswith('.json'):
            doc_id += ".json"
            
        content = doc.get('content', '')
        lines = content.split('\n')
        
        chunks = []
        current_chunk_lines = []
        current_heading = doc.get('title', 'Unknown Section')
        current_heading_level = 1
        
        def save_chunk():
            nonlocal current_chunk_lines
            if not current_chunk_lines:
                return
            
            text = '\n'.join(current_chunk_lines).strip()
            if not text:
                return
                
            token_count = _estimate_tokens(text)
            
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "chunk_index": len(chunks),
                "heading": current_heading,
                "heading_level": current_heading_level,
                "chunk_type": "paragraph",
                "text": text,
                "token_count": token_count
            })
            current_chunk_lines = []

        for line in lines:
            line = line.strip()
            if _is_junk_line(line):
                continue
                
            if _is_heading(line):
                # If we have accumulated text and hit a new heading, save the old chunk
                if current_chunk_lines:
                    save_chunk()
                current_heading = line
                current_heading_level = 1 if line.isupper() else 2
                current_chunk_lines.append(line)
            else:
                current_chunk_lines.append(line)
                # If current chunk exceeds MAX_TOKENS, we try to break here since it's a line break.
                # In plain text, a line break is a natural paragraph boundary.
                if _estimate_tokens('\n'.join(current_chunk_lines)) > MAX_TOKENS:
                    save_chunk()
                    
        # Save remaining
        if current_chunk_lines:
            save_chunk()
            
        # Post-process: Filter out chunks that are just one line or have no real content
        valid_chunks = []
        for i, c in enumerate(chunks):
            if c['token_count'] > 2:
                c['chunk_index'] = len(valid_chunks)
                c['chunk_id'] = f"{doc_id}_chunk_{len(valid_chunks)}"
                valid_chunks.append(c)

        return {
            "document_id": doc_id,
            "title": doc.get('title', ''),
            "source": doc.get('source', ''),
            "page_type": doc.get('page_type', ''),
            "scraped_at": doc.get('scraped_at', ''),
            "chunk_count": len(valid_chunks),
            "chunks": valid_chunks
        }

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Process all JSONs in input_dir and save to output_dir."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = 0
        failed = 0
        total_chunks = 0
        
        for json_path in input_dir.glob("*.json"):
            try:
                print(f"Processing: {json_path.name}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                chunked_doc = self.chunk_document(doc)
                
                out_path = output_dir / f"{json_path.stem}_chunks.json"
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(chunked_doc, f, ensure_ascii=False, indent=2)
                
                num_chunks = chunked_doc['chunk_count']
                print(f"Chunks Created: {num_chunks}")
                print("-" * 32)
                
                success += 1
                total_chunks += num_chunks
            except Exception as e:
                print(f"Failed to process {json_path.name}: {e}")
                failed += 1
                
        print("At completion")
        print("Web Pages Processed")
        print(f"Successful: {success}")
        print(f"Failed: {failed}")
        print(f"Total Chunks: {total_chunks}")

def main(argv: List[str] = None):
    import argparse
    parser = argparse.ArgumentParser(description="Web Chunker")
    parser.add_argument("--input-dir", default="data/processed", help="Input directory")
    parser.add_argument("--output-dir", default="data/web_chunks", help="Output directory")
    args = parser.add_argument_args() if hasattr(parser, 'add_argument_args') else parser.parse_args(argv)
    
    chunker = WebChunker()
    chunker.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
