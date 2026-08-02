import json
from pathlib import Path
from typing import List, Dict, Any

from pipeline.embeddings.models import EmbeddingDocument

def load_json_chunks(file_path: Path) -> List[EmbeddingDocument]:
    """Reads a chunk JSON file and converts it into a list of EmbeddingDocument objects."""
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    chunks = data.get("chunks", [])
    
    # Identify source type
    is_pdf = "source_pdf" in data
    
    for c in chunks:
        chunk_id = c.get("chunk_id", "")
        # Different keys for text depending on pipeline source
        text = c.get("text") or c.get("content") or ""
        
        # Build metadata by combining root metadata and chunk-level metadata
        metadata = {}
        if is_pdf:
            source = data.get("source_pdf", "")
            metadata["source_pdf"] = source
            metadata["document_type"] = data.get("document_type", "")
            # Merge chunk metadata
            chunk_meta = c.get("metadata", {})
            metadata.update(chunk_meta)
        else:
            source = data.get("source", "")
            metadata["title"] = data.get("title", "")
            metadata["page_type"] = data.get("page_type", "")
            metadata["scraped_at"] = data.get("scraped_at", "")
            metadata["heading"] = c.get("heading", "")
            metadata["heading_level"] = c.get("heading_level", 0)
            
        doc = EmbeddingDocument(
            chunk_id=chunk_id,
            source=source,
            text=text,
            metadata=metadata
        )
        docs.append(doc)
        
    return docs
