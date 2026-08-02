from typing import List, Dict, Any
from pipeline.retrieval.search_models import RetrievedChunk

class OrderManager:
    def group_and_sort(self, chunks: List[RetrievedChunk]) -> Dict[str, List[RetrievedChunk]]:
        """Groups chunks by document and sorts them logically to preserve reading order."""
        grouped = {}
        for chunk in chunks:
            doc = chunk.document
            if doc not in grouped:
                grouped[doc] = []
            grouped[doc].append(chunk)
            
        for doc in grouped:
            grouped[doc].sort(key=self._sort_key)
            
        return grouped
        
    def _sort_key(self, chunk: RetrievedChunk) -> tuple:
        # 1. Page Number
        page = chunk.metadata.get("page", 0)
        
        # 2. Chunk Index
        chunk_idx = 0
        if "chunk_" in chunk.chunk_id:
            try:
                # E.g. rules.pdf_chunk_12 -> 12
                chunk_idx = int(chunk.chunk_id.split("chunk_")[-1])
            except ValueError:
                pass
                
        # 3. Fallback to original rank (from similarity)
        return (page, chunk_idx, chunk.rank)
