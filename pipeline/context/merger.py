from typing import List, Tuple
from pipeline.retrieval.search_models import RetrievedChunk
from pipeline.context.models import ContextBlock

class ChunkMerger:
    def __init__(self):
        self.merged_count = 0
        self.duplicate_count = 0
        
    def process_document_group(self, chunks: List[RetrievedChunk]) -> List[ContextBlock]:
        if not chunks:
            return []
            
        unique_chunks = self._deduplicate(chunks)
        blocks = self._merge_adjacent(unique_chunks)
        
        return blocks
        
    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        highest_ranked = {}
        for c in chunks:
            key = c.chunk_id
            if key not in highest_ranked or c.rank < highest_ranked[key].rank:
                highest_ranked[key] = c
                
        seen_texts = set()
        final_unique = set()
        for c in highest_ranked.values():
            text_key = c.text.strip().lower()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                final_unique.add(c.chunk_id)
            else:
                self.duplicate_count += 1
                
        # Preserve original sorted order (page -> index) by iterating over original list
        result = []
        seen = set()
        for c in chunks:
            if c.chunk_id in final_unique and c.chunk_id not in seen:
                seen.add(c.chunk_id)
                result.append(c)
            elif c.chunk_id not in final_unique and c.chunk_id not in seen:
                # We already counted text duplicates, if it's an ID duplicate we count here
                self.duplicate_count += 1
                seen.add(c.chunk_id)
                
        return result

    def _merge_adjacent(self, chunks: List[RetrievedChunk]) -> List[ContextBlock]:
        blocks = []
        current_block = None
        
        for chunk in chunks:
            if current_block is None:
                current_block = self._create_block(chunk)
                continue
                
            same_heading = chunk.heading == current_block.heading
            
            curr_idx = self._get_chunk_idx(chunk.chunk_id)
            prev_idx = self._get_chunk_idx(current_block.chunk_ids[-1])
            is_adjacent = (curr_idx == prev_idx + 1) if curr_idx is not None and prev_idx is not None else False
            
            if same_heading and is_adjacent:
                current_block.text += "\n\n" + chunk.text
                current_block.chunk_ids.append(chunk.chunk_id)
                current_block.score = max(current_block.score, chunk.score)
                
                page = chunk.metadata.get("page")
                if page:
                    page_str = str(page)
                    pages_list = current_block.pages.split(",")
                    if page_str not in pages_list and page_str != "":
                        if current_block.pages == "":
                            current_block.pages = page_str
                        else:
                            current_block.pages += f",{page_str}"
                        
                self.merged_count += 1
            else:
                blocks.append(current_block)
                current_block = self._create_block(chunk)
                
        if current_block:
            blocks.append(current_block)
            
        return blocks
        
    def _create_block(self, chunk: RetrievedChunk) -> ContextBlock:
        return ContextBlock(
            document=chunk.document,
            heading=chunk.heading,
            pages=str(chunk.metadata.get("page", "")),
            source=chunk.source,
            score=chunk.score,
            text=chunk.text,
            metadata=chunk.metadata,
            chunk_ids=[chunk.chunk_id]
        )
        
    def _get_chunk_idx(self, chunk_id: str):
        if "chunk_" in chunk_id:
            try:
                return int(chunk_id.split("chunk_")[-1])
            except ValueError:
                pass
        return None
