from typing import List
from pipeline.retrieval.search_models import RetrievedChunk

class Ranker:
    def rank(self, ids: List[str], distances: List[float], metadatas: List[dict], documents: List[str]) -> List[RetrievedChunk]:
        chunks = []
        for rank, (chunk_id, dist, meta, text) in enumerate(zip(ids, distances, metadatas, documents)):
            # Distance is 1 - cosine_similarity for cosine space in ChromaDB
            score = round(1.0 - dist, 4)
            
            # Extract fields safely
            source = meta.get("source", "Unknown")
            doc_type = meta.get("document_type", "Unknown")
            heading = meta.get("heading", "No Heading")
            
            chunks.append(RetrievedChunk(
                rank=rank + 1,
                score=score,
                chunk_id=chunk_id,
                document=source,
                source="pdf" if source.endswith(".pdf") else "web",
                document_type=doc_type,
                heading=heading,
                metadata=meta,
                text=text
            ))
            
        # We assume they are already sorted by distance from ChromaDB, so rank is correct
        return chunks
