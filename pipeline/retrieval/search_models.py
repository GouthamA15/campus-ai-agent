from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RetrievedChunk:
    rank: int
    score: float
    chunk_id: str
    document: str
    source: str
    document_type: str
    heading: str
    metadata: Dict[str, Any]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "document": self.document,
            "source": self.source,
            "document_type": self.document_type,
            "heading": self.heading,
            "metadata": self.metadata,
            "text": self.text
        }

@dataclass
class RetrievalResult:
    query: str
    retrieval_time_ms: int
    embedding_time_ms: int
    search_time_ms: int
    candidate_count: int
    returned_count: int
    ranked_chunks: List[RetrievedChunk]
    applied_filters: Dict[str, Any]
    fallback_used: bool
    query_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_time_ms": self.retrieval_time_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "search_time_ms": self.search_time_ms,
            "candidate_count": self.candidate_count,
            "returned_count": self.returned_count,
            "ranked_chunks": [c.to_dict() for c in self.ranked_chunks],
            "applied_filters": self.applied_filters,
            "fallback_used": self.fallback_used,
            "query_metadata": self.query_metadata
        }
