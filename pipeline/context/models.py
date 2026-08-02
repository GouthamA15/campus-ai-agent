from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ContextBlock:
    document: str
    heading: str
    pages: str
    source: str
    score: float
    text: str
    metadata: Dict[str, Any]
    
    token_estimate: int = 0
    chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "heading": self.heading,
            "pages": self.pages,
            "source": self.source,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
            "token_estimate": self.token_estimate
        }

@dataclass
class ContextPackage:
    query: str
    documents: List[str]
    context_blocks: List[ContextBlock]
    token_count: int
    merged_blocks: int
    removed_duplicates: int
    removed_blocks: int
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "documents": self.documents,
            "context_blocks": [b.to_dict() for b in self.context_blocks],
            "token_count": self.token_count,
            "merged_blocks": self.merged_blocks,
            "removed_duplicates": self.removed_duplicates,
            "removed_blocks": self.removed_blocks,
            "statistics": self.statistics
        }
