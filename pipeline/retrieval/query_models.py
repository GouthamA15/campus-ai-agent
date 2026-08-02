from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class StructuredQuery:
    original_query: str
    normalized_query: str
    query_type: str
    intent: str
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    possible_document_types: List[str] = field(default_factory=list)
    possible_sources: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "query_type": self.query_type,
            "intent": self.intent,
            "entities": self.entities,
            "keywords": self.keywords,
            "possible_document_types": self.possible_document_types,
            "possible_sources": self.possible_sources,
            "metadata_filters": self.metadata_filters,
            "confidence": self.confidence
        }
