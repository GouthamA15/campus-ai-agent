from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class EmbeddingDocument:
    """Universal chunk model for Embeddings."""
    chunk_id: str
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
