import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class ChromaManager:
    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "kucet_knowledge_base"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        logger.info(f"Connecting to ChromaDB at {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # We use cosine similarity to match what we tested earlier
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def get_collection(self):
        return self.collection
