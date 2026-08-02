import logging
from typing import List, Dict, Any

from pipeline.vectordb.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

class CollectionBuilder:
    def __init__(self, manager: ChromaManager):
        self.collection = manager.get_collection()
        
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB only accepts str, int, float, bool as metadata values."""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = ", ".join(str(x) for x in v)
            else:
                clean[k] = str(v)
        return clean

    def build_collection(self, docs: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Safely inserts new embeddings in batches, skipping existing ones."""
        inserted_count = 0
        total_docs = len(docs)
        
        for i in range(0, total_docs, batch_size):
            batch = docs[i:i + batch_size]
            
            ids = [doc["chunk_id"] for doc in batch]
            
            # Check for existing IDs
            existing = self.collection.get(ids=ids)
            existing_ids = set(existing.get("ids", []))
            
            new_ids = []
            new_embeddings = []
            new_texts = []
            new_metadatas = []
            
            for doc in batch:
                cid = doc["chunk_id"]
                if cid not in existing_ids:
                    new_ids.append(cid)
                    new_embeddings.append(doc["embedding"])
                    new_texts.append(doc["text"])
                    
                    # Store source at root level of metadata for consistency
                    meta = doc.get("metadata", {}).copy()
                    meta["source"] = doc.get("source", "")
                    
                    new_metadatas.append(self._sanitize_metadata(meta))
            
            if new_ids:
                self.collection.add(
                    ids=new_ids,
                    embeddings=new_embeddings,
                    documents=new_texts,
                    metadatas=new_metadatas
                )
                inserted_count += len(new_ids)
                
            logger.info(f"Processed batch {i//batch_size + 1}, Inserted {len(new_ids)} new vectors.")
            
        return inserted_count
