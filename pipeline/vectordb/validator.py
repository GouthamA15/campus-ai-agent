import json
import logging
from pathlib import Path
from collections import Counter

from pipeline.vectordb.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

class DatabaseValidator:
    def __init__(self, manager: ChromaManager):
        self.collection = manager.get_collection()
        self.report = {
            "total_vectors": 0,
            "duplicate_ids": 0,
            "missing_text": 0,
            "missing_embeddings": 0,
            "incorrect_dimension": 0,
            "missing_metadata": 0,
            "status": "PENDING"
        }
        
    def validate(self, output_path: Path):
        logger.info("Running Database Validator...")
        
        # Get everything
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        embeddings = data.get("embeddings", [])
        
        self.report["total_vectors"] = len(ids)
        
        # Duplicate IDs
        id_counts = Counter(ids)
        self.report["duplicate_ids"] = sum(1 for count in id_counts.values() if count > 1)
        
        # Missing text
        self.report["missing_text"] = sum(1 for d in documents if not d or not d.strip())
        
        # Metadata integrity
        self.report["missing_metadata"] = sum(1 for m in metadatas if not m)
        
        # Embeddings
        if embeddings is None:
            self.report["missing_embeddings"] = len(ids)
        else:
            self.report["missing_embeddings"] = sum(1 for e in embeddings if e is None or len(e) == 0)
            self.report["incorrect_dimension"] = sum(1 for e in embeddings if e is not None and len(e) != 384)
            
        all_passed = (
            self.report["total_vectors"] > 0 and
            self.report["duplicate_ids"] == 0 and
            self.report["missing_text"] == 0 and
            self.report["missing_embeddings"] == 0 and
            self.report["incorrect_dimension"] == 0 and
            self.report["missing_metadata"] == 0
        )
        
        self.report["status"] = "PASS" if all_passed else "FAIL"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2)
            
        logger.info(f"Validation Report saved to {output_path.name} | Status: {self.report['status']}")
        return all_passed
