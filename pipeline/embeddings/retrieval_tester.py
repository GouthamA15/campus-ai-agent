import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RetrievalTester:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", embeddings_path: str = "data/embeddings/sample_embeddings.json"):
        self.model_name = model_name
        self.embeddings_path = Path(embeddings_path)
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model for retrieval: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def load_embeddings(self) -> List[Dict[str, Any]]:
        if not self.embeddings_path.exists():
            logger.error(f"Embeddings file not found: {self.embeddings_path}")
            return []
            
        with open(self.embeddings_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        logger.info(f"Loaded {len(docs)} embeddings from {self.embeddings_path.name}")
        return docs

    def run_tests(self):
        self._load_model()
        docs = self.load_embeddings()
        
        if not docs:
            return
            
        # Prepare vectors
        doc_vectors = np.array([doc["embedding"] for doc in docs])
        
        queries = [
            "What is the attendance requirement?",
            "What are B.Tech admissions?",
            "How can I apply for scholarship?",
            "Where is the college located?"
        ]
        
        # BGE models use this prompt for queries for retrieval
        query_prompt = "Represent this sentence for searching relevant passages: "
        formatted_queries = [query_prompt + q for q in queries]
        
        query_vectors = self.model.encode(formatted_queries, convert_to_numpy=True)
        
        report = {
            "model": self.model_name,
            "total_documents_searched": len(docs),
            "results": []
        }
        
        for q_idx, query in enumerate(queries):
            q_vec = query_vectors[q_idx]
            
            # Compute similarities
            similarities = [cosine_similarity(q_vec, d_vec) for d_vec in doc_vectors]
            
            # Get top 3
            top_k = 3
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            query_results = {
                "query": query,
                "top_k": []
            }
            
            for rank, idx in enumerate(top_indices):
                score = similarities[idx]
                doc = docs[idx]
                query_results["top_k"].append({
                    "rank": rank + 1,
                    "score": float(score),
                    "chunk_id": doc["chunk_id"],
                    "source": doc["source"],
                    "text_preview": doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"],
                    "metadata": doc["metadata"]
                })
                
            report["results"].append(query_results)
            
            # Log first result
            first = query_results["top_k"][0]
            logger.info(f"Q: '{query}' -> Top Match: [{first['source']}] (Score: {first['score']:.4f})")
            
        # Save report
        out_path = self.embeddings_path.parent / "sample_retrieval_report.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Saved retrieval report to {out_path}")

if __name__ == "__main__":
    tester = RetrievalTester()
    tester.run_tests()
