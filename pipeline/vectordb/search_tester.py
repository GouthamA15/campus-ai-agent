import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from pipeline.vectordb.chroma_manager import ChromaManager

logger = logging.getLogger(__name__)

class SearchTester:
    def __init__(self, manager: ChromaManager, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.collection = manager.get_collection()
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model for search: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def test_search(self, output_path: Path):
        self._load_model()
        logger.info("Running Search Tests...")
        
        queries = [
            "What is the attendance requirement?",
            "Explain fee reimbursement.",
            "How many seats are available in Data Science?",
            "Where is the Mechanical Engineering department?",
            "List scholarship information."
        ]
        
        # Format queries for BGE model retrieval
        query_prompt = "Represent this sentence for searching relevant passages: "
        formatted_queries = [query_prompt + q for q in queries]
        
        query_vectors = self.model.encode(formatted_queries, convert_to_numpy=True).tolist()
        
        report = {
            "model": self.model_name,
            "results": []
        }
        
        # We can query all at once
        results = self.collection.query(
            query_embeddings=query_vectors,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        for q_idx, query in enumerate(queries):
            query_results = {
                "query": query,
                "top_k": []
            }
            
            # ChromaDB returns a list of lists for multiple queries
            doc_list = results["documents"][q_idx]
            meta_list = results["metadatas"][q_idx]
            dist_list = results["distances"][q_idx]
            id_list = results["ids"][q_idx]
            
            for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(id_list, doc_list, meta_list, dist_list)):
                # Chroma distances for cosine space are 1 - cosine_similarity. So lower is better, similarity = 1 - distance
                similarity = 1.0 - dist
                
                query_results["top_k"].append({
                    "rank": rank + 1,
                    "score": round(similarity, 4),
                    "chunk_id": doc_id,
                    "metadata": meta,
                    "text_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                })
                
            report["results"].append(query_results)
            
            first = query_results["top_k"][0]
            logger.info(f"Q: '{query}' -> Match: {first['chunk_id']} (Score: {first['score']})")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Search Test Report saved to {output_path.name}")
        return True
