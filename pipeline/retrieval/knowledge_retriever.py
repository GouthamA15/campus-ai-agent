import time
import logging
from typing import Dict, Any

from sentence_transformers import SentenceTransformer
from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.retrieval.query_models import StructuredQuery
from pipeline.retrieval.search_models import RetrievalResult
from pipeline.retrieval.metadata_filter import MetadataFilterBuilder
from pipeline.retrieval.ranking import Ranker

logger = logging.getLogger(__name__)

class KnowledgeRetriever:
    def __init__(self, manager: ChromaManager, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.collection = manager.get_collection()
        self.model_name = model_name
        self.model = None
        self.filter_builder = MetadataFilterBuilder()
        self.ranker = Ranker()

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model for retrieval: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def retrieve(self, query: StructuredQuery, top_k: int = 8, initial_candidates: int = 20) -> RetrievalResult:
        start_time = time.time()
        
        # 1. Embed Query
        self._load_model()
        embed_start = time.time()
        # Format query for BGE retrieval
        query_text = "Represent this sentence for searching relevant passages: " + query.normalized_query
        query_vector = self.model.encode(query_text, convert_to_numpy=True).tolist()
        embed_time = int((time.time() - embed_start) * 1000)
        
        # 2. Build Metadata Filter
        where_clause = self.filter_builder.build(query.metadata_filters)
        
        search_start = time.time()
        fallback_used = False
        
        try:
            # First try with filters
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=initial_candidates,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # If nothing returned, fallback to no filter
            if not results["ids"][0]:
                fallback_used = True
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=initial_candidates,
                    include=["documents", "metadatas", "distances"]
                )
                
        except Exception as e:
            logger.warning(f"Metadata filter failed: {e}. Falling back to similarity-only.")
            fallback_used = True
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=initial_candidates,
                include=["documents", "metadatas", "distances"]
            )
            
        search_time = int((time.time() - search_start) * 1000)
        
        # 3. Ranking
        ids = results["ids"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0]
        docs = results["documents"][0]
        
        ranked_chunks = self.ranker.rank(ids, dists, metas, docs)
        
        # Keep top k
        final_chunks = ranked_chunks[:top_k]
        
        retrieval_time = int((time.time() - start_time) * 1000)
        
        return RetrievalResult(
            query=query.original_query,
            retrieval_time_ms=retrieval_time,
            embedding_time_ms=embed_time,
            search_time_ms=search_time,
            candidate_count=len(ranked_chunks),
            returned_count=len(final_chunks),
            ranked_chunks=final_chunks,
            applied_filters=where_clause,
            fallback_used=fallback_used,
            query_metadata=query.to_dict()
        )
