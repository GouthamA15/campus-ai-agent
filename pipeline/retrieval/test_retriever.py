import json
import logging
from pathlib import Path

from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.retrieval.query_understanding import QueryUnderstandingEngine
from pipeline.retrieval.knowledge_retriever import KnowledgeRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("======================================")
    logger.info("Knowledge Retriever Validation")
    logger.info("======================================")
    
    manager = ChromaManager(persist_dir="data/chroma", collection_name="kucet_knowledge_base")
    engine = QueryUnderstandingEngine()
    retriever = KnowledgeRetriever(manager)
    
    queries = [
        "What is the attendance requirement?",
        "Show Data Science syllabus.",
        "List scholarship details.",
        "How many seats are available in Data Science?",
        "Where is the Mechanical Engineering department?"
    ]
    
    report = []
    
    for q in queries:
        logger.info(f"Query: {q}")
        
        # 1. Understand Query
        structured_query = engine.analyze(q)
        
        # 2. Retrieve
        result = retriever.retrieve(structured_query, top_k=8, initial_candidates=20)
        
        logger.info(f"Initial Candidates: {result.candidate_count}")
        logger.info(f"Returned: {result.returned_count}")
        logger.info(f"Fallback Used: {result.fallback_used}")
        
        if result.returned_count > 0:
            best_match = result.ranked_chunks[0]
            logger.info(f"Best Match Document: {best_match.document}")
            logger.info(f"Best Match Score: {best_match.score}")
        else:
            logger.warning("No matches found!")
            
        logger.info("--------------------------------------")
        report.append(result.to_dict())
        
    out_path = Path("retrieval_validation_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Saved validation report to {out_path}")
    logger.info("======================================")

if __name__ == "__main__":
    main()
