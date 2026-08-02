import json
import logging
from pathlib import Path

from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.retrieval.query_understanding import QueryUnderstandingEngine
from pipeline.retrieval.knowledge_retriever import KnowledgeRetriever
from pipeline.context.context_builder import ContextBuilder
from pipeline.context.context_validator import ContextValidator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=======================================")
    logger.info("Context Builder")
    logger.info("=======================================")
    
    manager = ChromaManager(persist_dir="data/chroma", collection_name="kucet_knowledge_base")
    engine = QueryUnderstandingEngine()
    retriever = KnowledgeRetriever(manager)
    context_builder = ContextBuilder(max_tokens=4000)
    validator = ContextValidator()
    
    queries = [
        "What is the attendance requirement?",
        "Show Data Science syllabus.",
        "List scholarship details.",
        "How many seats are available in Data Science?",
        "Where is the Mechanical Engineering department?"
    ]
    
    report = []
    
    for q in queries:
        structured_query = engine.analyze(q)
        # Fetch more chunks for testing the merger logic
        retrieval_result = retriever.retrieve(structured_query, top_k=20, initial_candidates=40)
        
        package = context_builder.build_context(retrieval_result)
        is_valid = validator.validate(package)
        
        logger.info(f"Retrieved Chunks\n{retrieval_result.returned_count}")
        logger.info("---------------------------------------")
        logger.info(f"Merged Blocks\n{package.merged_blocks}")
        logger.info("---------------------------------------")
        logger.info(f"Duplicates Removed\n{package.removed_duplicates}")
        logger.info("---------------------------------------")
        logger.info(f"Documents\n{len(package.documents)}")
        logger.info("---------------------------------------")
        logger.info(f"Final Context Blocks\n{len(package.context_blocks)}")
        logger.info("---------------------------------------")
        logger.info(f"Token Count\n{package.token_count}")
        logger.info("---------------------------------------")
        logger.info(f"Validation\n{'PASS' if is_valid else 'FAIL'}")
        logger.info("=======================================")
        
        report.append(package.to_dict())
        
    out_path = Path("context_validation_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Saved validation report to {out_path}")

if __name__ == "__main__":
    main()
