import json
import logging
from pathlib import Path

from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.retrieval.query_understanding import QueryUnderstandingEngine
from pipeline.retrieval.knowledge_retriever import KnowledgeRetriever
from pipeline.context.context_builder import ContextBuilder
from pipeline.prompt.prompt_builder import PromptBuilder
from pipeline.prompt.prompt_validator import PromptValidator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("========================================")
    logger.info("Prompt Builder")
    logger.info("========================================")
    
    manager = ChromaManager(persist_dir="data/chroma", collection_name="kucet_knowledge_base")
    engine = QueryUnderstandingEngine()
    retriever = KnowledgeRetriever(manager)
    context_builder = ContextBuilder(max_tokens=4000)
    prompt_builder = PromptBuilder(token_limit=7000)
    validator = PromptValidator()
    
    queries = [
        "What is the attendance requirement?",
        "Show Data Science syllabus.",
        "List scholarship details.",
        "How many seats are available in Data Science?",
        "Where is the Mechanical Engineering department?"
    ]
    
    reports = []
    stats_all = []
    
    for q in queries:
        structured_query = engine.analyze(q)
        retrieval_result = retriever.retrieve(structured_query, top_k=20, initial_candidates=40)
        context_package = context_builder.build_context(retrieval_result)
        
        prompt_package = prompt_builder.build_prompt(context_package)
        is_valid = validator.validate(prompt_package)
        
        logger.info(f"Context Blocks\n{prompt_package.statistics['total_blocks']}")
        logger.info("----------------------------------------")
        logger.info(f"Documents\n{prompt_package.statistics['documents_referenced']}")
        logger.info("----------------------------------------")
        logger.info(f"Estimated Tokens\n{prompt_package.estimated_tokens}")
        logger.info("----------------------------------------")
        logger.info(f"Prompt Validation\n{'PASS' if is_valid else 'FAIL'}")
        logger.info("========================================")
        
        reports.append(prompt_package.to_dict())
        stats_all.append({
            "query": q,
            "estimated_tokens": prompt_package.estimated_tokens,
            "documents_referenced": prompt_package.statistics['documents_referenced']
        })
        
    with open(Path("prompt_validation_report.json"), 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2)
        
    with open(Path("prompt_statistics.json"), 'w', encoding='utf-8') as f:
        json.dump(stats_all, f, indent=2)
        
if __name__ == "__main__":
    main()
