import json
import logging
import os
from pathlib import Path

from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.retrieval.query_understanding import QueryUnderstandingEngine
from pipeline.retrieval.knowledge_retriever import KnowledgeRetriever
from pipeline.context.context_builder import ContextBuilder
from pipeline.prompt.prompt_builder import PromptBuilder
from pipeline.llm.llm_engine import LLMEngine
from pipeline.llm.validator import LLMValidator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("========================================")
    logger.info("LLM Engine Test")
    logger.info("========================================")
    
    manager = ChromaManager(persist_dir="data/chroma", collection_name="kucet_knowledge_base")
    engine = QueryUnderstandingEngine()
    retriever = KnowledgeRetriever(manager)
    context_builder = ContextBuilder(max_tokens=4000)
    prompt_builder = PromptBuilder(token_limit=7000)
    
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not found in environment. The LLM engine will gracefully return an error.")
        
    llm_engine = LLMEngine()
    validator = LLMValidator()
    
    queries = [
        "What is the attendance requirement?",
        "Show Data Science syllabus.",
        "List scholarship details.",
        "How many seats are available in Data Science?",
        "Where is the Mechanical Engineering department?"
    ]
    
    reports = []
    
    for q in queries:
        logger.info(f"Query: {q}")
        
        structured_query = engine.analyze(q)
        retrieval_result = retriever.retrieve(structured_query, top_k=20, initial_candidates=40)
        context_package = context_builder.build_context(retrieval_result)
        prompt_package = prompt_builder.build_prompt(context_package)
        
        llm_response = llm_engine.process(prompt_package)
        is_valid = validator.validate(prompt_package, llm_response)
        
        logger.info("----------------------------------------")
        if llm_response.success:
            logger.info(f"Response snippet: {llm_response.response[:150]}...")
        else:
            logger.info(f"Error: {llm_response.error}")
        logger.info("========================================")
        
        reports.append({
            "query": q,
            "response_stats": llm_response.to_dict(),
            "valid": is_valid
        })
        
    with open(Path("llm_validation_report.json"), 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2)
        
if __name__ == "__main__":
    main()
