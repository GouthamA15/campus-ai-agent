import json
import logging
from pathlib import Path

from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.retrieval.query_understanding import QueryUnderstandingEngine
from pipeline.retrieval.knowledge_retriever import KnowledgeRetriever
from pipeline.context.context_builder import ContextBuilder
from pipeline.prompt.prompt_builder import PromptBuilder
from pipeline.llm.llm_engine import LLMEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    manager = ChromaManager(persist_dir="data/chroma", collection_name="kucet_knowledge_base")
    engine = QueryUnderstandingEngine()
    retriever = KnowledgeRetriever(manager)
    context_builder = ContextBuilder(max_tokens=4000)
    prompt_builder = PromptBuilder(token_limit=7000)
    llm_engine = LLMEngine()
    
    queries = [
        "What is the attendance requirement?",
        "Show Data Science syllabus.",
        "List scholarship details.",
        "How many seats are available in Data Science?",
        "Where is the Mechanical Engineering department?",
        "What are the eligibility criteria for B.Tech admission?",
        "What certificates can students request through the college?"
    ]
    
    out_dir = Path("debug")
    out_dir.mkdir(exist_ok=True)
    
    query_debug = {}
    retrieval_debug = {}
    context_debug = {}
    prompt_debug_text = ""
    llm_response_text = ""
    trace_report = []
    
    for i, q in enumerate(queries):
        logger.info(f"Processing Query {i+1}: {q}")
        
        structured_query = engine.analyze(q)
        query_debug[q] = structured_query.to_dict()
        
        retrieval_result = retriever.retrieve(structured_query, top_k=20, initial_candidates=40)
        retrieval_debug[q] = retrieval_result.to_dict()
        
        context_package = context_builder.build_context(retrieval_result)
        context_debug[q] = context_package.to_dict()
        
        prompt_package = prompt_builder.build_prompt(context_package)
        prompt_debug_text += f"\n\n{'='*50}\nQUERY: {q}\n{'='*50}\n"
        prompt_debug_text += prompt_package.system_prompt + "\n\n"
        prompt_debug_text += prompt_package.context + "\n\n"
        prompt_debug_text += prompt_package.user_question + "\n"
        
        # Mock LLM for the first run so we don't hit rate limits while debugging retrieval
        # Wait, the instruction says "Dump the EXACT prompt sent to the LLM... LLM Response Capture Raw response... etc"
        # I'll just run it. If it hits a rate limit, it will back off.
        llm_resp = llm_engine.process(prompt_package)
        llm_response_text += f"\n\n{'='*50}\nQUERY: {q}\n{'='*50}\n"
        llm_response_text += f"Status: {'SUCCESS' if llm_resp.success else 'FAIL'}\n"
        llm_response_text += f"Response:\n{llm_resp.response}\n"
        
        trace = {
            "query": q,
            "structured_query": structured_query.to_dict(),
            "retrieval_stats": {
                "count": retrieval_result.returned_count,
                "top_chunks": [{"id": c.chunk_id, "score": c.score, "doc": c.document} for c in retrieval_result.ranked_chunks[:5]]
            },
            "context_stats": context_package.statistics,
            "llm_stats": llm_resp.to_dict() if llm_resp else None
        }
        trace_report.append(trace)
        
    with open(out_dir / "query_debug.json", "w") as f:
        json.dump(query_debug, f, indent=2)
    with open(out_dir / "retrieval_debug.json", "w") as f:
        json.dump(retrieval_debug, f, indent=2)
    with open(out_dir / "context_debug.json", "w") as f:
        json.dump(context_debug, f, indent=2)
    with open(out_dir / "prompt_debug.txt", "w", encoding="utf-8") as f:
        f.write(prompt_debug_text)
    with open(out_dir / "llm_response.txt", "w", encoding="utf-8") as f:
        f.write(llm_response_text)
    with open(out_dir / "pipeline_trace_report.json", "w") as f:
        json.dump(trace_report, f, indent=2)

if __name__ == "__main__":
    main()
