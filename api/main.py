"""FastAPI backend for the Interactive Campus Info AI Agent.

This version uses the fully implemented RAG pipeline.
"""

from typing import Any, Dict, List

import os
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables with override before importing internal modules
load_dotenv(override=True)

from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.retrieval.query_understanding import QueryUnderstandingEngine
from pipeline.retrieval.knowledge_retriever import KnowledgeRetriever
from pipeline.context.context_builder import ContextBuilder
from pipeline.prompt.prompt_builder import PromptBuilder
from pipeline.llm.llm_engine import LLMEngine
from pipeline.llm.response_processor import ResponseProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)



class AskRequest(BaseModel):
    question: str

class CitationModel(BaseModel):
    source: str
    document_id: str
    text_snippet: str

class AskResponse(BaseModel):
    answer: str
    markdown: str
    confidence_score: float
    citations: List[CitationModel]
    metadata: Dict[str, Any]

app = FastAPI(title="Interactive Campus Info AI Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline state
pipeline_state = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline components on startup."""
    try:
        manager = ChromaManager(persist_dir="data/chroma", collection_name="kucet_knowledge_base")
        pipeline_state["query_engine"] = QueryUnderstandingEngine()
        pipeline_state["retriever"] = KnowledgeRetriever(manager)
        pipeline_state["context_builder"] = ContextBuilder(max_tokens=4000)
        pipeline_state["prompt_builder"] = PromptBuilder(token_limit=7000)
        pipeline_state["llm_engine"] = LLMEngine()
        pipeline_state["response_processor"] = ResponseProcessor()
        logger.info("RAG pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if "query_engine" not in pipeline_state:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        # 1. Query Understanding
        structured_query = pipeline_state["query_engine"].analyze(request.question)
        
        # 2. Knowledge Retrieval
        retrieval_result = pipeline_state["retriever"].retrieve(structured_query, top_k=10)
        
        # 3. Context Building
        context_package = pipeline_state["context_builder"].build_context(retrieval_result)
        
        # 4. Prompt Building
        prompt_package = pipeline_state["prompt_builder"].build_prompt(context_package)
        
        # 5. LLM Engine
        llm_response = pipeline_state["llm_engine"].process(prompt_package)
        
        # 6. Response Processing
        final_resp = pipeline_state["response_processor"].process(llm_response, retrieval_result)
        
        return AskResponse(
            answer=final_resp.answer,
            markdown=final_resp.markdown,
            confidence_score=final_resp.confidence_score,
            citations=[
                CitationModel(source=c.source, document_id=c.document_id, text_snippet=c.text_snippet)
                for c in final_resp.citations
            ],
            metadata=final_resp.metadata
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "pipeline_initialized": str("query_engine" in pipeline_state).lower()}
