from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pipeline.llm.models import LLMResponse
from pipeline.retrieval.search_models import RetrievalResult

@dataclass
class Citation:
    source: str
    document_id: str
    text_snippet: str

@dataclass
class FinalResponse:
    answer: str
    markdown: str
    confidence_score: float
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "markdown": self.markdown,
            "confidence_score": self.confidence_score,
            "citations": [
                {
                    "source": c.source,
                    "document_id": c.document_id,
                    "text_snippet": c.text_snippet
                } for c in self.citations
            ],
            "metadata": self.metadata
        }

class ResponseProcessor:
    def __init__(self):
        pass

    def _extract_citations(self, response_text: str, retrieval_result: RetrievalResult) -> List[Citation]:
        citations = []
        # If the LLM referenced any document IDs, we can pull them out.
        # But simpler: just use the top K retrieval results that were actually used.
        # The prompt might say [rules.pdf] or [http://kucet.ac.in/...]
        # We check if document_id is present in the response text.
        for chunk in retrieval_result.ranked_chunks:
            if f"[{chunk.document}]" in response_text or chunk.document in response_text:
                citations.append(Citation(
                    source=chunk.source,
                    document_id=chunk.document,
                    text_snippet=chunk.text[:150] + "..."
                ))
        
        # If no explicit citation found, assume the top 1 is the main source
        if not citations and retrieval_result.ranked_chunks:
            chunk = retrieval_result.ranked_chunks[0]
            citations.append(Citation(
                source=chunk.source,
                document_id=chunk.document,
                text_snippet=chunk.text[:150] + "..."
            ))
            
        return citations

    def _calculate_confidence(self, retrieval_result: RetrievalResult) -> float:
        if not retrieval_result.ranked_chunks:
            return 0.0
        # Average score of top 3
        scores = [c.score for c in retrieval_result.ranked_chunks[:3]]
        return sum(scores) / len(scores)

    def process(self, llm_response: LLMResponse, retrieval_result: RetrievalResult) -> FinalResponse:
        if not llm_response.success:
            return FinalResponse(
                answer="I'm sorry, I am currently unable to process your request due to system limits. Please try again later.",
                markdown="I'm sorry, I am currently unable to process your request due to system limits. Please try again later.",
                confidence_score=0.0,
                metadata={"error": llm_response.error}
            )

        citations = self._extract_citations(llm_response.response, retrieval_result)
        confidence = self._calculate_confidence(retrieval_result)
        
        # Attach citations to the markdown
        markdown = llm_response.response
        if citations:
            markdown += "\n\n### Sources\n"
            seen = set()
            for c in citations:
                if c.document_id not in seen:
                    markdown += f"- **{c.document_id}**\n"
                    seen.add(c.document_id)

        return FinalResponse(
            answer=llm_response.response,
            markdown=markdown,
            confidence_score=confidence,
            citations=citations,
            metadata={
                "tokens": llm_response.total_tokens,
                "latency_ms": llm_response.latency_ms,
                "provider": llm_response.provider,
                "model": llm_response.model
            }
        )
