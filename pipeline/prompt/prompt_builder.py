import logging
from typing import Dict, Any, List

from pipeline.context.models import ContextPackage, ContextBlock
from pipeline.prompt.models import PromptPackage
from pipeline.prompt.prompt_templates import SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self, token_limit: int = 7000):
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE
        self.chars_per_token = 4
        self.token_limit = token_limit
        
    def build_prompt(self, package: ContextPackage) -> PromptPackage:
        logger.info("Building Prompt Package...")
        
        context_str = self._format_context(package.context_blocks)
        user_question = package.query
        
        estimated_tokens = self._estimate_tokens(self.system_prompt, context_str, user_question)
        
        if estimated_tokens > self.token_limit:
            logger.warning(f"Estimated token count ({estimated_tokens}) is approaching the safety limit of {self.token_limit}.")
            
        stats = {
            "total_blocks": len(package.context_blocks),
            "documents_referenced": len(package.documents),
            "estimated_tokens": estimated_tokens
        }
        
        return PromptPackage(
            system_prompt=self.system_prompt,
            context=context_str,
            user_question=user_question,
            estimated_tokens=estimated_tokens,
            context_documents=package.documents,
            statistics=stats
        )
        
    def _format_context(self, blocks: List[ContextBlock]) -> str:
        sections = []
        for block in blocks:
            section = "----------------------------------\n"
            section += f"Document: {block.document}\n"
            title = block.metadata.get("title", "")
            if title:
                section += f"Title: {title}\n"
            section += f"Heading: {block.heading}\n"
            if block.pages:
                section += f"Pages: {block.pages}\n"
            section += f"Source: {block.source.upper()}\n"
            section += f"Content:\n{block.text}\n"
            sections.append(section)
            
        if not sections:
            return "No context available."
            
        return "\n".join(sections) + "\n----------------------------------\n"
        
    def _estimate_tokens(self, sys_prompt: str, context: str, user_q: str) -> int:
        total_len = len(sys_prompt) + len(context) + len(user_q)
        # Add basic overhead for LLM framing
        return (total_len // self.chars_per_token) + 20
