import logging
from pipeline.prompt.models import PromptPackage
from pipeline.llm.models import LLMResponse

logger = logging.getLogger(__name__)

class LLMValidator:
    def validate(self, prompt: PromptPackage, response: LLMResponse, max_tokens: int = 8000) -> bool:
        logger.info("Validating LLM Output...")
        
        if not prompt or not prompt.context:
            logger.error("Validation failed: Prompt or context missing.")
            return False
            
        if prompt.estimated_tokens > max_tokens:
            logger.error(f"Validation failed: Prompt exceeds token budget ({prompt.estimated_tokens} > {max_tokens}).")
            return False
            
        if not response.success:
            logger.error(f"Validation failed: Provider returned error ({response.error}).")
            return False
            
        if not response.response or not response.response.strip():
            logger.error("Validation failed: Empty response from provider.")
            return False
            
        logger.info("Validation: PASS")
        return True
