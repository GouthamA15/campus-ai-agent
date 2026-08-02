import logging
from pipeline.prompt.models import PromptPackage

logger = logging.getLogger(__name__)

class PromptValidator:
    def validate(self, package: PromptPackage, max_tokens: int = 8000) -> bool:
        logger.info("Validating Prompt Package...")
        
        if not package.system_prompt:
            logger.error("Validation failed: System prompt missing.")
            return False
            
        if not package.user_question:
            logger.error("Validation failed: User question missing.")
            return False
            
        if not package.context or package.context == "No context available.":
            logger.error("Validation failed: Context is empty.")
            return False
            
        blocks = package.context.split("----------------------------------\n")
        blocks = [b.strip() for b in blocks if b.strip()]
        
        if len(blocks) != len(set(blocks)):
            logger.error("Validation failed: Duplicate context blocks detected.")
            return False
            
        if package.estimated_tokens > max_tokens:
            logger.error(f"Validation failed: Estimated tokens ({package.estimated_tokens}) exceed maximum limit ({max_tokens}).")
            return False
            
        logger.info("Validation: PASS")
        return True
