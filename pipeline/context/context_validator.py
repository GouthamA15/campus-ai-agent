import logging
from typing import List
from pipeline.context.models import ContextPackage

logger = logging.getLogger(__name__)

class ContextValidator:
    def validate(self, package: ContextPackage, max_tokens: int = 4000) -> bool:
        logger.info("Validating Context Package...")
        
        if not package.context_blocks:
            logger.warning("Validation failed: Empty context.")
            return False
            
        if package.token_count > max_tokens:
            logger.warning(f"Validation failed: Oversized context ({package.token_count} > {max_tokens}).")
            return False
            
        seen_texts = set()
        for i, block in enumerate(package.context_blocks):
            if not block.text.strip():
                logger.warning(f"Validation failed: Block {i} has missing text.")
                return False
                
            if not block.document:
                logger.warning(f"Validation failed: Block {i} is missing document metadata.")
                return False
                
            text_hash = block.text.strip().lower()
            if text_hash in seen_texts:
                logger.warning(f"Validation failed: Duplicate block text found in {block.document}.")
                return False
            seen_texts.add(text_hash)
            
        logger.info("Validation: PASS")
        return True
