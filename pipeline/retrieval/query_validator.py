import string
import logging

logger = logging.getLogger(__name__)

class QueryValidator:
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length

    def validate(self, query: str) -> bool:
        if not query:
            logger.warning("Validation failed: Empty query")
            return False
            
        if not isinstance(query, str):
            logger.warning("Validation failed: Query is not a string")
            return False
            
        stripped = query.strip()
        if not stripped:
            logger.warning("Validation failed: Whitespace-only query")
            return False
            
        if len(query) > self.max_length:
            logger.warning(f"Validation failed: Query exceeds {self.max_length} characters")
            return False
            
        if all(char in string.punctuation for char in stripped):
            logger.warning("Validation failed: Query is only punctuation")
            return False
            
        return True
