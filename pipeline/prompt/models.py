from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class PromptPackage:
    system_prompt: str
    context: str
    user_question: str
    estimated_tokens: int
    context_documents: List[str]
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "context": self.context,
            "user_question": self.user_question,
            "estimated_tokens": self.estimated_tokens,
            "context_documents": self.context_documents,
            "statistics": self.statistics
        }
