from typing import List
from pipeline.context.models import ContextBlock

class TokenBudgetManager:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        # Standard approximation: 1 token = 4 characters
        self.chars_per_token = 4
        self.removed_blocks = 0
        
    def enforce_budget(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        total_tokens = 0
        
        for block in blocks:
            # Estimate metadata overhead + text
            overhead = len(block.heading) + len(block.document) + 50
            block.token_estimate = (len(block.text) + overhead) // self.chars_per_token
            total_tokens += block.token_estimate
            
        if total_tokens <= self.max_tokens:
            return blocks
            
        indexed_blocks = list(enumerate(blocks))
        indexed_blocks.sort(key=lambda x: x[1].score)
        
        keep_indices = set(range(len(blocks)))
        
        for idx, block in indexed_blocks:
            if total_tokens <= self.max_tokens:
                break
            keep_indices.remove(idx)
            total_tokens -= block.token_estimate
            self.removed_blocks += 1
            
        final_blocks = [blocks[i] for i in range(len(blocks)) if i in keep_indices]
        return final_blocks
