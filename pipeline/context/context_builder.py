import logging
from typing import Dict, Any, List

from pipeline.retrieval.search_models import RetrievalResult
from pipeline.context.models import ContextPackage, ContextBlock
from pipeline.context.ordering import OrderManager
from pipeline.context.merger import ChunkMerger
from pipeline.context.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)

class ContextBuilder:
    def __init__(self, max_tokens: int = 4000):
        self.order_manager = OrderManager()
        self.budget_manager = TokenBudgetManager(max_tokens=max_tokens)
        
    def build_context(self, result: RetrievalResult) -> ContextPackage:
        logger.info("Building Context Package...")
        
        # 1. Group and sort by document
        grouped_chunks = self.order_manager.group_and_sort(result.ranked_chunks)
        
        all_blocks = []
        total_merged = 0
        total_duplicates = 0
        
        # 2. Merge adjacent chunks and remove duplicates within each document
        for doc, chunks in grouped_chunks.items():
            merger = ChunkMerger()
            doc_blocks = merger.process_document_group(chunks)
            all_blocks.extend(doc_blocks)
            total_merged += merger.merged_count
            total_duplicates += merger.duplicate_count
            
        # 3. Enforce token budget
        final_blocks = self.budget_manager.enforce_budget(all_blocks)
        
        total_tokens = sum(b.token_estimate for b in final_blocks)
        documents = list(set(b.document for b in final_blocks))
        
        stats = {
            "initial_chunks": result.returned_count,
            "final_blocks": len(final_blocks),
            "documents_included": len(documents)
        }
        
        package = ContextPackage(
            query=result.query,
            documents=documents,
            context_blocks=final_blocks,
            token_count=total_tokens,
            merged_blocks=total_merged,
            removed_duplicates=total_duplicates,
            removed_blocks=self.budget_manager.removed_blocks,
            statistics=stats
        )
        
        return package
