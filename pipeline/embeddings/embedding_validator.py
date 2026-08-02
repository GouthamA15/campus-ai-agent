import math
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

from pipeline.embeddings.models import EmbeddingDocument

class EmbeddingValidator:
    def __init__(self, expected_dim: int = 384):
        self.expected_dim = expected_dim
        self.report = {
            "pre_validation": {
                "total_input": 0,
                "passed": 0,
                "failed": 0,
                "errors": {}
            },
            "post_validation": {
                "total_input": 0,
                "passed": 0,
                "failed": 0,
                "errors": {}
            },
            "status": "PENDING"
        }
        self.seen_chunk_ids = set()
        self.seen_texts = set()
        self.seen_embeddings = [] # store tuples of embeddings to check duplicates, or maybe just hash them

    def _log_pre_error(self, doc: EmbeddingDocument, reason: str):
        if reason not in self.report["pre_validation"]["errors"]:
            self.report["pre_validation"]["errors"][reason] = []
        self.report["pre_validation"]["errors"][reason].append(doc.chunk_id)

    def _log_post_error(self, doc: EmbeddingDocument, reason: str):
        if reason not in self.report["post_validation"]["errors"]:
            self.report["post_validation"]["errors"][reason] = []
        self.report["post_validation"]["errors"][reason].append(doc.chunk_id)

    def pre_validate(self, docs: List[EmbeddingDocument]) -> List[EmbeddingDocument]:
        """Validate chunks before embedding. Skips invalid chunks."""
        valid_docs = []
        for doc in docs:
            self.report["pre_validation"]["total_input"] += 1
            is_valid = True
            
            if not doc.chunk_id:
                self._log_pre_error(doc, "Missing chunk_id")
                is_valid = False
            elif doc.chunk_id in self.seen_chunk_ids:
                self._log_pre_error(doc, "Duplicate chunk_id")
                is_valid = False
            
            if not doc.text:
                self._log_pre_error(doc, "Missing text")
                is_valid = False
            else:
                stripped_text = doc.text.strip()
                if not stripped_text:
                    self._log_pre_error(doc, "Whitespace-only chunks")
                    is_valid = False
                elif stripped_text in self.seen_texts:
                    self._log_pre_error(doc, "Duplicate text")
                    is_valid = False
                
            if not doc.metadata:
                self._log_pre_error(doc, "Missing metadata")
                is_valid = False
                
            if is_valid:
                valid_docs.append(doc)
                self.report["pre_validation"]["passed"] += 1
                self.seen_chunk_ids.add(doc.chunk_id)
                self.seen_texts.add(stripped_text)
            else:
                self.report["pre_validation"]["failed"] += 1
                
        return valid_docs

    def post_validate(self, docs: List[EmbeddingDocument]) -> List[EmbeddingDocument]:
        """Validate chunks after embedding."""
        all_passed = True
        seen_vecs = set()
        valid_docs = []
        
        for doc in docs:
            self.report["post_validation"]["total_input"] += 1
            is_valid = True
            
            if not doc.embedding:
                self._log_post_error(doc, "Missing embedding")
                is_valid = False
            else:
                if len(doc.embedding) != self.expected_dim:
                    self._log_post_error(doc, f"Incorrect vector dimension (got {len(doc.embedding)})")
                    is_valid = False
                    
                has_nan = any(math.isnan(v) for v in doc.embedding)
                has_inf = any(math.isinf(v) for v in doc.embedding)
                
                if has_nan:
                    self._log_post_error(doc, "Contains NaN values")
                    is_valid = False
                if has_inf:
                    self._log_post_error(doc, "Contains Inf values")
                    is_valid = False
                    
                if not has_nan and not has_inf:
                    norm = sum(v*v for v in doc.embedding) ** 0.5
                    if norm <= 0:
                        self._log_post_error(doc, "Vector norm <= 0")
                        is_valid = False
                
                # Check duplicates by converting to tuple
                vec_tuple = tuple(doc.embedding)
                if vec_tuple in seen_vecs:
                    self._log_post_error(doc, "Duplicate embedding vector")
                    is_valid = False
                else:
                    seen_vecs.add(vec_tuple)

            if not doc.metadata:
                self._log_post_error(doc, "Metadata not preserved")
                is_valid = False
                all_passed = False
            if not doc.text or not doc.text.strip():
                self._log_post_error(doc, "Text not preserved")
                is_valid = False
                all_passed = False
                
            if is_valid:
                self.report["post_validation"]["passed"] += 1
                valid_docs.append(doc)
            else:
                self.report["post_validation"]["failed"] += 1
                
        self.report["status"] = "PASS" if all_passed else "FAIL"
        return valid_docs

    def save_report(self, output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2)
