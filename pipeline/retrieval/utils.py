import json
from typing import Dict, Any

def pretty_print_query(result_dict: Dict[str, Any]) -> str:
    lines = [
        "=====================================",
        "Query Understanding",
        "=====================================",
        "Original Query",
        result_dict.get("original_query", ""),
        "Normalized",
        result_dict.get("normalized_query", ""),
        "Intent",
        result_dict.get("intent", ""),
        "Entities",
        ", ".join(result_dict.get("entities", [])),
        "Document Hint",
        ", ".join(result_dict.get("possible_document_types", [])),
        "Source Hint",
        ", ".join(result_dict.get("possible_sources", [])),
        "Confidence",
        str(result_dict.get("confidence", 0.0)),
        "====================================="
    ]
    return "\n".join(lines)
