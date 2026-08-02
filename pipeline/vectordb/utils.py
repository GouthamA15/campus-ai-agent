import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_embeddings_json(filepath: Path) -> List[Dict[str, Any]]:
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data
