from typing import Dict, Any

class MetadataFilterBuilder:
    def build(self, metadata_hints: Dict[str, str]) -> Dict[str, Any]:
        """Converts structured query metadata hints into a ChromaDB where clause."""
        if not metadata_hints:
            return {}
            
        conditions = []
        for key, value in metadata_hints.items():
            if value:
                if key == "source_type":
                    continue
                conditions.append({key: {"$eq": value}})
                
        if len(conditions) == 0:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
