import re
import string
import logging
from typing import List, Dict, Set, Tuple

from pipeline.retrieval.query_models import StructuredQuery

logger = logging.getLogger(__name__)

class QueryUnderstandingEngine:
    def __init__(self):
        # Abbreviation expansions
        self.abbreviations = {
            r"\bds\b": "data science",
            r"\bcse\b": "computer science engineering",
            r"\bece\b": "electronics and communication engineering",
            r"\beee\b": "electrical and electronics engineering",
            r"\bmech\b": "mechanical engineering",
            r"\bce\b": "civil engineering",
            r"\bcivil\b": "civil engineering",
            r"\bb\.?tech\b": "bachelor of technology",
            r"\bm\.?tech\b": "master of technology",
            r"\bcsd\b": "computer science and design"
        }
        
        # Entity matching dictionaries (lowercased for matching)
        self.entities_map = {
            "attendance": "Attendance",
            "scholarship": "Scholarship",
            "fee": "Fee",
            "reimbursement": "Fee",
            "admission": "Admission",
            "data science": "Data Science",
            "computer science": "Computer Science Engineering",
            "mechanical engineering": "Mechanical Engineering",
            "civil engineering": "Civil Engineering",
            "electrical and electronics": "Electrical and Electronics Engineering",
            "electronics and communication": "Electronics and Communication Engineering",
            "examination": "Examination",
            "exam": "Examination",
            "semester": "Semester",
            "faculty": "Faculty",
            "placement": "Placement",
            "hostel": "Hostel",
            "department": "Department",
            "principal": "Principal",
            "syllabus": "Syllabus"
        }
        
        # Keywords mapping to intents
        self.intent_patterns = {
            "procedure": [r"\bhow to\b", r"\bhow do i\b", r"\bapply\b", r"\bprocedure\b", r"\bsteps to\b"],
            "comparison": [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference between\b"],
            "lookup": [r"\bshow me\b", r"\bfind\b", r"\bwhere is\b", r"\bwho is\b", r"\bwhat is the\b", r"\blist\b"],
            "information": [r"\bexplain\b", r"\bwhat is\b", r"\bdetails about\b", r"\babout\b"]
        }
        
        # Keywords mapping to query types
        self.query_type_patterns = {
            "rules": [r"attendance", r"rule", r"regulation", r"requirement", r"mandatory"],
            "location": [r"where is", r"location", r"address"],
            "contact": [r"who is", r"contact", r"email", r"phone", r"principal"],
            "statistics": [r"how many", r"count", r"total", r"number of"],
            "eligibility": [r"eligible", r"eligibility", r"qualify", r"criteria"],
            "procedure": [r"how to", r"apply", r"process", r"procedure"],
            "list": [r"list", r"show all", r"names of"],
            "comparison": [r"compare", r"vs", r"difference"],
            "lookup": [r"show me", r"find", r"search"],
            "definition": [r"what is", r"define", r"explain"]
        }
        
        # Hints mapping
        self.document_type_hints = {
            "attendance": ["rules"],
            "rule": ["rules"],
            "syllabus": ["syllabus"],
            "admission": ["notification", "webpage"],
            "fee": ["notification", "rules", "quotation"],
            "exam": ["notification"],
            "faculty": ["department"],
            "department": ["department"]
        }
        
        # Source hints
        self.source_hints = {
            "notification": "pdf",
            "syllabus": "pdf",
            "rules": "pdf",
            "faculty": "web",
            "department": "web",
            "webpage": "web"
        }

    def normalize_query(self, query: str) -> str:
        # Lowercase
        normalized = query.lower()
        
        # Normalize punctuation (replace with spaces, keep alphanumeric)
        # Wait, the prompt says "normalize punctuation" - we might just want to remove trailing punctuation
        # and standardize spaces. We shouldn't remove all punctuation if they contain important symbols.
        normalized = re.sub(r'([^\w\s\.-])', r' \1 ', normalized)
        
        # Expand abbreviations
        for pattern, expansion in self.abbreviations.items():
            normalized = re.sub(pattern, expansion, normalized)
            
        # Remove duplicate whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove trailing punctuation (often from question marks)
        normalized = re.sub(r'^[\W_]+|[\W_]+$', '', normalized).strip()
        
        return normalized

    def extract_entities(self, normalized_query: str) -> List[str]:
        entities = set()
        
        # 1. Dictionary-based extraction
        for key, entity_name in self.entities_map.items():
            if re.search(r'\b' + re.escape(key) + r'\b', normalized_query):
                entities.add(entity_name)
                
        # 2. Pattern-based extraction (e.g., semesters)
        if re.search(r'semester\s+\d', normalized_query) or re.search(r'sem\s+\d', normalized_query):
            entities.add("Semester")
            
        return list(entities)
        
    def extract_keywords(self, normalized_query: str) -> List[str]:
        # Filter out common stop words and return meaningful words
        stopwords = {"what", "is", "the", "a", "an", "of", "in", "to", "for", "with", "on", "at", "by", "from", "how", "where", "who", "when", "why", "show", "me", "explain", "list", "about", "details", "many", "are", "do", "i", "can", "you", "tell"}
        words = normalized_query.split()
        keywords = [w for w in words if w.isalnum() and w not in stopwords]
        return keywords

    def detect_intent(self, normalized_query: str) -> str:
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized_query):
                    return intent
        return "information" # Default intent

    def detect_query_type(self, normalized_query: str) -> str:
        for qtype, patterns in self.query_type_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', normalized_query):
                    return qtype
        return "general" # Default type

    def get_document_hints(self, normalized_query: str, entities: List[str]) -> List[str]:
        hints = set()
        for key, doc_types in self.document_type_hints.items():
            if re.search(r'\b' + key + r'\b', normalized_query):
                hints.update(doc_types)
                
        # Entity-based mapping
        if "Syllabus" in entities:
            hints.add("syllabus")
        if "Attendance" in entities:
            hints.add("rules")
            
        return list(hints)

    def get_source_hints(self, doc_hints: List[str]) -> List[str]:
        sources = set()
        for hint in doc_hints:
            if hint in self.source_hints:
                sources.add(self.source_hints[hint])
                
        if len(sources) == 1:
            return list(sources)
        elif len(sources) > 1:
            return ["either"]
        return ["either"]

    def generate_metadata_filters(self, entities: List[str], doc_hints: List[str], sources: List[str], normalized_query: str) -> Dict[str, str]:
        filters = {}
        
        if len(doc_hints) == 1:
            filters["document_type"] = doc_hints[0]
            
        if len(sources) == 1 and sources[0] != "either":
            filters["source_type"] = sources[0] # assuming source_type maps to "pdf" or "web" later. Or maybe we don't strictly filter yet.
            
        # Optional: Department hints
        departments = ["Data Science", "Computer Science Engineering", "Mechanical Engineering", "Civil Engineering", "Electrical and Electronics Engineering", "Electronics and Communication Engineering"]
        for dept in departments:
            if dept in entities:
                filters["department"] = dept
                break
                
        return filters

    def calculate_confidence(self, entities: List[str], keywords: List[str], intent: str, qtype: str) -> float:
        score = 0.5 # Base confidence
        if entities:
            score += 0.2
        if keywords:
            score += 0.1
        if intent != "information":
            score += 0.1
        if qtype != "general":
            score += 0.1
            
        return min(1.0, round(score, 2))

    def analyze(self, query: str) -> StructuredQuery:
        normalized = self.normalize_query(query)
        qtype = self.detect_query_type(normalized)
        intent = self.detect_intent(normalized)
        entities = self.extract_entities(normalized)
        keywords = self.extract_keywords(normalized)
        
        doc_hints = self.get_document_hints(normalized, entities)
        sources = self.get_source_hints(doc_hints)
        
        filters = self.generate_metadata_filters(entities, doc_hints, sources, normalized)
        confidence = self.calculate_confidence(entities, keywords, intent, qtype)
        
        return StructuredQuery(
            original_query=query,
            normalized_query=normalized,
            query_type=qtype,
            intent=intent,
            entities=entities,
            keywords=keywords,
            possible_document_types=doc_hints,
            possible_sources=sources,
            metadata_filters=filters,
            confidence=confidence
        )
