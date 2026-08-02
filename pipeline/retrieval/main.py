import argparse
import logging
import sys

from pipeline.retrieval.query_understanding import QueryUnderstandingEngine
from pipeline.retrieval.query_validator import QueryValidator
from pipeline.retrieval.utils import pretty_print_query

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(description="Test Query Understanding Engine")
    parser.add_argument("query", nargs="*", help="Natural language query to analyze")
    args = parser.parse_args()
    
    query = " ".join(args.query) if args.query else ""
    
    validator = QueryValidator()
    if not query:
        print("Please provide a query to test, e.g.:")
        print('python -m pipeline.retrieval.main "What is the attendance requirement?"')
        sys.exit(1)
        
    if not validator.validate(query):
        print("Invalid query provided.")
        sys.exit(1)
        
    engine = QueryUnderstandingEngine()
    result = engine.analyze(query)
    
    print(pretty_print_query(result.to_dict()))

if __name__ == "__main__":
    main()
