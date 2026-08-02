import logging
from pathlib import Path

from pipeline.vectordb.utils import load_embeddings_json
from pipeline.vectordb.chroma_manager import ChromaManager
from pipeline.vectordb.collection_builder import CollectionBuilder
from pipeline.vectordb.validator import DatabaseValidator
from pipeline.vectordb.search_tester import SearchTester

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("======================================")
    logger.info("ChromaDB Builder")
    logger.info("======================================")
    
    embeddings_file = Path("data/embeddings/full_corpus_embeddings.json")
    logger.info("Loading Embeddings...")
    
    docs = load_embeddings_json(embeddings_file)
    
    if not docs:
        logger.error("No embeddings found. Run Phase 3.1 first.")
        return
        
    pdf_count = sum(1 for d in docs if d.get("source", "").endswith(".pdf"))
    web_count = len(docs) - pdf_count
    
    logger.info(f"PDF Chunks: {pdf_count}")
    logger.info(f"WEB Chunks: {web_count}")
    logger.info("--------------------------------------")
    
    manager = ChromaManager(persist_dir="data/chroma", collection_name="kucet_knowledge_base")
    logger.info(f"Creating/Loading Collection: {manager.collection_name}")
    logger.info("--------------------------------------")
    
    builder = CollectionBuilder(manager)
    inserted = builder.build_collection(docs, batch_size=100)
    
    logger.info("--------------------------------------")
    logger.info(f"Inserted: {inserted} new vectors")
    logger.info("--------------------------------------")
    
    validator = DatabaseValidator(manager)
    val_passed = validator.validate(Path("data/chroma/chroma_validation_report.json"))
    
    logger.info(f"Validation: {'PASS' if val_passed else 'FAIL'}")
    logger.info("--------------------------------------")
    
    if val_passed:
        tester = SearchTester(manager)
        tester.test_search(Path("data/chroma/search_test_report.json"))
        logger.info("Running Search Tests... PASS")
    else:
        logger.error("Skipping Search Tests due to validation failure.")
        
    logger.info("--------------------------------------")
    logger.info(f"Database saved to {manager.persist_dir}")
    logger.info("======================================")

if __name__ == "__main__":
    main()
