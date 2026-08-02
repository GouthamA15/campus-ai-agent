import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from pipeline.embeddings.models import EmbeddingDocument
from pipeline.embeddings.utils import load_json_chunks
from pipeline.embeddings.embedding_validator import EmbeddingValidator
from pipeline.embeddings.retrieval_tester import RetrievalTester

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class UniversalEmbeddingGenerator:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", output_dir: str = "data/embeddings"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.validator = EmbeddingValidator(expected_dim=384)
        
    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def scan_sources(self, pdf_dir: str = "data/pdf_chunks", web_dir: str = "data/web_chunks") -> Dict[str, List[Path]]:
        pdf_path = Path(pdf_dir)
        web_path = Path(web_dir)
        
        pdf_files = list(pdf_path.glob("*.json")) if pdf_path.exists() else []
        web_files = list(web_path.glob("*.json")) if web_path.exists() else []
        
        return {
            "pdf": pdf_files,
            "web": web_files
        }

    def generate_embeddings(self, docs: List[EmbeddingDocument]) -> List[EmbeddingDocument]:
        """Generate embeddings for a list of valid documents."""
        self._load_model()
        
        texts = [doc.text for doc in docs]
        
        # In bge-small-en-v1.5, standard embedding is enough for documents
        # (For queries, we often add "Represent this sentence for searching relevant passages: ")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb.tolist()
            
        return docs

    def process_files(self, file_paths: List[Path], save_prefix: str = "batch") -> List[EmbeddingDocument]:
        """Process a list of chunk JSON files."""
        all_docs = []
        for fp in file_paths:
            try:
                docs = load_json_chunks(fp)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {fp.name}: {e}")
                
        # Pre-validate
        valid_docs = self.validator.pre_validate(all_docs)
        
        # Generate
        embedded_docs = self.generate_embeddings(valid_docs)
        
        # Post-validate
        final_valid_docs = self.validator.post_validate(embedded_docs)
        
        # Save output
        out_path = self.output_dir / f"{save_prefix}_embeddings.json"
        
        out_data = [doc.to_dict() for doc in final_valid_docs]
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False)
            
        logger.info(f"Saved {len(embedded_docs)} embeddings to {out_path.name}")
        return embedded_docs

    def run_pipeline(self, pdf_dir: str = "data/pdf_chunks", web_dir: str = "data/web_chunks", sample_mode: bool = True):
        logger.info("====================================")
        logger.info("Embedding Generator")
        logger.info("====================================")
        logger.info(f"Embedding Model: {self.model_name}")
        logger.info(f"Dimension: {self.validator.expected_dim}")
        logger.info("------------------------------------")
        
        sources = self.scan_sources(pdf_dir, web_dir)
        
        # Gather stats
        total_pdf_files = len(sources['pdf'])
        total_web_files = len(sources['web'])
        
        # To get chunks count we'd need to parse all, but we can do it lazily or print approx
        logger.info(f"PDF Documents: {total_pdf_files}")
        logger.info(f"WEB Documents: {total_web_files}")
        logger.info("------------------------------------")
        
        if sample_mode:
            logger.info("Running Sample Validation...")
            # Pick samples
            sample_files = []
            
            # Find Syllabus, Rules, Notification from PDF
            for f in sources['pdf']:
                name = f.name.lower()
                if "rules" in name or "btech_year1" in name or "notification" in name:
                    sample_files.append(f)
                    
            # Find Admissions, About, Faculty from Web
            for f in sources['web']:
                name = f.name.lower()
                if "admissions" in name or "about" in name or "faculty" in name or "college" in name:
                    sample_files.append(f)
                    
            # Deduplicate just in case
            sample_files = list(set(sample_files))
            
            # Keep it small
            sample_files = sample_files[:6]
            
            logger.info(f"Selected {len(sample_files)} sample files.")
            self.process_files(sample_files, save_prefix="sample")
            
            self.validator.save_report(self.output_dir / "embedding_validation_report.json")
            if self.validator.report["status"] == "PASS":
                logger.info("Validation: PASS")
                logger.info("Running Sample Retrieval Test...")
                tester = RetrievalTester(model_name=self.model_name, embeddings_path=str(self.output_dir / "sample_embeddings.json"))
                tester.run_tests()
            else:
                logger.error("Validation: FAIL. Check embedding_validation_report.json")
                
        else:
            logger.info("Embedding Entire Corpus...")
            all_files = sources['pdf'] + sources['web']
            self.process_files(all_files, save_prefix="full_corpus")
            
            self.validator.save_report(self.output_dir / "embedding_validation_report.json")
            if self.validator.report["status"] == "PASS":
                logger.info("Validation: PASS")
            else:
                logger.error("Validation: FAIL. Check embedding_validation_report.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run in sample mode")
    parser.add_argument("--full", action="store_true", help="Run full corpus")
    args = parser.parse_args()
    
    gen = UniversalEmbeddingGenerator()
    if args.full:
        gen.run_pipeline(sample_mode=False)
    else:
        gen.run_pipeline(sample_mode=True)
