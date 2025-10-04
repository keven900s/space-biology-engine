import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    CSV_FILE = DATA_DIR / "SB_publication_PMC.csv"
    
    # Create directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model settings
    SUMMARIZER_MODEL = "facebook/bart-large-cnn"
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    NER_MODEL = "en_core_sci_md"  # Scientific NER from spaCy
    
    # Processing settings
    MAX_PAPERS_TO_PROCESS = 608  # All papers
    CHUNK_SIZE = 50  # Process in batches
    CACHE_ENABLED = True
    
    # Search settings
    TOP_K_RESULTS = 10
    SIMILARITY_THRESHOLD = 0.5
