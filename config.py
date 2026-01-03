import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
DB_DIR = DATA_DIR / "database"
MODEL_DIR = DATA_DIR / "models"

# Create directories
for dir_path in [PDF_DIR, DB_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Embedding model (local)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

# Precision settings
PRECISION_CONFIG = {
    "high": {"top_k": 3, "threshold": 0.8},
    "medium": {"top_k": 5, "threshold": 0.6},
    "low": {"top_k": 10, "threshold": 0.4}
}

# Local LLM settings
LOCAL_LLM_PATH = ""
LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_TEMPERATURE = 0.1
MAX_TOKENS = 1000
