"""Configuration management for vector_db."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATABASES_DIR = BASE_DIR / "databases"

# Embedding configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Search configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))


def get_db_path(db_name: str) -> Path:
    """Get the path for a specific database."""
    return DATABASES_DIR / db_name


def ensure_databases_dir():
    """Ensure the databases directory exists."""
    DATABASES_DIR.mkdir(exist_ok=True)
