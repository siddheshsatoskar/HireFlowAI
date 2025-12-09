"""
Configuration settings for HireFlow application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Pinecone Configuration (optional)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")

# Application Configuration
APP_NAME = os.getenv("APP_NAME", "HireFlow")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Vector Database Configuration
VECTOR_DB = os.getenv("VECTOR_DB", "faiss")  # Options: faiss, pinecone
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# Search Configuration
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "10"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Model Configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

# Create directories if they don't exist
VECTOR_STORE_PATH.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

