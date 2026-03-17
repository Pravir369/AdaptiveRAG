"""Application configuration with env defaults."""
import os
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _BACKEND_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
DATA_DIR = Path(os.getenv("ADAPTIVERAG_DATA_DIR", str(_BACKEND_ROOT / "data")))
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "hash").lower()
EMBED_DIM = 768
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = 12

# Hybrid retrieval weights (default 0.7 vector, 0.3 lexical)
HYBRID_W_VEC = float(os.getenv("HYBRID_W_VEC", "0.7"))
HYBRID_W_LEX = float(os.getenv("HYBRID_W_LEX", "0.3"))

# Self-heal v1 thresholds (trigger healing when ANY fails)
MIN_MAX_FUSED = float(os.getenv("MIN_MAX_FUSED", "0.25"))
MIN_LEXICAL_OVERLAP = float(os.getenv("MIN_LEXICAL_OVERLAP", "0.15"))
MIN_SCORE_SEPARATION = float(os.getenv("MIN_SCORE_SEPARATION", "0.02"))  # min stddev of fused in topK
MIN_QUERY_SPECIFICITY = float(os.getenv("MIN_QUERY_SPECIFICITY", "0.25"))  # vague queries e.g. "explain this"
MAX_SELF_HEAL_ATTEMPTS = int(os.getenv("MAX_SELF_HEAL_ATTEMPTS", "3"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

# Generation: extractive (default) or ollama (local LLM)
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "extractive").lower().strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
DEBUG_STORE_FULL_PROMPT = os.getenv("DEBUG_STORE_FULL_PROMPT", "").lower() in ("1", "true", "yes")
