"""Point the app at a throwaway data directory before anything imports it.

app.core reads ADAPTIVERAG_DATA_DIR at import time, so this must run first.
"""
import os
import sys
import tempfile
from pathlib import Path

_DATA_DIR = tempfile.mkdtemp(prefix="adaptiverag-test-")
os.environ["ADAPTIVERAG_DATA_DIR"] = _DATA_DIR
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["GENERATION_PROVIDER"] = "extractive"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
