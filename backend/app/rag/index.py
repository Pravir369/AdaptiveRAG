"""Persistent cosine-similarity index with NumPy; persist to backend/data/index/ as .npz + JSON."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from app.core import INDEX_DIR, EMBED_DIM

logger = logging.getLogger(__name__)

INDEX_NPZ = INDEX_DIR / "vectors.npz"
INDEX_META = INDEX_DIR / "metadata.json"


class VectorIndex:
    """Persistent cosine index: add(embeddings, chunk_ids), search(query_embedding, top_k)."""

    def __init__(self):
        self._matrix: np.ndarray = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self._chunk_ids: List[str] = []
        self._load()

    def _load(self):
        if INDEX_NPZ.exists() and INDEX_META.exists():
            try:
                data = np.load(INDEX_NPZ)
                self._matrix = data["embeddings"]
                with open(INDEX_META, "r") as f:
                    meta = json.load(f)
                self._chunk_ids = meta.get("chunk_ids", [])
                if len(self._chunk_ids) != len(self._matrix):
                    self._matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
                    self._chunk_ids = []
                    logger.warning("Index metadata length mismatch; starting empty")
            except Exception as e:
                logger.warning("Could not load index: %s", e)
                self._matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
                self._chunk_ids = []

    def add(self, embeddings: np.ndarray, chunk_ids: List[str]):
        if len(embeddings) != len(chunk_ids):
            raise ValueError("embeddings and chunk_ids length mismatch")
        if len(embeddings) == 0:
            return
        # L2 normalize rows
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float32)
        self._matrix = np.vstack([self._matrix, embeddings]) if len(self._matrix) else embeddings
        self._chunk_ids.extend(chunk_ids)
        self._persist()

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if len(self._matrix) == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return []
        q = q / q_norm
        scores = np.dot(self._matrix, q.T).flatten()
        idx = np.argsort(-scores)[:top_k]
        return [
            {"chunk_id": self._chunk_ids[i], "score": float(scores[i])}
            for i in idx
        ]

    def _persist(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(INDEX_NPZ, embeddings=self._matrix)
        with open(INDEX_META, "w") as f:
            json.dump({"chunk_ids": self._chunk_ids}, f)
        logger.debug("Index persisted: %d vectors", len(self._chunk_ids))

    def chunk_ids(self) -> List[str]:
        return list(self._chunk_ids)


_index: VectorIndex | None = None


def get_vector_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index
