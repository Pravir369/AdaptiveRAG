"""Embedding abstraction: embed_texts(list[str]) -> ndarray, embed_query(str) -> ndarray."""
import logging
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.core import EMBEDDING_PROVIDER, EMBED_DIM

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic signed hashing into dim dimensions, L2 normalized."""
    import hashlib
    vec = np.zeros(dim, dtype=np.float32)
    # Tokenize loosely: words + small n-grams
    tokens = text.lower().replace("\n", " ").split()
    # SHA256 hexdigest has 64 chars (32 bytes). Use first 32 for idx, second 32 for sign.
    for i, t in enumerate(tokens):
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        for j in range(0, min(dim, 32)):
            idx = (int(h[j * 2 : j * 2 + 2], 16) + i + j) % dim
            sign_idx = (j + 16) % 32  # keep within 0..31 so slice stays in h[0:64]
            sign = 1 if int(h[sign_idx * 2 : sign_idx * 2 + 2], 16) % 2 == 0 else -1
            vec[idx] += sign * (1.0 / (1 + len(t)))
    # Bigram contributions
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        h = hashlib.sha256(bigram.encode("utf-8")).hexdigest()
        for j in range(0, min(16, dim)):
            idx = (int(h[j * 2 : j * 2 + 2], 16) + i) % dim
            sign = 1 if int(h[16:18], 16) % 2 == 0 else -1
            vec[idx] += sign * 0.5
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


class HashEmbeddingProvider(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return EMBED_DIM

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array([_hash_embed(t, self.dim) for t in texts], dtype=np.float32)


class SBERTEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            raw_dim = self._model.get_sentence_embedding_dimension()
            # Index expects EMBED_DIM (768); pad if needed
            self._dim = max(raw_dim, EMBED_DIM)
            self._raw_dim = raw_dim
        except Exception as e:
            logger.warning("sentence-transformers not available: %s", e)
            raise

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        out = self._model.encode(texts, convert_to_numpy=True, dtype=np.float32)
        if self._raw_dim < EMBED_DIM:
            padded = np.zeros((len(out), EMBED_DIM), dtype=np.float32)
            padded[:, : self._raw_dim] = out
            # L2 normalize in full dim so magnitude is 1
            n = np.linalg.norm(padded, axis=1, keepdims=True)
            n = np.where(n < 1e-9, 1.0, n)
            out = (padded / n).astype(np.float32)
        return out


def get_embedding_provider() -> EmbeddingProvider:
    provider = (os.getenv("EMBEDDING_PROVIDER") or "hash").lower()
    if provider == "sbert":
        try:
            return SBERTEmbeddingProvider()
        except Exception:
            logger.warning("Falling back to hash embeddings")
            return HashEmbeddingProvider()
    return HashEmbeddingProvider()
