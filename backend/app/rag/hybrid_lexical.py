"""Lightweight BM25 lexical index; persist under backend/data/index/. No heavy deps."""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from app.core import INDEX_DIR

logger = logging.getLogger(__name__)

LEXICAL_INDEX_PATH = INDEX_DIR / "lexical_index.json"
LEXICAL_SCHEMA_VERSION = 1


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    doc_len: int,
    avg_doc_len: float,
    N: int,
    df: Dict[str, int],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 score for one document. Pure Python."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_tf: Dict[str, int] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    score = 0.0
    for t in set(query_tokens):
        if t not in doc_tf:
            continue
        tf = doc_tf[t]
        n_t = df.get(t, 0)
        idf = max(0.0, (N - n_t + 0.5) / (n_t + 0.5) + 1.0)
        idf = max(1e-6, idf)
        denom = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1e-9))
        score += idf * (tf * (k1 + 1)) / denom
    return score


class LexicalIndex:
    """BM25 index: chunk_id -> token list; persist to JSON. Normalize scores to [0,1] for fusion."""

    def __init__(self):
        self._chunk_ids: List[str] = []
        self._terms: Dict[str, List[str]] = {}  # chunk_id -> list of terms
        self._doc_lens: List[int] = []
        self._load()

    def _load(self):
        if LEXICAL_INDEX_PATH.exists():
            try:
                with open(LEXICAL_INDEX_PATH, "r") as f:
                    data = json.load(f)
                if data.get("schema_version") == LEXICAL_SCHEMA_VERSION:
                    self._chunk_ids = data.get("chunk_ids", [])
                    self._terms = {cid: data["terms"].get(cid, []) for cid in self._chunk_ids}
                    self._doc_lens = data.get("doc_lens", [len(self._terms.get(cid, [])) for cid in self._chunk_ids])
                    if len(self._doc_lens) != len(self._chunk_ids):
                        self._doc_lens = [len(self._terms.get(cid, [])) for cid in self._chunk_ids]
                else:
                    self._chunk_ids = []
                    self._terms = {}
                    self._doc_lens = []
            except Exception as e:
                logger.warning("Could not load lexical index: %s", e)
                self._chunk_ids = []
                self._terms = {}
                self._doc_lens = []

    def add(self, chunk_ids: List[str], texts: List[str]):
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts length mismatch")
        for cid, text in zip(chunk_ids, texts):
            toks = _tokenize(text)
            if cid in self._terms:
                idx = self._chunk_ids.index(cid)
                self._doc_lens[idx] = len(toks)
                self._terms[cid] = toks
            else:
                self._chunk_ids.append(cid)
                self._terms[cid] = toks
                self._doc_lens.append(len(toks))
        self._persist()

    def search(
        self, query: str, top_k: int, chunk_ids_subset: List[str] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Return list of {chunk_id, score} sorted by BM25.
        score is raw BM25; caller may normalize. If chunk_ids_subset, only score those.
        """
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        ids_to_score = chunk_ids_subset if chunk_ids_subset is not None else self._chunk_ids
        if not ids_to_score:
            return []
        N = len(self._chunk_ids)
        avg_doc_len = sum(self._doc_lens) / max(N, 1)
        df: Dict[str, int] = {}
        for toks in self._terms.values():
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        scored = []
        for cid in ids_to_score:
            if cid not in self._terms:
                continue
            toks = self._terms[cid]
            doc_len = len(toks)
            sc = _bm25_score(q_tokens, toks, doc_len, avg_doc_len, N, df)
            scored.append({"chunk_id": cid, "score": sc})
        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

    def search_all(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self.search(query, top_k, chunk_ids_subset=None)

    def get_terms(self, chunk_id: str) -> List[str]:
        return self._terms.get(chunk_id, [])

    def score_chunks(self, query: str, chunk_ids: List[str]) -> Dict[str, float]:
        """Return BM25 score for each chunk_id (0 if not in index). For hybrid fusion."""
        if not chunk_ids or not self._chunk_ids:
            return {cid: 0.0 for cid in chunk_ids}
        q_tokens = _tokenize(query)
        if not q_tokens:
            return {cid: 0.0 for cid in chunk_ids}
        N = len(self._chunk_ids)
        avg_doc_len = sum(self._doc_lens) / max(N, 1)
        df: Dict[str, int] = {}
        for toks in self._terms.values():
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        out = {}
        for cid in chunk_ids:
            if cid not in self._terms:
                out[cid] = 0.0
                continue
            toks = self._terms[cid]
            out[cid] = _bm25_score(q_tokens, toks, len(toks), avg_doc_len, N, df)
        return out

    def chunk_ids(self) -> List[str]:
        return list(self._chunk_ids)

    def _persist(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(LEXICAL_INDEX_PATH, "w") as f:
            json.dump({
                "schema_version": LEXICAL_SCHEMA_VERSION,
                "chunk_ids": self._chunk_ids,
                "terms": self._terms,
                "doc_lens": self._doc_lens,
            }, f)
        logger.debug("Lexical index persisted: %d chunks", len(self._chunk_ids))


_lexical_index: LexicalIndex | None = None


def get_lexical_index() -> LexicalIndex:
    global _lexical_index
    if _lexical_index is None:
        _lexical_index = LexicalIndex()
    return _lexical_index


def normalize_scores_to_unit(scored: List[Dict[str, Any]], key: str = "score") -> List[Dict[str, Any]]:
    """In-place normalize key to [0,1] by max; if max 0 leave as is."""
    if not scored:
        return scored
    mx = max(r.get(key, 0) for r in scored)
    if mx <= 0:
        return scored
    for r in scored:
        r[key] = r.get(key, 0) / mx
    return scored
