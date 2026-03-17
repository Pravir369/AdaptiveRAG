"""Generator interface: extractive (default) or ollama. build_answer_with_citations returns (answer, citations, metadata)."""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.db import Chunk, Doc
from app.db.session import SessionLocal
from app.core import GENERATION_PROVIDER, OLLAMA_MODEL, OLLAMA_MAX_TOKENS, OLLAMA_TEMPERATURE

from .ollama import OllamaGenerator

logger = logging.getLogger(__name__)

_INLINE_CITE_NOISE = re.compile(r"\(chunk_id[=:]?\s*([0-9a-f-]+)\)", re.IGNORECASE)
_BRACKET_CITE_NOISE = re.compile(r"\[chunk_id=([0-9a-f-]+)\]")
_CHUNK_ID_COLON = re.compile(r"chunk_id:\s*([0-9a-f-]+)")


def _clean_answer_text(text: str) -> str:
    """Clean up verbose citation noise in model output for user-facing display."""
    text = _INLINE_CITE_NOISE.sub(r"(\1)", text)
    text = _BRACKET_CITE_NOISE.sub(r"(\1)", text)
    text = _CHUNK_ID_COLON.sub(r"(\1)", text)
    text = re.sub(r"\(([0-9a-f-]{36})\)(\s*\(\1\))+", r"(\1)", text)
    return text.strip()


class Generator:
    """Base interface for answer generation."""

    def generate(self, query: str, selected_chunks: List[Dict[str, Any]], chunk_texts: Optional[Dict[str, str]] = None) -> str:
        raise NotImplementedError


class ExtractiveGenerator(Generator):
    """Build answer from selected chunks; no LLM. If no evidence, say so."""

    def generate(
        self,
        query: str,
        selected_chunks: List[Dict[str, Any]],
        chunk_texts: Optional[Dict[str, str]] = None,
    ) -> str:
        if not selected_chunks:
            return "I don't have enough info in the indexed sources."
        if chunk_texts:
            texts = [chunk_texts.get(s["chunk_id"], "") for s in selected_chunks]
        else:
            db = SessionLocal()
            try:
                texts = []
                for s in selected_chunks:
                    c = db.query(Chunk).filter(Chunk.chunk_id == s["chunk_id"]).first()
                    if c:
                        texts.append(c.text)
            finally:
                db.close()
        texts = [t for t in texts if t]
        if not texts:
            return "I don't have enough info in the indexed sources."
        combined = " ".join(texts[:3])
        sentences = []
        for s in combined.replace("\n", " ").split(". "):
            s = s.strip()
            if s and len(" ".join(sentences)) + len(s) < 500:
                sentences.append(s + ("." if not s.endswith(".") else ""))
            else:
                break
        answer = " ".join(sentences) if sentences else (combined[:400].rsplit(". ", 1)[0] + "." if ". " in combined[:400] else combined[:400])
        return answer.strip() or "I don't have enough info in the indexed sources."


def _overview_bullets(selected: List[Dict[str, Any]], db_session, chunk_texts: Optional[Dict[str, str]] = None) -> str:
    """One bullet per selected chunk; label = section_title or 'Key point 1/2/3'. No LLM."""
    bullets = []
    for i, s in enumerate(selected):
        c = db_session.query(Chunk).filter(Chunk.chunk_id == s["chunk_id"]).first()
        if not c:
            continue
        label = (c.section_title or "").strip() or f"Key point {i + 1}"
        text = chunk_texts.get(c.chunk_id, c.text) if chunk_texts else c.text
        first_sentence = text.split(". ")[0].strip()
        if not first_sentence.endswith("."):
            first_sentence += "."
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117].rsplit(" ", 1)[0] + "…"
        bullets.append(f"• {label}: {first_sentence}")
    return "\n".join(bullets) if bullets else "I don't have enough info in the indexed sources."


def _build_citations_for_chunk_ids(
    chunk_ids: List[str],
    db_session,
) -> List[Dict[str, Any]]:
    """Build list of {doc_id, chunk_id, snippet} for given chunk_ids."""
    citations = []
    for cid in chunk_ids:
        c = db_session.query(Chunk).filter(Chunk.chunk_id == cid).first()
        if not c:
            continue
        doc = db_session.query(Doc).filter(Doc.doc_id == c.doc_id).first()
        doc_id = doc.doc_id if doc else c.doc_id
        snippet = (c.text[:200] + "…") if len(c.text) > 200 else c.text
        citations.append({"doc_id": doc_id, "chunk_id": c.chunk_id, "snippet": snippet})
    return citations


def build_answer_with_citations(
    query: str,
    selected: List[Dict[str, Any]],
    db_session,
    chunk_texts: Optional[Dict[str, str]] = None,
    mode: Optional[str] = None,
    provider: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build answer and citations. Returns (answer, citations, generator_metadata).
    If mode=='vague_summary', use deterministic overview bullets (no LLM).
    provider: 'extractive' | 'ollama' (default from config).
    """
    if provider is None:
        provider = GENERATION_PROVIDER
    metadata = {}

    if mode == "vague_summary":
        answer = _overview_bullets(selected, db_session, chunk_texts=chunk_texts)
        citations = _build_citations_for_chunk_ids([s["chunk_id"] for s in selected[:10]], db_session)
        return answer, citations, metadata

    if provider == "ollama":
        try:
            gen = OllamaGenerator()
            out = gen.generate_answer(query, selected, chunk_texts or {}, conversation_history=conversation_history)
            text = _clean_answer_text(out["text"])
            citations_used = out.get("citations_used") or []
            citations = _build_citations_for_chunk_ids(citations_used, db_session)
            metadata = {
                "model": OLLAMA_MODEL,
                "num_context_chunks": len(selected),
                "max_tokens": OLLAMA_MAX_TOKENS,
                "temperature": OLLAMA_TEMPERATURE,
                "citation_strict_mode": True,
            }
            return text, citations, metadata
        except Exception as e:
            logger.warning("Ollama generation failed (%s), fallback to extractive", e)
            raise

    gen = ExtractiveGenerator()
    answer = gen.generate(query, selected, chunk_texts=chunk_texts)
    citations = _build_citations_for_chunk_ids([s["chunk_id"] for s in selected[:10]], db_session)
    return answer, citations, metadata


def get_generator(provider: Optional[str] = None) -> Generator:
    """Return generator instance. Default extractive."""
    if provider is None:
        provider = GENERATION_PROVIDER
    if provider == "ollama":
        return OllamaGenerator()
    return ExtractiveGenerator()
