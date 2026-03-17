"""Ollama-backed grounded generator: HTTP API first, CLI fallback. Flexible citation parsing."""
import os
import re
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import requests

from app.core import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
)

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_OLLAMA_BIN = _BACKEND_ROOT / "bin" / "ollama"

_UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)

_CITE_PATTERNS = [
    re.compile(r"\(([^)]+)\)"),
    re.compile(r"\[chunk_id=([^\]]+)\]"),
    re.compile(r"\[([^\]]*[0-9a-f]{8}-[^\]]*)\]"),
    re.compile(r"chunk_id[=:\s]+([0-9a-f]{8}-[0-9a-f-]+)"),
    _UUID_PATTERN,
]


def _ollama_cli_path() -> str:
    if _LOCAL_OLLAMA_BIN.exists():
        return str(_LOCAL_OLLAMA_BIN)
    return "ollama"


def parse_citations(text: str, valid_chunk_ids: Optional[Set[str]] = None) -> List[str]:
    """Extract chunk_ids from text using multiple patterns. Returns only valid IDs if set given."""
    found = []
    for pattern in _CITE_PATTERNS:
        for m in pattern.finditer(text):
            inner = m.group(1) if pattern.groups else m.group(0)
            inner = inner.strip()
            for part in re.split(r"[\s,;]+", inner):
                part = part.strip().strip("()")
                if not part:
                    continue
                if valid_chunk_ids is not None:
                    if part in valid_chunk_ids:
                        if part not in found:
                            found.append(part)
                    else:
                        for vid in valid_chunk_ids:
                            if vid.startswith(part) or part.startswith(vid[:8]):
                                if vid not in found:
                                    found.append(vid)
                else:
                    if part not in found:
                        found.append(part)
    return found


def _build_prompt(question: str, context_chunks: List[Dict[str, Any]], chunk_texts: Dict[str, str], conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Build a conversational grounded prompt."""
    chunk_ids = [s.get("chunk_id", "") for s in context_chunks]
    short_ids = [cid[:8] for cid in chunk_ids if cid]

    system_block = (
        "You are a helpful, friendly assistant. Answer the user's question using ONLY the information "
        "from the context documents provided below. Be conversational and natural in your response.\n\n"
        "Important rules:\n"
        "- Use ONLY information from the provided context. Do not make up facts.\n"
        "- If the context doesn't contain the answer, say so honestly.\n"
        "- Write in a natural, conversational tone — not robotic bullet points.\n"
        "- Be thorough but concise. A few sentences to a short paragraph is ideal.\n"
        "- At the end of your response, list which sources you used on a new line like: Sources: [id1], [id2]\n"
        f"- Use the short chunk IDs: {', '.join(short_ids[:5])}\n"
    )

    history_block = ""
    if conversation_history:
        history_block = "\n--- Previous conversation ---\n"
        for msg in conversation_history[-8:]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:400]
            history_block += f"{role_label}: {content}\n"
        history_block += "--- End of previous conversation ---\n\n"

    context_block = "--- Context Documents ---\n"
    for s in context_chunks:
        cid = s.get("chunk_id", "")
        text = chunk_texts.get(cid, "").strip()
        if text:
            truncated = text[:2000] if len(text) > 2000 else text
            context_block += f"\n[{cid[:8]}] {truncated}\n"
    context_block += "--- End of Context ---\n"

    return (
        f"{system_block}\n"
        f"{history_block}"
        f"{context_block}\n"
        f"User: {question}\n\n"
        "Assistant:"
    )


def _call_ollama_http(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_MAX_TOKENS,
        },
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def _call_ollama_cli(prompt: str) -> str:
    bin_path = _ollama_cli_path()
    cmd = [bin_path, "run", OLLAMA_MODEL]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or f"ollama exit {proc.returncode}")
        return (proc.stdout or "").strip()
    except FileNotFoundError:
        raise RuntimeError(f"ollama CLI not found at {bin_path}")


def call_ollama(prompt: str) -> str:
    try:
        return _call_ollama_http(prompt)
    except Exception as e:
        logger.warning("Ollama HTTP failed (%s), trying CLI fallback", e)
        return _call_ollama_cli(prompt)


def _extract_citations_from_response(text: str, valid_chunk_ids: Set[str]) -> List[str]:
    """Extract all valid chunk_id references from the model response."""
    found = []
    for vid in valid_chunk_ids:
        short_id = vid[:8]
        if short_id in text or vid in text:
            if vid not in found:
                found.append(vid)

    extra = parse_citations(text, valid_chunk_ids)
    for cid in extra:
        if cid not in found:
            found.append(cid)

    return found


def _clean_source_line(text: str) -> str:
    """Remove the 'Sources: [...]' line from the answer since we show citations separately in the UI."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("sources:") or stripped.startswith("source:"):
            continue
        if stripped.startswith("references:") or stripped.startswith("citation"):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result if result else text.strip()


class OllamaGenerator:
    """Grounded generator using Ollama (HTTP or CLI)."""

    def generate_answer(
        self,
        query: str,
        selected_chunks: List[Dict[str, Any]],
        chunk_texts: Dict[str, str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not selected_chunks or not chunk_texts:
            return {"text": "I don't have any documents to answer from yet. Try ingesting some documents first!", "citations_used": []}

        valid_ids = {s["chunk_id"] for s in selected_chunks}
        prompt = _build_prompt(query, selected_chunks, chunk_texts, conversation_history=conversation_history)
        logger.info("Calling Ollama (%s) with %d context chunks, prompt ~%d chars", OLLAMA_MODEL, len(selected_chunks), len(prompt))

        raw = call_ollama(prompt)
        logger.info("Ollama raw response (%d chars): %s", len(raw), raw[:500])

        citations_used = _extract_citations_from_response(raw, valid_ids)

        if not citations_used:
            citations_used = [s["chunk_id"] for s in selected_chunks[:3]]
            logger.info("No explicit citations found in response, using top %d context chunks", len(citations_used))

        answer = _clean_source_line(raw)

        if not answer or answer.lower().startswith("not found"):
            answer = raw.strip() if raw.strip() else "I couldn't find a clear answer in the documents for that question."

        logger.info("Final answer length %d, %d citations", len(answer), len(citations_used))
        return {"text": answer, "citations_used": citations_used}
