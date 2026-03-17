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

# Patterns that match various citation formats models might use
_CITE_PATTERNS = [
    re.compile(r"\(([^)]+)\)"),                          # (id) or (id1, id2)
    re.compile(r"\[chunk_id=([^\]]+)\]"),                 # [chunk_id=id]
    re.compile(r"chunk_id[=:\s]+([0-9a-f]{8}-[0-9a-f-]+)"),  # chunk_id=UUID or chunk_id: UUID
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
            inner = m.group(1).strip()
            for part in re.split(r"[\s,]+", inner):
                part = part.strip()
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
    """Build grounded prompt with system rules, question, context, and output format."""
    chunk_ids = [s.get("chunk_id", "") for s in context_chunks]
    id_list = ", ".join(chunk_ids[:5])

    system_block = (
        "You are a precise, grounded assistant. Your ONLY knowledge source is the context chunks below.\n"
        "Rules:\n"
        "1. Answer ONLY from the provided context. Do NOT use prior knowledge.\n"
        "2. If the answer is not in the context, reply EXACTLY: Not found in the provided documents.\n"
        "3. Format: short bullets. Each bullet MUST end with a citation.\n"
        f"4. Citation format: put the chunk_id in parentheses at the end of each bullet, like ({chunk_ids[0] if chunk_ids else 'chunk_id'})\n"
        "5. Do NOT cite a chunk you did not use. Do NOT invent information.\n"
        "6. Keep answers concise - 2-5 bullets maximum.\n"
    )

    history_block = ""
    if conversation_history:
        history_block = "Previous conversation:\n"
        for msg in conversation_history[-6:]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:300]
            history_block += f"{role_label}: {content}\n"
        history_block += "\n"

    context_block = "Context:\n"
    for s in context_chunks:
        cid = s.get("chunk_id", "")
        text = chunk_texts.get(cid, "").strip()
        if text:
            truncated = text[:1500] if len(text) > 1500 else text
            context_block += f"\n[chunk_id={cid}]\n{truncated}\n"

    return (
        f"{system_block}\n"
        f"{history_block}"
        f"Question: {question}\n\n"
        f"{context_block}\n"
        f"Available chunk_ids for citation: {id_list}\n\n"
        "Answer (short bullets, each ending with (chunk_id) citation):"
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


def enforce_citations(
    text: str, valid_chunk_ids: Set[str]
) -> Tuple[str, List[str]]:
    """
    Two-pass citation enforcement:
    1. Try strict: keep only lines with valid citations
    2. If strict yields nothing, try lenient: extract all valid citations from full text,
       and return full text with the found citations
    """
    if not text or not valid_chunk_ids:
        return "Not found in the provided documents.", []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Pass 1: strict - keep lines with valid citations
    strict_kept = []
    strict_citations = set()
    for line in lines:
        ids_in_line = parse_citations(line, valid_chunk_ids)
        if ids_in_line:
            strict_kept.append(line)
            strict_citations.update(ids_in_line)

    if strict_kept and strict_citations:
        return "\n".join(strict_kept), list(strict_citations)

    # Pass 2: lenient - extract citations from full text, keep all content lines
    all_citations = parse_citations(text, valid_chunk_ids)
    if all_citations:
        content_lines = [ln for ln in lines if not ln.startswith("Not found")]
        if content_lines:
            return "\n".join(content_lines), list(all_citations)

    # Pass 3: if model gave content but no citations, return with all chunk_ids as citations
    content_lines = [ln for ln in lines if ln.strip() and not ln.startswith("Not found")]
    if content_lines and len(content_lines) >= 2:
        logger.warning("Model produced answer but no valid citations; attaching top chunk_ids")
        return "\n".join(content_lines), list(valid_chunk_ids)[:3]

    return "Not found in the provided documents.", []


class OllamaGenerator:
    """Grounded generator using Ollama (HTTP or CLI). Every claim should cite a chunk_id."""

    def generate_answer(
        self,
        query: str,
        selected_chunks: List[Dict[str, Any]],
        chunk_texts: Dict[str, str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not selected_chunks or not chunk_texts:
            return {"text": "Not found in the provided documents.", "citations_used": []}
        valid_ids = {s["chunk_id"] for s in selected_chunks}
        prompt = _build_prompt(query, selected_chunks, chunk_texts, conversation_history=conversation_history)
        logger.info("Calling Ollama (%s) with %d context chunks, prompt ~%d chars", OLLAMA_MODEL, len(selected_chunks), len(prompt))
        raw = call_ollama(prompt)
        logger.info("Ollama raw response (%d chars): %s", len(raw), raw[:500])
        text, citations_used = enforce_citations(raw, valid_ids)
        logger.info("After citation processing: %d citations used, answer length %d", len(citations_used), len(text))
        return {"text": text, "citations_used": citations_used}
