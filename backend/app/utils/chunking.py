"""Structure-aware chunking for any document type: markdown/HTML or plain text. Metadata per chunk (section_title, chunk_type, char_len, word_len)."""
import re
import logging
from typing import List, Dict, Any

from app.core import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

CHUNK_META_VERSION = 1


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _split_by_headings_md(text: str) -> List[str]:
    """Split by markdown or HTML headings; return list of segments (heading + content)."""
    parts = re.split(r"\n(?=#{1,6}\s)", text.strip())
    if len(parts) <= 1:
        parts = re.split(r"\n(?=<h[1-6])", text.strip(), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _is_heading_line(line: str) -> bool:
    """Best-effort: short line, all caps, ends with colon, underline context, or numbered heading."""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if s.endswith(":"):
        return True
    if re.match(r"^[A-Z][A-Z\s\-]+$", s) and len(s) < 60:
        return True
    if re.match(r"^\d+[.)]\s+.+", s) and len(s) < 100:  # "1. Introduction" style
        return True
    if re.match(r"^[A-Za-z][A-Za-z\s\-]{0,50}$", s) and "\n" not in s and not s.startswith(("*", "-", "•", "·")):
        if len(s.split()) <= 6 and len(s) < 80:
            return True
    return False


def _is_underline_line(line: str) -> bool:
    """Lines of --- or === often follow a heading in plain text."""
    s = line.strip()
    return bool(s and len(s) >= 2 and re.match(r"^[-=]{2,}$", s))


def _block_type(block: str) -> str:
    """One of 'heading', 'bullets', 'paragraph'."""
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    if not lines:
        return "paragraph"
    first = lines[0]
    if _is_heading_line(first) and len(lines) == 1:
        return "heading"
    if any(ln.startswith(("*", "-", "•", "·")) or re.match(r"^\d+[.)]\s", ln) for ln in lines):
        return "bullets"
    return "paragraph"


def _split_into_blocks_plain_text(text: str) -> List[Dict[str, Any]]:
    """
    Plain text: split by blank lines; detect headings (all caps, colon, underlines, numbered); keep lists grouped.
    Each block: { "text", "section_title", "block_type" }.
    """
    blocks: List[Dict[str, Any]] = []
    current_section = ""
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    for raw in raw_blocks:
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.split("\n")
        # Consume leading underline (e.g. --- under a heading)
        while len(lines) >= 2 and _is_underline_line(lines[-1]):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
        if not raw:
            continue
        # If first line looks like heading and rest is content, treat first line as section
        parts = raw.split("\n", 1)
        if len(parts) == 2 and _is_heading_line(parts[0]):
            current_section = parts[0].rstrip(":").strip()
            raw = parts[1].strip()
            if not raw:
                continue
        bt = _block_type(raw)
        if bt == "heading":
            current_section = raw.rstrip(":").strip()
            continue
        blocks.append({
            "text": raw,
            "section_title": current_section or "",
            "block_type": "bullets" if bt == "bullets" else "paragraph",
        })
    return blocks


def _split_into_blocks_markdown(text: str) -> List[Dict[str, Any]]:
    """
    Markdown/HTML: split by headings; keep list blocks together under same section.
    Each block: { "text", "section_title", "block_type" }.
    """
    segments = _split_by_headings_md(text)
    blocks: List[Dict[str, Any]] = []
    current_section = ""
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        lines = seg.split("\n")
        first_line = lines[0].strip() if lines else ""
        if _is_heading_line(first_line) and len(lines) == 1:
            current_section = first_line.rstrip(":").strip()
            continue
        if _is_heading_line(first_line) and len(lines) > 1:
            current_section = first_line.rstrip(":").strip()
            seg = "\n".join(lines[1:]).strip()
        bt = _block_type(seg)
        blocks.append({
            "text": seg,
            "section_title": current_section or "",
            "block_type": "bullets" if bt == "bullets" else "paragraph",
        })
    return blocks


def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    out = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            out.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return out


def _blocks_to_chunks(
    blocks: List[Dict[str, Any]],
    text_fallback: str,
) -> List[Dict[str, Any]]:
    """Merge blocks into chunks under CHUNK_SIZE; emit metadata per chunk."""
    out: List[Dict[str, Any]] = []
    current = ""
    current_section = ""
    current_type = "paragraph"
    for blk in blocks:
        seg = blk["text"]
        section = blk.get("section_title", "") or current_section
        btype = blk.get("block_type", "paragraph")
        if len(current) + len(seg) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + seg).strip() if current else seg
            current_section = section
            current_type = btype
        else:
            if current:
                out.append({
                    "text": current,
                    "section_title": current_section or "",
                    "chunk_type": current_type,
                    "char_len": len(current),
                    "word_len": _word_count(current),
                    "schema_version": CHUNK_META_VERSION,
                })
            if len(seg) > CHUNK_SIZE:
                for c in _sliding_window(seg, CHUNK_SIZE, CHUNK_OVERLAP):
                    out.append({
                        "text": c,
                        "section_title": section,
                        "chunk_type": btype,
                        "char_len": len(c),
                        "word_len": _word_count(c),
                        "schema_version": CHUNK_META_VERSION,
                    })
                current = ""
            else:
                current = seg
                current_section = section
                current_type = btype
    if current:
        out.append({
            "text": current,
            "section_title": current_section or "",
            "chunk_type": current_type,
            "char_len": len(current),
            "word_len": _word_count(current),
            "schema_version": CHUNK_META_VERSION,
        })
    if not out and text_fallback:
        for c in _sliding_window(text_fallback, CHUNK_SIZE, CHUNK_OVERLAP):
            out.append({
                "text": c,
                "section_title": "",
                "chunk_type": "paragraph",
                "char_len": len(c),
                "word_len": _word_count(c),
                "schema_version": CHUNK_META_VERSION,
            })
    return out


def chunk_text(text: str, is_markdown_or_html: bool = False) -> List[str]:
    """Legacy: return list of chunk strings only (backward compatible)."""
    return [c["text"] for c in chunk_text_with_metadata(text, is_markdown_or_html)]


def chunk_text_with_metadata(
    text: str, is_markdown_or_html: bool = False
) -> List[Dict[str, Any]]:
    """
    Structure-aware chunking for any document type.
    - Markdown/HTML: split by headings; keep lists together; preserve sections.
    - Plain text: split by blank lines; detect headings (all caps, colon, underlines, numbered); keep bullet/numbered lists grouped; fallback to sliding window.
    Returns list of dicts: text, section_title, chunk_type (heading|bullets|paragraph), char_len, word_len, schema_version.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if is_markdown_or_html:
        blocks = _split_into_blocks_markdown(text)
    else:
        blocks = _split_into_blocks_plain_text(text)
    return _blocks_to_chunks(blocks, text)
