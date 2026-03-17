"""File upload ingestion: .txt, .md, .json (as text). Optional .pdf if extras installed."""
import hashlib
import logging
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import Doc, Chunk
from app.utils.chunking import chunk_text_with_metadata
from app.rag.embeddings import get_embedding_provider
from app.rag.index import get_vector_index
from app.rag.hybrid_lexical import get_lexical_index

logger = logging.getLogger(__name__)


def _read_pdf(content: bytes) -> Optional[str]:
    try:
        import pypdf  # optional
        from io import BytesIO
        reader = pypdf.PdfReader(BytesIO(content))
        parts = []
        for p in reader.pages:
            parts.append(p.extract_text() or "")
        return "\n\n".join(parts).strip()
    except Exception as e:
        logger.debug("pypdf not available or failed: %s", e)
        return None


def extract_text_from_file(filename: str, content: bytes) -> Tuple[bool, str]:
    """Extract text from upload. Returns (success, text). PDF optional."""
    ext = (filename or "").lower().split(".")[-1]
    if ext == "txt":
        return True, content.decode("utf-8", errors="replace")
    if ext == "md":
        return True, content.decode("utf-8", errors="replace")
    if ext == "json":
        return True, content.decode("utf-8", errors="replace")
    if ext == "pdf":
        text = _read_pdf(content)
        return text is not None, text or ""
    return False, ""


def ingest_upload(
    db: Session,
    filename: str,
    content: bytes,
    title: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Ingest uploaded file into docs + chunks + vector index.
    Returns (success, doc_id, message).
    """
    ok, text = extract_text_from_file(filename, content)
    if not ok:
        return False, "", f"Unsupported file type or PDF not available: {filename}"
    if not text or not text.strip():
        return False, "", "No text extracted"
    title = title or (filename[:1024] if filename else "upload")
    doc = Doc(title=title, source_type="upload", source_ref=filename[:2048])
    db.add(doc)
    db.flush()
    doc_id = doc.doc_id
    is_md = (filename or "").lower().endswith(".md")
    chunks_with_meta = chunk_text_with_metadata(text, is_markdown_or_html=is_md)
    texts = [c["text"] for c in chunks_with_meta]
    provider = get_embedding_provider()
    index = get_vector_index()
    lex_index = get_lexical_index()
    chunk_ids = []
    for i, cmeta in enumerate(chunks_with_meta):
        ct = cmeta["text"]
        text_hash = hashlib.sha256(ct.encode("utf-8")).hexdigest()[:64]
        ch = Chunk(
            doc_id=doc_id,
            chunk_index=i,
            text=ct,
            text_hash=text_hash,
            section_title=cmeta.get("section_title") or None,
            chunk_type=cmeta.get("chunk_type") or None,
        )
        db.add(ch)
        db.flush()
        chunk_ids.append(ch.chunk_id)
    db.commit()
    if chunk_ids:
        embeddings = provider.embed_texts(texts)
        index.add(embeddings, chunk_ids)
        lex_index.add(chunk_ids, texts)
    return True, doc_id, f"Ingested {len(texts)} chunks"
