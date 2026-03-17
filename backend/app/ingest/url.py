"""URL ingestion: readability-lxml if available, else BeautifulSoup + heuristics. Robust to bad pages."""
import hashlib
import logging
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from sqlalchemy.orm import Session

from app.db.models import Doc, Chunk
from app.utils.chunking import chunk_text_with_metadata
from app.rag.embeddings import get_embedding_provider
from app.rag.index import get_vector_index
from app.rag.hybrid_lexical import get_lexical_index
from app.core import EMBEDDING_PROVIDER

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; AdaptiveRAG/1.0; +https://github.com/adaptiverag)"


def _extract_with_readability(html: str, url: str) -> Optional[str]:
    try:
        from readability import Document
        doc = Document(html)
        title = doc.title()
        body = doc.summary()
        # Strip HTML tags for plain text
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(body, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.debug("readability failed: %s", e)
        return None


def _extract_with_bs4(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception as e:
        logger.warning("BeautifulSoup extract failed: %s", e)
        return ""


def extract_text_from_html(html: str, url: str = "") -> str:
    """Extract main text from HTML. readability-lxml first, else BeautifulSoup."""
    text = _extract_with_readability(html, url)
    if not text or len(text.strip()) < 50:
        text = _extract_with_bs4(html)
    return text or ""


def fetch_url(url: str) -> Tuple[bool, str, str]:
    """Fetch URL; return (ok, title_or_url, body_text). Does not crash on bad pages."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        html = resp.text
        if not html or len(html) < 100:
            return False, url, ""
        text = extract_text_from_html(html, url)
        title = urlparse(url).path.rstrip("/").split("/")[-1] or url
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()[:200]
        except Exception:
            pass
        return True, title, text
    except Exception as e:
        logger.warning("URL fetch failed %s: %s", url, e)
        return False, url, ""


def ingest_url(db: Session, url: str, title: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Ingest URL into docs + chunks + vector index.
    Returns (success, doc_id, message).
    """
    ok, fetched_title, text = fetch_url(url)
    if not ok or not text.strip():
        return False, "", "Failed to fetch or extract text from URL"
    title = title or fetched_title
    doc = Doc(title=title[:1024], source_type="url", source_ref=url[:2048])
    db.add(doc)
    db.flush()
    doc_id = doc.doc_id
    is_md = ".md" in url or "markdown" in url.lower()
    chunks_with_meta = chunk_text_with_metadata(text, is_markdown_or_html=True)
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
