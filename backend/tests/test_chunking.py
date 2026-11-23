from app.utils.chunking import chunk_text, chunk_text_with_metadata


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_markdown_sections_keep_their_titles():
    md = "# Install\n\nRun pip install.\n\n# Usage\n\nCall the API."
    chunks = chunk_text_with_metadata(md, is_markdown_or_html=True)
    assert chunks, "expected at least one chunk"
    joined = " ".join(c["text"] for c in chunks)
    assert "pip install" in joined and "Call the API" in joined
    for c in chunks:
        assert {"text", "section_title", "chunk_type"} <= set(c)


def test_long_text_is_split():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(20))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
