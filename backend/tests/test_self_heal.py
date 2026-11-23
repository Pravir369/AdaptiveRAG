"""The self-heal loop fires on vague queries and records what it did."""
from app.db import init_db
from app.db.session import SessionLocal
from app.ingest import ingest_upload
from app.rag.retriever import retrieve_and_rerank


def test_vague_query_triggers_self_heal_and_is_traced():
    init_db()
    db = SessionLocal()
    try:
        for i, topic in enumerate(["sqlite connections", "zip archives", "url parsing"]):
            ingest_upload(db, f"heal{i}.txt", f"Notes on {topic}. ".encode() * 20)
    finally:
        db.close()
    reranked, selected, _, attempts, heals, triggered, failed, best_no, mode = retrieve_and_rerank(
        "Explain this", 12, "hash"
    )
    assert reranked, "should still retrieve something"
    assert triggered is True
    assert failed, "the failing thresholds should be recorded"
    assert len(attempts) >= 2
    assert heals and heals[0]["type"] == "QUERY_REWRITE"
    assert 1 <= best_no <= len(attempts)
