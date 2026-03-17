"""Save and fetch trace by trace_id."""
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import Trace

logger = logging.getLogger(__name__)


def save_trace(
    db: Session,
    query: str,
    embedding_provider: str,
    top_k: int,
    retrieved: List[Dict[str, Any]],
    reranked: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    answer: str,
    latency_ms: int,
    attempts: Optional[List[Dict[str, Any]]] = None,
    heals_applied: Optional[List[Dict[str, Any]]] = None,
    final_selected: Optional[List[Dict[str, Any]]] = None,
    self_heal_triggered: Optional[bool] = None,
    trigger_failed_thresholds: Optional[List[Dict[str, Any]]] = None,
    best_attempt_no: Optional[int] = None,
    generator_provider: Optional[str] = None,
    generator_metadata: Optional[Dict[str, Any]] = None,
    generator_error: Optional[str] = None,
) -> str:
    trace_id = str(uuid4())
    payload_attempts = list(attempts or [])
    payload_heals = list(heals_applied or [])
    t = Trace(
        trace_id=trace_id,
        query=query,
        embedding_provider=embedding_provider,
        top_k=top_k,
        retrieved_json=json.dumps(retrieved),
        reranked_json=json.dumps(reranked),
        selected_json=json.dumps(selected),
        citations_json=json.dumps(citations),
        answer=answer,
        latency_ms=latency_ms,
        attempts_json=json.dumps(payload_attempts) if payload_attempts else None,
        heals_json=json.dumps(payload_heals) if payload_heals else None,
        final_selected_json=json.dumps(final_selected) if final_selected else None,
        self_heal_triggered=self_heal_triggered,
        trigger_failed_thresholds_json=json.dumps(trigger_failed_thresholds) if trigger_failed_thresholds else None,
        best_attempt_no=best_attempt_no,
        generator_provider=generator_provider,
        generator_metadata_json=json.dumps(generator_metadata) if generator_metadata else None,
        generator_error=generator_error,
    )
    db.add(t)
    db.commit()
    return trace_id


def get_trace(db: Session, trace_id: str) -> Optional[Dict[str, Any]]:
    from app.db.models import Chunk
    t = db.query(Trace).filter(Trace.trace_id == trace_id).first()
    if not t:
        return None
    retrieved = json.loads(t.retrieved_json) if t.retrieved_json else []
    chunk_ids = [r.get("chunk_id") for r in retrieved]
    chunk_to_doc = {}
    if chunk_ids:
        for c in db.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids)):
            chunk_to_doc[c.chunk_id] = c.doc_id
    for r in retrieved:
        r["doc_id"] = chunk_to_doc.get(r.get("chunk_id"), "")
    retrieved_doc_ids = list({chunk_to_doc.get(r.get("chunk_id")) for r in retrieved if chunk_to_doc.get(r.get("chunk_id"))})
    attempts = json.loads(t.attempts_json) if t.attempts_json else []
    heals_applied = json.loads(t.heals_json) if t.heals_json else []
    final_selected = json.loads(t.final_selected_json) if t.final_selected_json else None
    trigger_failed_thresholds = json.loads(t.trigger_failed_thresholds_json) if t.trigger_failed_thresholds_json else []
    return {
        "trace_id": t.trace_id,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "query": t.query,
        "embedding_provider": t.embedding_provider,
        "top_k": t.top_k,
        "retrieved": retrieved,
        "retrieved_doc_ids": retrieved_doc_ids,
        "reranked": json.loads(t.reranked_json) if t.reranked_json else [],
        "selected": json.loads(t.selected_json) if t.selected_json else [],
        "citations": json.loads(t.citations_json) if t.citations_json else [],
        "answer": t.answer,
        "latency_ms": t.latency_ms,
        "attempts": attempts,
        "heals_applied": heals_applied,
        "final_selected": final_selected,
        "self_heal_triggered": bool(t.self_heal_triggered) if t.self_heal_triggered is not None else False,
        "trigger_failed_thresholds": trigger_failed_thresholds,
        "best_attempt_no": t.best_attempt_no if t.best_attempt_no is not None else 1,
        "generator_provider": getattr(t, "generator_provider", None),
        "generator_metadata": json.loads(t.generator_metadata_json) if (getattr(t, "generator_metadata_json", None) and t.generator_metadata_json) else None,
        "generator_error": getattr(t, "generator_error", None),
    }
