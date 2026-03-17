"""FastAPI routes: health, ingest, chat, trace, conversations."""
import time
import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.db import get_db, init_db, Doc, Chunk, Trace
from app.db.models import Conversation as ConvModel, Message as MsgModel
from app.db.session import SessionLocal
from app.models import ChatRequest, ChatResponse, Citation, IngestURLRequest, TraceResponse
from app.ingest import ingest_upload, ingest_url
from app.rag.embeddings import get_embedding_provider
from app.rag.retriever import retrieve_and_rerank
from app.rag.generator import build_answer_with_citations
from app.trace import save_trace, get_trace
from app.core import EMBEDDING_PROVIDER, DEFAULT_TOP_K, GENERATION_PROVIDER

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ingest/upload")
def upload_ingest(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Multipart upload: .txt, .md, .json; optional .pdf if installed."""
    content = file.file.read()
    success, doc_id, msg = ingest_upload(db, file.filename or "upload", content)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"doc_id": doc_id, "message": msg}


@router.post("/ingest/url")
def url_ingest(
    body: IngestURLRequest,
    db: Session = Depends(get_db),
):
    """Ingest from URL."""
    success, doc_id, msg = ingest_url(db, body.url, title=body.title)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"doc_id": doc_id, "message": msg}


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
):
    """RAG chat: query -> answer + citations + trace_id, with conversation memory."""
    top_k = req.top_k or DEFAULT_TOP_K
    provider = get_embedding_provider()
    provider_name = getattr(provider, "__class__", type(provider)).__name__.replace("EmbeddingProvider", "").lower() or "hash"
    if "sbert" in provider_name:
        provider_name = "sbert"
    else:
        provider_name = "hash"

    # Load or create conversation
    conversation_id = req.conversation_id
    if conversation_id:
        conv = db.query(ConvModel).filter(ConvModel.conversation_id == conversation_id).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = ConvModel(title=req.query[:60] + ("..." if len(req.query) > 60 else ""))
        db.add(conv)
        db.flush()
        conversation_id = conv.conversation_id

    # Load conversation history (last 10 messages for context)
    history_messages = db.query(MsgModel).filter(
        MsgModel.conversation_id == conversation_id
    ).order_by(MsgModel.created_at.asc()).all()
    conversation_history = [
        {"role": m.role, "content": m.content}
        for m in history_messages[-10:]
    ]

    start = time.perf_counter()
    generator_provider = GENERATION_PROVIDER
    generator_metadata = None
    generator_error = None
    try:
        reranked, selected, chunk_texts, attempts, heals_applied, self_heal_triggered, trigger_failed_thresholds, best_attempt_no, mode = retrieve_and_rerank(
            req.query, top_k, provider_name
        )
        if GENERATION_PROVIDER == "ollama":
            try:
                answer, citations, generator_metadata = build_answer_with_citations(
                    req.query, selected, db, chunk_texts=chunk_texts, mode=mode, provider="ollama",
                    conversation_history=conversation_history,
                )
                generator_provider = "ollama"
            except Exception as e:
                logger.warning("Ollama unavailable (%s), using extractive fallback", e)
                answer, citations, _ = build_answer_with_citations(
                    req.query, selected, db, chunk_texts=chunk_texts, mode=mode, provider="extractive",
                    conversation_history=conversation_history,
                )
                generator_provider = "extractive_fallback"
                generator_error = str(e)
        else:
            answer, citations, generator_metadata = build_answer_with_citations(
                req.query, selected, db, chunk_texts=chunk_texts, mode=mode, provider="extractive",
                conversation_history=conversation_history,
            )
    except Exception as e:
        logger.exception("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = int((time.perf_counter() - start) * 1000)
    from app.db.models import Chunk as ChunkModel
    retrieved_for_trace = []
    for r in reranked:
        c = db.query(ChunkModel).filter(ChunkModel.chunk_id == r["chunk_id"]).first()
        doc_id = c.doc_id if c else ""
        retrieved_for_trace.append({
            "chunk_id": r["chunk_id"], "doc_id": doc_id,
            "score": r.get("fused_score") or r.get("score"),
            "vector_score": r.get("vector_score"), "lexical_score": r.get("lexical_score"), "fused_score": r.get("fused_score"),
        })
    selected_for_trace = [{"chunk_id": s["chunk_id"], "score": s["score"]} for s in selected]
    trace_id = save_trace(
        db,
        query=req.query,
        embedding_provider=provider_name,
        top_k=top_k,
        retrieved=retrieved_for_trace,
        reranked=retrieved_for_trace,
        selected=selected_for_trace,
        citations=citations,
        answer=answer,
        latency_ms=latency_ms,
        attempts=attempts,
        heals_applied=heals_applied,
        final_selected=selected_for_trace,
        self_heal_triggered=self_heal_triggered,
        trigger_failed_thresholds=trigger_failed_thresholds,
        best_attempt_no=best_attempt_no,
        generator_provider=generator_provider,
        generator_metadata=generator_metadata,
        generator_error=generator_error,
    )

    # Save user and assistant messages
    user_msg = MsgModel(
        conversation_id=conversation_id,
        role="user",
        content=req.query,
    )
    assistant_msg = MsgModel(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        trace_id=trace_id,
        generator_provider=generator_provider,
    )
    db.add(user_msg)
    db.add(assistant_msg)
    conv.updated_at = datetime.utcnow()
    db.commit()

    return ChatResponse(
        answer=answer,
        citations=[Citation(doc_id=c["doc_id"], chunk_id=c["chunk_id"], snippet=c["snippet"]) for c in citations],
        trace_id=trace_id,
        generator_provider=generator_provider,
        conversation_id=conversation_id,
    )


@router.get("/trace/{trace_id}", response_model=TraceResponse)
def trace_detail(
    trace_id: str,
  db: Session = Depends(get_db),
):
    """Return full trace JSON."""
    data = get_trace(db, trace_id)
    if not data:
        raise HTTPException(status_code=404, detail="Trace not found")
    return TraceResponse(
        trace_id=data["trace_id"],
        created_at=data.get("created_at"),
        query=data["query"],
        embedding_provider=data.get("embedding_provider"),
        top_k=data.get("top_k"),
        retrieved=data.get("retrieved", []),
        retrieved_doc_ids=data.get("retrieved_doc_ids", []),
        reranked=data.get("reranked", []),
        selected=data.get("selected", []),
        citations=data.get("citations", []),
        answer=data.get("answer"),
        latency_ms=data.get("latency_ms"),
        attempts=data.get("attempts", []),
        heals_applied=data.get("heals_applied", []),
        final_selected=data.get("final_selected"),
        self_heal_triggered=data.get("self_heal_triggered", False),
        trigger_failed_thresholds=data.get("trigger_failed_thresholds", []),
        best_attempt_no=data.get("best_attempt_no", 1),
        generator_provider=data.get("generator_provider"),
        generator_metadata=data.get("generator_metadata"),
        generator_error=data.get("generator_error"),
    )


@router.get("/conversations")
def list_conversations(db: Session = Depends(get_db)):
    """List all conversations, newest first."""
    convs = db.query(ConvModel).order_by(ConvModel.updated_at.desc()).all()
    out = []
    for c in convs:
        msg_count = db.query(MsgModel).filter(MsgModel.conversation_id == c.conversation_id).count()
        last_msg = db.query(MsgModel).filter(
            MsgModel.conversation_id == c.conversation_id
        ).order_by(MsgModel.created_at.desc()).first()
        out.append({
            "conversation_id": c.conversation_id,
            "title": c.title,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "message_count": msg_count,
            "last_message": last_msg.content[:100] if last_msg else None,
        })
    return {"conversations": out}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Get conversation with all messages."""
    conv = db.query(ConvModel).filter(ConvModel.conversation_id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.query(MsgModel).filter(
        MsgModel.conversation_id == conversation_id
    ).order_by(MsgModel.created_at.asc()).all()
    return {
        "conversation_id": conv.conversation_id,
        "title": conv.title,
        "messages": [
            {
                "message_id": m.message_id,
                "role": m.role,
                "content": m.content,
                "trace_id": m.trace_id,
                "generator_provider": m.generator_provider,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Delete a conversation and its messages."""
    db.query(MsgModel).filter(MsgModel.conversation_id == conversation_id).delete()
    db.query(ConvModel).filter(ConvModel.conversation_id == conversation_id).delete()
    db.commit()
    return {"ok": True}


@router.get("/docs")
def list_docs(db: Session = Depends(get_db)):
    """List docs with chunk counts."""
    docs = db.query(Doc).order_by(Doc.created_at.desc()).all()
    out = []
    for d in docs:
        count = db.query(Chunk).filter(Chunk.doc_id == d.doc_id).count()
        out.append({"doc_id": d.doc_id, "title": d.title, "source_type": d.source_type, "source_ref": d.source_ref, "created_at": d.created_at.isoformat() if d.created_at else None, "chunk_count": count})
    return {"docs": out}


@router.get("/analytics/failures")
def analytics_failures(db: Session = Depends(get_db)):
    """Aggregate counts by diagnosis label and top docs by failure frequency (traces with heals)."""
    import json
    from collections import Counter
    traces = db.query(Trace).all()
    label_counts = Counter()
    doc_failures = Counter()
    example_traces_by_label = {}
    for t in traces:
        attempts = json.loads(t.attempts_json) if t.attempts_json else []
        heals = json.loads(t.heals_json) if t.heals_json else []
        if not heals:
            continue
        doc_ids = set()
        for a in attempts:
            for d in a.get("diagnoses", []):
                lab = d.get("label")
                if lab:
                    label_counts[lab] += 1
                    if lab not in example_traces_by_label:
                        example_traces_by_label[lab] = t.trace_id
            for r in a.get("retrieved", []):
                did = r.get("doc_id")
                if did:
                    doc_ids.add(did)
        for did in doc_ids:
            doc_failures[did] += 1
    top_docs = doc_failures.most_common(10)
    doc_titles = {}
    for doc_id, _ in top_docs:
        d = db.query(Doc).filter(Doc.doc_id == doc_id).first()
        doc_titles[doc_id] = d.title if d else doc_id
    return {
        "by_diagnosis_label": dict(label_counts),
        "top_docs_by_failure_count": [{"doc_id": did, "title": doc_titles.get(did, did), "count": c} for did, c in top_docs],
        "example_trace_ids_by_label": example_traces_by_label,
    }
