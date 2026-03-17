"""Pydantic request/response schemas."""
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=12, ge=1, le=100)
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    snippet: str


class MessageItem(BaseModel):
    message_id: str
    role: str
    content: str
    trace_id: Optional[str] = None
    generator_provider: Optional[str] = None
    created_at: Optional[str] = None


class ConversationItem(BaseModel):
    conversation_id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int = 0
    last_message: Optional[str] = None


class ConversationDetail(BaseModel):
    conversation_id: str
    title: str
    messages: List[MessageItem] = []


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    trace_id: str
    generator_provider: Optional[str] = None
    conversation_id: str


class IngestURLRequest(BaseModel):
    url: str
    title: Optional[str] = None


class TraceResponse(BaseModel):
    trace_id: str
    created_at: Optional[str] = None
    query: str
    embedding_provider: Optional[str] = None
    top_k: Optional[int] = None
    retrieved: List[Any] = []
    retrieved_doc_ids: List[str] = []
    reranked: List[Any] = []
    selected: List[Any] = []
    citations: List[Any] = []
    answer: Optional[str] = None
    latency_ms: Optional[int] = None
    attempts: List[Any] = []
    heals_applied: List[Any] = []
    final_selected: Optional[List[Any]] = None
    self_heal_triggered: bool = False
    trigger_failed_thresholds: List[Any] = []
    best_attempt_no: int = 1
    generator_provider: Optional[str] = None
    generator_metadata: Optional[Dict[str, Any]] = None
    generator_error: Optional[str] = None
