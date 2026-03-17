"""SQLite models: docs, chunks, traces."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.dialects.sqlite import CHAR
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def gen_uuid():
    return str(uuid.uuid4())


class Doc(Base):
    __tablename__ = "docs"
    doc_id = Column(CHAR(36), primary_key=True, default=gen_uuid)
    title = Column(String(1024), nullable=False, default="")
    source_type = Column(String(32), nullable=False)  # 'upload' | 'url'
    source_ref = Column(String(2048), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Chunk(Base):
    __tablename__ = "chunks"
    chunk_id = Column(CHAR(36), primary_key=True, default=gen_uuid)
    doc_id = Column(CHAR(36), ForeignKey("docs.doc_id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    text_hash = Column(String(64), nullable=True)
    section_title = Column(String(512), nullable=True)
    chunk_type = Column(String(32), nullable=True)  # 'heading' | 'bullets' | 'paragraph'
    created_at = Column(DateTime, default=datetime.utcnow)


class Trace(Base):
    __tablename__ = "traces"
    trace_id = Column(CHAR(36), primary_key=True, default=gen_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
    query = Column(Text, nullable=False)
    embedding_provider = Column(String(64), nullable=True)
    top_k = Column(Integer, nullable=True)
    retrieved_json = Column(Text, nullable=True)
    reranked_json = Column(Text, nullable=True)
    selected_json = Column(Text, nullable=True)
    citations_json = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    attempts_json = Column(Text, nullable=True)
    heals_json = Column(Text, nullable=True)
    final_selected_json = Column(Text, nullable=True)
    self_heal_triggered = Column(Boolean, nullable=True)
    trigger_failed_thresholds_json = Column(Text, nullable=True)
    best_attempt_no = Column(Integer, nullable=True)
    generator_provider = Column(String(64), nullable=True)
    generator_metadata_json = Column(Text, nullable=True)
    generator_error = Column(Text, nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id = Column(CHAR(36), primary_key=True, default=gen_uuid)
    title = Column(String(512), nullable=False, default="New conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    message_id = Column(CHAR(36), primary_key=True, default=gen_uuid)
    conversation_id = Column(CHAR(36), ForeignKey("conversations.conversation_id"), nullable=False)
    role = Column(String(16), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    trace_id = Column(CHAR(36), nullable=True)
    generator_provider = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
