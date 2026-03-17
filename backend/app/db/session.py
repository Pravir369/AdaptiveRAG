"""SQLite session and init (create tables if not exist; add columns if missing)."""
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from app.core import DATA_DIR
from .models import Base

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "adaptive_rag.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _add_column_if_missing(table: str, column: str, col_type: str):
    """SQLite: add column to table if not present (run after create_all)."""
    with engine.connect() as conn:
        r = conn.execute(text(f"PRAGMA table_info({table})"))
        names = [row[1] for row in r.fetchall()]
        if column not in names:
            conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {col_type}'))
            conn.commit()
            logger.info("Added column %s.%s", table, column)


def init_db():
    """Create tables if they do not exist; add new columns for self-heal v1 if missing."""
    Base.metadata.create_all(bind=engine)
    _add_column_if_missing("chunks", "section_title", "VARCHAR(512)")
    _add_column_if_missing("chunks", "chunk_type", "VARCHAR(32)")
    _add_column_if_missing("traces", "attempts_json", "TEXT")
    _add_column_if_missing("traces", "heals_json", "TEXT")
    _add_column_if_missing("traces", "final_selected_json", "TEXT")
    _add_column_if_missing("traces", "self_heal_triggered", "INTEGER")
    _add_column_if_missing("traces", "trigger_failed_thresholds_json", "TEXT")
    _add_column_if_missing("traces", "best_attempt_no", "INTEGER")
    _add_column_if_missing("traces", "generator_provider", "VARCHAR(64)")
    _add_column_if_missing("traces", "generator_metadata_json", "TEXT")
    _add_column_if_missing("traces", "generator_error", "TEXT")
    logger.info("Database initialized at %s", DB_PATH)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
