"""AdaptiveRAG v0 FastAPI app. CORS enabled; create tables on startup."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import CORS_ORIGINS
from app.db import init_db
from app.api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    # shutdown if needed
    pass


app = FastAPI(title="AdaptiveRAG v0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api", tags=["api"])


@app.get("/")
def root():
    return {"service": "AdaptiveRAG v0", "docs": "/docs"}
