# AdaptiveRAG

A production-grade **Retrieval-Augmented Generation** system with **self-healing retrieval**, **local LLM generation**, and **multi-turn conversation memory** — fully offline, zero cloud dependencies.

Built to demonstrate advanced AI/ML systems engineering: the system automatically detects when retrieval quality degrades, diagnoses the root cause, and applies corrective actions in real-time — all without human intervention.

---

## Why This Project Is Different

Most RAG implementations are basic: embed → retrieve → generate. AdaptiveRAG goes significantly further:

| Feature | Typical RAG | AdaptiveRAG |
|---------|-------------|-------------|
| Retrieval | Single vector search | Hybrid vector + BM25 with reciprocal rank fusion |
| Failure handling | None | Self-healing loop with diagnosis + automated repair |
| Generation | Cloud API call | Local LLM (Ollama) with citation enforcement |
| Conversation | Stateless | Multi-turn memory with context-aware prompting |
| Observability | None | Full trace pipeline with per-attempt metrics |
| Query understanding | None | Intent classification, vague query detection, conversational routing |

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    AdaptiveRAG Pipeline                 │
                    └─────────────────────────────────────────────────────────┘

  User Query ──► Conversational    ──► Hybrid Retrieval  ──► Quality      ──► Self-Heal
                 Router                 (Vector + BM25)      Assessment       Loop
                   │                        │                    │              │
                   │ greeting/              │ Reciprocal         │ Metrics:     │ Diagnose:
                   │ small talk             │ Rank Fusion        │ - max_fused  │ - QUERY_TOO_VAGUE
                   ▼                        │ + Intent-Aware     │ - overlap    │ - RETRIEVAL_MISS
              Direct LLM                    │   Boosting         │ - separation │ - LOW_SPECIFICITY
              Response                      ▼                    │ - specificity│
                                     Context Selection           ▼              │ Heal Actions:
                                     (diversity +          Pass / Fail ◄────────┤ - Query Rewrite
                                      relevance                                 │ - Weight Adjust
                                      heuristics)                               │ - K Increase
                                           │                                    │
                                           ▼                                    │
                                    ┌──────────────┐         ┌─────────────┐    │
                                    │   Ollama     │◄────────│ Conversation │    │
                                    │  Generator   │         │   Memory     │    │
                                    │  (grounded)  │         │ (SQLite)     │    │
                                    └──────┬───────┘         └─────────────┘    │
                                           │                                    │
                                           ▼                                    │
                                    Citation Extraction                         │
                                    + Answer Cleaning ──► Response + Sources    │
                                                              │                 │
                                                              ▼                 ▼
                                                         Full Trace ──► Failure Analytics
                                                         (stored per              Dashboard
                                                          attempt)
```

---

## Core Technical Components

### 1. Hybrid Retrieval Engine
- **Dual-index architecture**: persistent NumPy cosine similarity index + BM25 lexical index
- **Reciprocal Rank Fusion** combines vector and lexical scores with configurable weights
- **Intent-aware post-fusion boosting** adjusts scores based on query keyword presence in chunks
- **Context selection heuristics** balance relevance, diversity, and content type coverage

### 2. Self-Healing Retrieval Loop
When retrieval quality metrics fall below thresholds, the system autonomously:
- **Diagnoses** the failure: `QUERY_TOO_VAGUE`, `RETRIEVAL_MISS`, `LOW_SPECIFICITY`, `WEAK_EVIDENCE`
- **Selects a heal action**: query rewrite (with grounding, structured, or keyword-anchored strategies), dynamic weight adjustment, or K increase
- **Re-retrieves** with the corrected parameters and measures improvement
- **Scores multiple query rewrites** and selects the one with the best quality metrics
- All attempts, diagnoses, and metric deltas are traced for full observability

### 3. Local LLM Generation (Ollama)
- **Grounded generation**: the LLM only answers from provided context chunks
- **Conversational routing**: greetings and small talk bypass retrieval entirely
- **Citation extraction**: parses model output for chunk references across multiple formats
- **Automatic fallback**: if Ollama is unavailable, falls back to extractive generation seamlessly
- **HTTP API + CLI fallback**: tries Ollama HTTP first, falls back to CLI binary

### 4. Multi-Turn Conversation Memory
- **Persistent conversations** stored in SQLite with full message history
- **Context-aware prompting**: last 4 exchanges (8 messages) are injected into the LLM prompt
- **Conversation management**: create, load, switch, and delete conversations via API and UI
- **Per-message trace linking**: every assistant response links to its retrieval trace

### 5. Observability & Analytics
- **Full trace pipeline**: every query stores retrieval results, metrics, diagnoses, heals, and generation metadata
- **Self-Heal Report UI**: visual timeline, before/after chunk comparisons, metric delta cards, rewrite scoring tables
- **Failure Analytics dashboard**: aggregate diagnosis counts, top documents by failure frequency, example trace links
- **Generator metadata**: model name, temperature, token count, citation mode tracked per response

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, SQLAlchemy, Pydantic |
| Frontend | React 18, TypeScript, Vite |
| Database | SQLite (auto-created at runtime) |
| LLM | Ollama (local, offline — qwen2.5 / llama3.1) |
| Embeddings | Hash-based (default) or sentence-transformers |
| Vector Index | NumPy cosine similarity (persistent) |
| Lexical Index | BM25 (persistent JSON) |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node 18+
- Ollama (optional — works without it using extractive generation)

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Ollama (optional)

```bash
# Install: https://ollama.com/download
ollama serve &
ollama pull qwen2.5:3b    # 8GB RAM
# or: ollama pull llama3.1:8b  # 16GB+ RAM
```

Set `GENERATION_PROVIDER=ollama` in `backend/.env`.

### One-command start

```bash
./start.sh  # Starts Ollama + backend + frontend
```

Open **http://localhost:5173**

---

## Project Structure

```
AdaptiveRAG/
├── backend/
│   ├── app/
│   │   ├── api/routes.py            # REST endpoints, conversation management, query routing
│   │   ├── core/config.py           # Environment configuration
│   │   ├── db/models.py             # SQLAlchemy models (Doc, Chunk, Trace, Conversation, Message)
│   │   ├── db/session.py            # DB initialization with auto-migration
│   │   ├── models/schemas.py        # Pydantic request/response schemas
│   │   ├── rag/
│   │   │   ├── retriever.py         # Hybrid retrieval, self-heal loop, context selection
│   │   │   ├── generator/
│   │   │   │   ├── __init__.py      # Generator interface, extractive fallback, answer builder
│   │   │   │   └── ollama.py        # Ollama client, prompt engineering, citation extraction
│   │   │   ├── diagnose.py          # Failure diagnosis engine
│   │   │   ├── heal.py              # Heal action dispatcher (rewrite, weights, K)
│   │   │   ├── metrics.py           # Quality metrics computation
│   │   │   ├── hybrid_lexical.py    # BM25 lexical index
│   │   │   └── index.py             # NumPy vector index
│   │   ├── ingest/                  # Document ingestion (file upload, URL fetch)
│   │   └── trace/store.py           # Trace persistence
│   ├── eval/                        # Evaluation harness (HitRate@10)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx                  # App shell with sidebar navigation
│       ├── api.ts                   # API client with full type definitions
│       └── pages/
│           ├── Chat.tsx             # Chat UI: conversations, message thread, citations
│           ├── Ingest.tsx           # Document ingestion (drag-drop + URL)
│           ├── Docs.tsx             # Document browser with chunk counts
│           ├── Trace.tsx            # Raw trace inspector
│           ├── TraceReport.tsx      # Self-heal report visualization
│           └── FailuresAnalytics.tsx # Failure analytics dashboard
├── start.sh                         # One-command startup script
└── .gitignore
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | RAG query with conversation memory |
| `GET` | `/api/conversations` | List all conversations |
| `GET` | `/api/conversations/:id` | Get conversation with messages |
| `DELETE` | `/api/conversations/:id` | Delete a conversation |
| `POST` | `/api/ingest/upload` | Upload and index a document |
| `POST` | `/api/ingest/url` | Fetch and index a URL |
| `GET` | `/api/docs` | List indexed documents |
| `GET` | `/api/trace/:id` | Get full retrieval trace |
| `GET` | `/api/analytics/failures` | Failure analytics aggregation |
| `GET` | `/api/health` | Health check |

---

## Self-Healing Demo

1. **Ingest a document** — upload any `.txt`, `.md`, or `.json` file
2. **Ask a vague query** — try: *"Explain this"*, *"Summarize everything"*, *"What are the key points?"*
3. **Watch self-healing activate:**
   - Attempt 1 fails quality thresholds (low overlap, vague intent)
   - System diagnoses: `QUERY_TOO_VAGUE`, `RETRIEVAL_MISS`
   - Applies heal: generates multiple query rewrites, scores each, picks the best
   - Attempt 2 runs with the optimized query
4. **Inspect the Self-Heal Report** — click the link after any answer to see:
   - Pipeline timeline visualization
   - Diagnosis details with reason strings
   - Rewrite scoring table (quality, max_fused, overlap, separation)
   - Before vs. after chunk comparison with highlighted query terms
   - Metric delta cards showing improvement

---

## Configuration

All settings via environment variables (`backend/.env`):

```bash
EMBEDDING_PROVIDER=hash          # hash (default) or sbert
GENERATION_PROVIDER=ollama       # ollama or extractive
OLLAMA_MODEL=qwen2.5:3b          # any Ollama model
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_MAX_TOKENS=512
OLLAMA_TEMPERATURE=0.5
```

| RAM | Recommended Model | Size | Notes |
|-----|-------------------|------|-------|
| 8 GB | `qwen2.5:3b` | 1.9 GB | Fast, fits in memory |
| 16 GB+ | `llama3.1:8b` | 4.9 GB | Higher quality |

---

## Without Ollama

Set `GENERATION_PROVIDER=extractive` or simply don't start Ollama. The system uses deterministic extractive generation — no LLM required. If Ollama is configured but unavailable, the system **falls back automatically** and indicates the fallback in the UI.

---

## Optional Enhancements

- **Better embeddings**: `pip install sentence-transformers` → set `EMBEDDING_PROVIDER=sbert`
- **PDF support**: `pip install pypdf` → `.pdf` accepted by upload endpoint

---

## License

MIT
