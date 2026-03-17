# AdaptiveRAG v0 + Self-Healing RAG v1

Baseline RAG with full tracing and **self-healing v1**: when retrieval quality metrics fall below thresholds, the system diagnoses failures and applies healing actions (query rewrite, weight adjustment, K increase). Runs locally with **no paid API keys**. Default: SQLite + deterministic hash embeddings + NumPy cosine index + BM25 lexical index.

- **Backend:** FastAPI (Python 3.11+)
- **Frontend:** React + Vite
- **DB:** SQLite (created at runtime under `backend/data/`)
- **Embeddings:** `hash` (default) or optional `sbert` via env
- **Vector index:** Persistent NumPy cosine index in `backend/data/index/`
- **Lexical index:** BM25 in `backend/data/index/lexical_index.json`
- **Self-heal:** Threshold-based triggering; rule-based diagnosis; no LLM
- **Generation:** Ollama (local LLM) or extractive (no LLM)

---

## Prerequisites

- **Python 3.11+**
- **Node 18+**
- No Postgres or FAISS required.

---

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # optional
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

- App: [http://localhost:5173](http://localhost:5173) (Vite proxies `/api` to backend).

---

## Ingest sample docs and smoke test

1. Start backend and frontend (see above).
2. Open **Ingest**: upload a `.txt` or `.md` file, or paste a URL and click "Ingest URL".
3. Open **Chat**: ask a question; you should get an answer, citations, and a "Open trace" link.
4. Open **Docs**: confirm the document and chunk count.

**CLI smoke test (no frontend):**

```bash
# From repo root, with backend running
curl -X POST http://localhost:8000/api/ingest/url -H "Content-Type: application/json" -d '{"url":"https://example.com"}'
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"query":"What is this about?","top_k":5}'
```

---

## Quick Self-Heal Demo

Self-healing triggers only when retrieval quality metrics fall below thresholds (e.g. vague query + low lexical overlap). To see it in action:

1. **Ingest a document:** In the app, go to **Ingest** and upload any `.txt` or `.md` file (or use "Ingest URL").
2. **Ask a vague query:** In **Chat**, try: **"Explain this"**, **"Summarize the source"**, or **"What are the key points?"**.
3. **Observe:**  
   - The first attempt may fail thresholds (generic intent, low overlap).  
   - The system diagnoses (e.g. `QUERY_TOO_VAGUE`, `RETRIEVAL_MISS`) and applies a heal (e.g. **QUERY_REWRITE** with grounding, structured, or keyword-anchored rewrites).  
   - A second attempt runs with the chosen rewrite; metrics (overlap, max fused, separation) can improve.
4. **Inspect the report:** After the answer, click **Self-Heal Report** (or open the trace and click **Self-Heal Report**). You'll see:  
   - **Timeline:** Input → Attempt 1 → Heal → Attempt 2 → Final  
   - **Diagnoses** with reason strings  
   - **Heals applied** (rewrites tried, chosen rewrite, weight changes, or K increase)  
   - **Before vs After** tables of top retrieved chunks and **metric deltas**  
   - Query terms highlighted in snippets  
5. **Failures analytics:** Open **Failures** in the nav to see aggregate counts by diagnosis label and top docs by failure frequency, with links to example traces.

---

## Eval harness

From repo root or `backend`:

```bash
cd backend
# Ensure backend is running on port 8000
python eval/eval_runner.py
```

- Reads `eval/sample_eval.jsonl` (one JSON object per line: `query`, `expected_doc_ids`, optional `id`).
- Calls `POST /api/chat` for each query, then `GET /api/trace/{trace_id}`.
- Computes **HitRate@10** using `expected_doc_ids` vs trace `retrieved_doc_ids`.
- Writes `eval/eval_results.json`: overall metrics + per-item results (hit/miss, trace_id).

Schema: see `backend/eval/schema.md`. Fill `expected_doc_ids` with doc IDs that should appear in the retrieved set for your test corpus.

---

## Local LLM generation (Ollama)

The app uses a **local LLM** via [Ollama](https://ollama.com) for grounded answers with citations — fully offline, no cloud calls.

### How it works

1. **Retrieval** (hybrid vector + BM25 lexical) finds the most relevant document chunks
2. **Self-healing** diagnoses and fixes retrieval quality issues (query rewrite, weight adjustment, K increase)
3. **Ollama generation** produces a concise, grounded answer citing specific chunks
4. **Citation enforcement** validates that every claim in the output references a real retrieved chunk

### Setup

**Option A — Bundled binary (recommended for macOS):**

The project includes an Ollama binary at `backend/bin/ollama`. If not present, download it:

```bash
cd backend/bin
curl -fsSL -o ollama-darwin.tgz https://github.com/ollama/ollama/releases/download/v0.18.0/ollama-darwin.tgz
tar xzf ollama-darwin.tgz
```

**Option B — System install:** [https://ollama.com/download](https://ollama.com/download)

### Pull a model and start

```bash
# Start the Ollama server
backend/bin/ollama serve &

# Pull a model (choose based on your RAM):
#   8GB RAM:  qwen2.5:3b  (~2 GB, fast)
#   16GB RAM: llama3.1:8b (~5 GB, best quality)
backend/bin/ollama pull qwen2.5:3b
```

### Configure

In `backend/.env`:

```
GENERATION_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_MAX_TOKENS=512
OLLAMA_TEMPERATURE=0.2
```

### Quick start (all services)

```bash
./start.sh
```

This starts Ollama, the backend, and the frontend in one command.

### Without Ollama

Set `GENERATION_PROVIDER=extractive` (or just don't start Ollama). The app uses deterministic extractive generation — no LLM needed. If Ollama is configured but unavailable, the app **falls back automatically** and shows "Ollama unavailable, used extractive fallback" in the UI.

### Model recommendations by RAM

| RAM | Model | Size | Speed |
|-----|-------|------|-------|
| 8 GB | `qwen2.5:3b` | 1.9 GB | ~2 tok/s |
| 16 GB | `llama3.1:8b` | 4.9 GB | ~15 tok/s |
| 32 GB | `llama3.1:8b` | 4.9 GB | ~30 tok/s |

### Troubleshooting

- **Check Ollama is running:** `curl http://localhost:11434/api/tags`
- **List models:** `backend/bin/ollama list`
- **MLX crash on macOS:** The Ollama binary must run outside sandboxed environments (needs Metal GPU access). Run it in a normal terminal.
- **Slow generation:** With only 8GB RAM and a large model, the system swaps to disk. Use a smaller model (`qwen2.5:3b`).

---

## Optional

- **sentence-transformers:** `pip install sentence-transformers` and set `EMBEDDING_PROVIDER=sbert` in `.env`.
- **PDF upload:** `pip install pypdf`; then `.pdf` is accepted by `POST /api/ingest/upload`.

---

## Architecture

```
Query → Hybrid Retrieval (Vector + BM25) → Self-Heal Loop → Context Selection → Ollama/Extractive Generator → Answer + Citations
                                               ↓
                                    Metrics → Diagnosis → Heal Actions
                                    (overlap, fused score, specificity)
                                               ↓
                                    Query Rewrite / Weight Adjust / K Increase
```

**Key files:**
- `backend/app/rag/retriever.py` — Hybrid retrieval, self-heal loop, context selection
- `backend/app/rag/generator/ollama.py` — Ollama HTTP/CLI client, prompt construction, citation parsing
- `backend/app/rag/generator/__init__.py` — Generator interface, extractive fallback, answer builder
- `backend/app/api/routes.py` — FastAPI endpoints, orchestration, trace storage
- `frontend/src/pages/Chat.tsx` — Chat UI with generator status display
- `frontend/src/pages/TraceReport.tsx` — Self-heal report with generator metadata
