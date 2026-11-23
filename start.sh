#!/usr/bin/env bash
# One-command dev start: Ollama (if installed) + FastAPI backend + Vite frontend.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Model to check for: OLLAMA_MODEL from backend/.env, else the env, else qwen2.5:3b.
if [ -f "$BACKEND_DIR/.env" ]; then
    MODEL_FROM_ENV="$(grep -E '^OLLAMA_MODEL=' "$BACKEND_DIR/.env" | tail -1 | cut -d= -f2-)"
fi
OLLAMA_MODEL="${MODEL_FROM_ENV:-${OLLAMA_MODEL:-qwen2.5:3b}}"

# Prefer an ollama on PATH; fall back to a binary dropped in backend/bin.
OLLAMA_BIN="$(command -v ollama || true)"
[ -z "$OLLAMA_BIN" ] && [ -x "$BACKEND_DIR/bin/ollama" ] && OLLAMA_BIN="$BACKEND_DIR/bin/ollama"

echo "=== AdaptiveRAG ==="

if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[ok] Ollama already running on :11434"
elif [ -n "$OLLAMA_BIN" ]; then
    echo "[..] Starting Ollama server..."
    "$OLLAMA_BIN" serve >/dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "[ok] Ollama started (PID $OLLAMA_PID)"
    else
        echo "[!] Ollama failed to start. The backend will use extractive answers."
    fi
else
    echo "[!] Ollama not found (optional). Install from https://ollama.com/download"
    echo "    The backend will use extractive answers."
fi

if curl -s http://localhost:11434/api/tags 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
    echo "[ok] Model $OLLAMA_MODEL available"
elif [ -n "$OLLAMA_BIN" ]; then
    echo "[!] Model $OLLAMA_MODEL not pulled yet: $OLLAMA_BIN pull $OLLAMA_MODEL"
fi

echo "[..] Starting FastAPI backend on :8000"
cd "$BACKEND_DIR"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

echo "[..] Starting React frontend on :5173"
cd "$FRONTEND_DIR"
[ -d node_modules ] || npm install
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8000/api/health"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop."

trap 'kill ${OLLAMA_PID:-} $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT TERM
wait
