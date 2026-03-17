#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_BIN="$SCRIPT_DIR/backend/bin/ollama"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "=== AdaptiveRAG with Ollama ==="

# 1. Start Ollama if not already running
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[✓] Ollama already running on :11434"
else
    if [ -x "$OLLAMA_BIN" ]; then
        echo "[*] Starting Ollama server..."
        "$OLLAMA_BIN" serve &
        OLLAMA_PID=$!
        sleep 3
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "[✓] Ollama started (PID $OLLAMA_PID)"
        else
            echo "[!] Ollama failed to start. Backend will use extractive fallback."
        fi
    else
        echo "[!] Ollama binary not found at $OLLAMA_BIN"
        echo "    Download: curl -fsSL -o backend/bin/ollama-darwin.tgz https://github.com/ollama/ollama/releases/download/v0.18.0/ollama-darwin.tgz"
        echo "    Extract:  cd backend/bin && tar xzf ollama-darwin.tgz"
        echo "    Backend will use extractive fallback."
    fi
fi

# 2. Check model
if curl -s http://localhost:11434/api/tags 2>/dev/null | grep -q "llama3.1:8b"; then
    echo "[✓] Model llama3.1:8b available"
else
    echo "[!] Model llama3.1:8b not found. Pull it: $OLLAMA_BIN pull llama3.1:8b"
fi

# 3. Start backend
echo "[*] Starting FastAPI backend..."
cd "$BACKEND_DIR"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "[✓] Backend starting (PID $BACKEND_PID)"

# 4. Start frontend
echo "[*] Starting React frontend..."
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
echo "[✓] Frontend starting (PID $FRONTEND_PID)"

echo ""
echo "=== Services ==="
echo "  Backend:  http://localhost:8000/api/health"
echo "  Frontend: http://localhost:5173"
echo "  Ollama:   http://localhost:11434"
echo ""
echo "Press Ctrl+C to stop all services."

trap "kill $OLLAMA_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
