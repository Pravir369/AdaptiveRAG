# AdaptiveRAG terminal demo. Regenerate with:  cd ~/AdaptiveRAG && vhs docs/demo.tape
#
# The tape only records; it does not set anything up. Before running it:
#   1. backend/.env  ->  EMBEDDING_PROVIDER=sbert, GENERATION_PROVIDER=ollama,
#                        OLLAMA_MODEL=llama3.2:3b, OLLAMA_MAX_TOKENS=200
#   2. uvicorn app.main:app --port 8000   (from backend/, venv active)
#   3. ingest the eval corpus:
#        for f in eval/corpus/*.txt; do curl -s -F "file=@$f" localhost:8000/api/ingest/upload >/dev/null; done
#   4. one warm-up /api/chat call so the model is resident
#   5. eval/results/stdlib_sbert_ollama.md present
#   6. curl and jq on PATH

Output docs/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1400
Set Height 520
Set Padding 14
Set Theme "Catppuccin Mocha"
Set TypingSpeed 10ms
Set Framerate 10
Set WindowBar Colorful
Set BorderRadius 8

Hide
Type "cd ~/AdaptiveRAG/backend && export PS1='$ ' && clear" Enter
Sleep 1s
Show

Type "# AdaptiveRAG: local RAG that grades its own retrieval. MiniLM embeddings + llama3.2:3b. No API keys." Enter
Sleep 1500ms

Type "# 1) A real question, worded nothing like the docs." Enter
Sleep 800ms
Type `curl -sS localhost:8000/api/chat -H 'content-type: application/json' -d '{"query":"Which function nukes a folder and everything inside it, and how do I ignore failures?","top_k":4}' | jq -rf ../docs/answer.jq --argjson d "$(curl -s localhost:8000/api/docs)"` Enter
Sleep 24s

Type "clear" Enter
Sleep 400ms
Type "# 2) A vague question. The quality checks fail, so it repairs the query and retries." Enter
Sleep 800ms
Type `curl -sS localhost:8000/api/chat -H 'content-type: application/json' -d '{"query":"Explain this","top_k":4}' | jq -r .trace_id > .demo_trace` Enter
Sleep 3500ms
Type `curl -sS localhost:8000/api/trace/$(cat .demo_trace) | jq -rf ../docs/selfheal.jq` Enter
Sleep 5s

Type "clear" Enter
Sleep 400ms
Type "# 3) The offline eval: 42 questions over 33 stdlib docs, rerunnable by anyone." Enter
Sleep 800ms
Type "cat eval/results/stdlib_sbert_ollama.md" Enter
Sleep 7s

Hide
Type "rm -f .demo_trace" Enter
Sleep 300ms
