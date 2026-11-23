"""End-to-end through the HTTP API: ingest -> chat -> trace, with the offline defaults."""
from fastapi.testclient import TestClient

from app.main import app

DOCS = {
    "rmtree.txt": "shutil.rmtree\n\nRecursively delete a directory tree. If ignore_errors is set, errors are ignored.",
    "median.txt": "statistics.median\n\nReturn the median (middle value) of numeric data.",
    "dumps.txt": "json.dumps\n\nSerialize obj to a JSON formatted str. If indent is set, output is pretty-printed.",
}


def _client():
    return TestClient(app)


def test_health():
    with _client() as c:
        assert c.get("/api/health").json() == {"status": "ok"}


def test_ingest_chat_and_trace():
    with _client() as c:
        ids = {}
        for name, text in DOCS.items():
            r = c.post("/api/ingest/upload", files={"file": (name, text.encode(), "text/plain")})
            assert r.status_code == 200, r.text
            ids[name] = r.json()["doc_id"]

        r = c.post("/api/chat", json={"query": "How do I recursively delete a directory tree?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"]
        assert body["citations"], "answer should cite at least one chunk"
        assert body["citations"][0]["doc_id"] == ids["rmtree.txt"]
        assert body["generator_provider"] == "extractive"

        t = c.get(f"/api/trace/{body['trace_id']}")
        assert t.status_code == 200
        assert ids["rmtree.txt"] in t.json()["retrieved_doc_ids"]


def test_unsupported_upload_is_rejected():
    with _client() as c:
        r = c.post("/api/ingest/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
        assert r.status_code == 400


def test_conversation_memory_is_kept():
    with _client() as c:
        c.post("/api/ingest/upload", files={"file": ("m.txt", DOCS["median.txt"].encode(), "text/plain")})
        first = c.post("/api/chat", json={"query": "What does statistics median return?"}).json()
        conv = first["conversation_id"]
        c.post("/api/chat", json={"query": "And for numeric data?", "conversation_id": conv})
        msgs = c.get(f"/api/conversations/{conv}").json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
