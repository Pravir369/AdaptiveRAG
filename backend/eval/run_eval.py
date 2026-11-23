#!/usr/bin/env python3
"""Offline, reproducible retrieval eval for AdaptiveRAG. No server, no API keys, no network.

What it does:
  1. Builds a fresh index in a temporary data directory from eval/corpus/ (one doc per file).
  2. Runs every question in the eval set through four retrieval configurations:
       - bm25         BM25 over every chunk (lexical baseline)
       - vector       vector search only (w_vec=1, w_lex=0), one attempt
       - hybrid       vector + BM25 fusion (default weights), one attempt, no self-heal
       - adaptive     the full pipeline the API uses: hybrid + self-heal loop
  3. Scores each at the document level and writes JSON + a Markdown table.

Metrics (document level, top_k=12 like the API default):
  - Hit@1 / Hit@5: the expected doc is the 1st / in the first 5 distinct docs retrieved
  - MRR@10:        mean of 1/rank of the expected doc among the first 10 distinct docs
  - Answer match:  (adaptive only) the generated answer contains the expected phrase.
                   This is a strict substring check, not an LLM-judged accuracy score.
                   With --generator ollama the answer comes from the local LLM, so the
                   check is looser on wording and the run is much slower.
  - Latency:       wall-clock retrieval time per query, in process, on this machine

Usage (from backend/):
    python eval/run_eval.py                         # hash embeddings (default, offline)
    EMBEDDING_PROVIDER=sbert python eval/run_eval.py   # needs sentence-transformers
    EMBEDDING_PROVIDER=sbert python eval/run_eval.py --generator ollama   # needs a running Ollama
"""
import argparse
import json
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
DEFAULT_CORPUS = EVAL_DIR / "corpus"
DEFAULT_QA = EVAL_DIR / "stdlib_qa.jsonl"
DEFAULT_OUT = EVAL_DIR / "results"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def distinct_docs(chunk_ids, chunk_doc) -> list:
    seen, out = set(), []
    for cid in chunk_ids:
        d = chunk_doc.get(cid)
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def score_ranking(ranked_titles: list, expected: set) -> dict:
    rank = next((i + 1 for i, t in enumerate(ranked_titles[:10]) if t in expected), None)
    return {
        "hit1": bool(rank == 1),
        "hit5": bool(rank is not None and rank <= 5),
        "rr": (1.0 / rank) if rank else 0.0,
        "rank": rank,
    }


def p95(xs):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--qa", type=Path, default=DEFAULT_QA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--generator", choices=["extractive", "ollama"],
                    default=(os.getenv("GENERATION_PROVIDER") or "extractive").lower(),
                    help="answer generator used for the answer-match check (default: extractive)")
    ap.add_argument("--keep-data", action="store_true", help="keep the temporary index directory")
    args = ap.parse_args()

    # Configure the app BEFORE importing it: config is read at import time.
    data_dir = Path(tempfile.mkdtemp(prefix="adaptiverag-eval-"))
    os.environ["ADAPTIVERAG_DATA_DIR"] = str(data_dir)
    os.environ["GENERATION_PROVIDER"] = args.generator
    os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
    sys.path.insert(0, str(BACKEND_DIR))

    import logging
    logging.basicConfig(level=logging.WARNING)

    from app.core import HYBRID_W_VEC, HYBRID_W_LEX, OLLAMA_MODEL
    from app.db import init_db, Chunk, Doc
    from app.db.session import SessionLocal
    from app.ingest import ingest_upload
    from app.rag.embeddings import get_embedding_provider
    from app.rag.embeddings.base import SBERTEmbeddingProvider
    from app.rag.hybrid_lexical import get_lexical_index
    from app.rag.retriever import single_attempt, retrieve_and_rerank
    from app.rag.generator import build_answer_with_citations

    provider = get_embedding_provider()
    provider_name = "sbert" if isinstance(provider, SBERTEmbeddingProvider) else "hash"
    if os.environ["EMBEDDING_PROVIDER"].lower() == "sbert" and provider_name != "sbert":
        print("EMBEDDING_PROVIDER=sbert was requested but sentence-transformers could not load; "
              "refusing to report hash results under the sbert label.", file=sys.stderr)
        return 2

    if args.generator == "ollama":
        import requests
        from app.core import OLLAMA_BASE_URL
        try:
            tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).json()
        except Exception as e:
            print(f"--generator ollama was requested but {OLLAMA_BASE_URL} is unreachable ({e}); "
                  "refusing to report extractive results under the ollama label.", file=sys.stderr)
            return 2
        installed = {m.get("name", "") for m in tags.get("models", [])}
        if OLLAMA_MODEL not in installed:
            print(f"--generator ollama was requested but model {OLLAMA_MODEL!r} is not installed "
                  f"(have: {sorted(installed)}). Run `ollama pull {OLLAMA_MODEL}` or set OLLAMA_MODEL.",
                  file=sys.stderr)
            return 2

    init_db()
    db = SessionLocal()
    files = sorted(p for p in args.corpus.glob("*.txt"))
    if not files:
        print(f"No .txt files in {args.corpus}", file=sys.stderr)
        return 1
    for p in files:
        ok, _, msg = ingest_upload(db, p.name, p.read_bytes(), title=p.stem)
        if not ok:
            print(f"Ingest failed for {p.name}: {msg}", file=sys.stderr)
            return 1

    doc_title = {d.doc_id: d.title for d in db.query(Doc).all()}
    chunk_doc = {c.chunk_id: doc_title[c.doc_id] for c in db.query(Chunk).all()}
    n_chunks = len(chunk_doc)
    lex = get_lexical_index()

    items = load_jsonl(args.qa)
    scored_items = [it for it in items if it.get("expected_doc_titles")]
    vague_items = [it for it in items if not it.get("expected_doc_titles")]

    def run_bm25(q):
        return [h["chunk_id"] for h in lex.search_all(q, args.top_k)]

    def run_single(q, w_vec, w_lex):
        reranked, *_ = single_attempt(q, args.top_k, w_vec, w_lex, provider_name)
        return [r["chunk_id"] for r in reranked]

    configs = {
        "bm25": run_bm25,
        "vector": lambda q: run_single(q, 1.0, 0.0),
        "hybrid": lambda q: run_single(q, HYBRID_W_VEC, HYBRID_W_LEX),
    }

    per_item = []
    summary = {}
    for name, fn in configs.items():
        rows, lat = [], []
        for it in scored_items:
            t0 = time.perf_counter()
            ids = fn(it["query"])
            lat.append((time.perf_counter() - t0) * 1000)
            rows.append(score_ranking(distinct_docs(ids, chunk_doc), set(it["expected_doc_titles"])))
        summary[name] = rows, lat

    # Full adaptive pipeline (what /api/chat runs), plus extractive answers.
    rows, lat, gen_lat, answer_hits, healed = [], [], [], [], 0
    for it in scored_items:
        t0 = time.perf_counter()
        reranked, selected, chunk_texts, attempts, heals, triggered, _, best_no, mode = retrieve_and_rerank(
            it["query"], args.top_k, provider_name
        )
        lat.append((time.perf_counter() - t0) * 1000)
        healed += int(bool(triggered))
        ranked = distinct_docs([r["chunk_id"] for r in reranked], chunk_doc)
        s = score_ranking(ranked, set(it["expected_doc_titles"]))
        t_gen = time.perf_counter()
        answer, _, _ = build_answer_with_citations(
            it["query"], selected, db, chunk_texts=chunk_texts, mode=mode, provider=args.generator
        )
        gen_lat.append((time.perf_counter() - t_gen) * 1000)
        match = _norm(it["expected_answer_contains"]) in _norm(answer)
        answer_hits.append(match)
        rows.append(s)
        per_item.append({
            "id": it["id"], "style": it.get("style"), "query": it["query"],
            "expected": it["expected_doc_titles"], "top_docs": ranked[:5], "rank": s["rank"],
            "self_heal_triggered": bool(triggered), "attempts": len(attempts),
            "heals": [h.get("type") for h in heals], "answer_match": match,
            "answer": answer,
        })
    summary["adaptive"] = rows, lat

    vague_triggered = 0
    for it in vague_items:
        *_, triggered, _, _, _ = retrieve_and_rerank(it["query"], args.top_k, provider_name)
        vague_triggered += int(bool(triggered))

    def agg(rows, lat, subset=None):
        idx = range(len(rows)) if subset is None else subset
        r = [rows[i] for i in idx]
        n = len(r) or 1
        return {
            "n": len(r),
            "hit_at_1": round(sum(x["hit1"] for x in r) / n, 3),
            "hit_at_5": round(sum(x["hit5"] for x in r) / n, 3),
            "mrr_at_10": round(sum(x["rr"] for x in r) / n, 3),
            "latency_ms_p50": round(statistics.median([lat[i] for i in idx]), 1) if r else 0.0,
            "latency_ms_p95": round(p95([lat[i] for i in idx]), 1) if r else 0.0,
        }

    paraphrase_idx = [i for i, it in enumerate(scored_items) if it.get("style") == "paraphrase"]
    keyword_idx = [i for i, it in enumerate(scored_items) if it.get("style") == "keyword"]
    results = {
        "embedding_provider": provider_name,
        "generator": args.generator,
        "generator_model": OLLAMA_MODEL if args.generator == "ollama" else None,
        "python": platform.python_version(),
        "corpus": {"path": str(args.corpus.relative_to(BACKEND_DIR)) if args.corpus.is_relative_to(BACKEND_DIR) else str(args.corpus),
                   "documents": len(files), "chunks": n_chunks},
        "questions": {"scored": len(scored_items), "paraphrase": len(paraphrase_idx),
                      "keyword": len(keyword_idx), "vague_unscored": len(vague_items)},
        "top_k": args.top_k,
        "configs": {name: {"all": agg(r, l), "paraphrase": agg(r, l, paraphrase_idx), "keyword": agg(r, l, keyword_idx)}
                    for name, (r, l) in summary.items()},
        "adaptive_extras": {
            "answer_match_rate": round(sum(answer_hits) / max(1, len(answer_hits)), 3),
            "answer_latency_ms_p50": round(statistics.median(gen_lat), 1) if gen_lat else 0.0,
            "answer_latency_ms_p95": round(p95(gen_lat), 1) if gen_lat else 0.0,
            "self_heal_triggered_on_scored": healed,
            "self_heal_triggered_on_vague": f"{vague_triggered}/{len(vague_items)}",
        },
        "per_item": per_item,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"stdlib_{provider_name}" + ("_ollama" if args.generator == "ollama" else "")
    (args.out / f"{stem}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(results)
    (args.out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)

    db.close()
    if not args.keep_data:
        shutil.rmtree(data_dir, ignore_errors=True)
    return 0


def render_markdown(r: dict) -> str:
    q = r["questions"]
    lines = [
        f"Corpus: {r['corpus']['documents']} Python stdlib docstrings ({r['corpus']['chunks']} chunks). "
        f"{q['scored']} scored questions ({q['paraphrase']} paraphrased, {q['keyword']} keyword). "
        f"Embeddings: `{r['embedding_provider']}`. Generator: `{r['generator']}`"
        + (f" ({r['generator_model']})" if r.get("generator_model") else "")
        + f". top_k={r['top_k']}. Python {r['python']}.",
        "",
        "| Retrieval | Hit@1 | Hit@5 | MRR@10 | Hit@1 (paraphrased) | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {"bm25": "BM25 only", "vector": "Vector only", "hybrid": "Hybrid (no self-heal)",
              "adaptive": "Hybrid + self-heal (full pipeline)"}
    for name, c in r["configs"].items():
        a, p = c["all"], c["paraphrase"]
        lines.append(f"| {labels.get(name, name)} | {a['hit_at_1']:.2f} | {a['hit_at_5']:.2f} | {a['mrr_at_10']:.2f} | "
                     f"{p['hit_at_1']:.2f} | {a['latency_ms_p50']:.1f} ms | {a['latency_ms_p95']:.1f} ms |")
    x = r["adaptive_extras"]
    lines += [
        "",
        f"Full pipeline answer match ({r['generator']} answer contains the expected phrase): "
        f"{x['answer_match_rate']:.2f} "
        f"(answer latency p50 {x['answer_latency_ms_p50']:.0f} ms, p95 {x['answer_latency_ms_p95']:.0f} ms). "
        f"Self-heal triggered on {x['self_heal_triggered_on_scored']} of {q['scored']} scored questions "
        f"and {x['self_heal_triggered_on_vague']} vague ones.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
