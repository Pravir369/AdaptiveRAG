#!/usr/bin/env python3
"""
Eval harness: read JSONL, call /api/chat per query, fetch trace, compute HitRate@10.
Writes eval_results.json with overall metrics + per-item results.
"""
import json
import logging
import os
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default backend URL when running from repo root or backend/eval
BASE_URL = os.getenv("ADAPTIVERAG_API_URL", "http://localhost:8000")
EVAL_JSONL = Path(__file__).resolve().parent / "sample_eval.jsonl"
OUTPUT_JSON = Path(__file__).resolve().parent / "eval_results.json"


def load_eval_jsonl(path: Path) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def run_eval():
    if not EVAL_JSONL.exists():
        logger.error("Eval file not found: %s", EVAL_JSONL)
        sys.exit(1)
    items = load_eval_jsonl(EVAL_JSONL)
    results = []
    for item in items:
        query = item.get("query", "")
        expected_doc_ids = set(item.get("expected_doc_ids") or [])
        example_id = item.get("id", "")
        try:
            r = requests.post(
                f"{BASE_URL}/api/chat",
                json={"query": query, "top_k": 12},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            trace_id = data.get("trace_id")
        except Exception as e:
            logger.warning("Chat failed for %s: %s", example_id or query[:50], e)
            results.append({
                "id": example_id,
                "query": query,
                "hit": False,
                "trace_id": None,
                "error": str(e),
            })
            continue
        try:
            tr = requests.get(f"{BASE_URL}/api/trace/{trace_id}", timeout=10)
            tr.raise_for_status()
            trace = tr.json()
        except Exception as e:
            logger.warning("Trace fetch failed: %s", e)
            results.append({
                "id": example_id,
                "query": query,
                "hit": False,
                "trace_id": trace_id,
                "error": str(e),
            })
            continue
        retrieved_doc_ids = set(trace.get("retrieved_doc_ids") or [])
        # Hit if any expected doc is in the retrieved set (top-k retrieved docs)
        hit = bool(expected_doc_ids and (expected_doc_ids & retrieved_doc_ids))
        if not expected_doc_ids:
            hit = None  # no ground truth to score
        results.append({
            "id": example_id,
            "query": query,
            "expected_doc_ids": list(expected_doc_ids),
            "retrieved_doc_ids": list(retrieved_doc_ids),
            "hit": hit,
            "trace_id": trace_id,
        })
    # HitRate@10: fraction of items with at least one expected_doc_id in retrieved_doc_ids
    scored = [r for r in results if r.get("hit") is not None]
    hits = sum(1 for r in scored if r.get("hit") is True)
    total = len(scored) if scored else 0
    hit_rate_at_10 = (hits / total) if total else 0.0
    out = {
        "overall": {
            "hit_rate_at_10": hit_rate_at_10,
            "total_queries": len(items),
            "scored_queries": total,
            "hits": hits,
        },
        "per_item": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("HitRate@10: %s (scored %d)", hit_rate_at_10, total)
    logger.info("Results written to %s", OUTPUT_JSON)


if __name__ == "__main__":
    run_eval()
