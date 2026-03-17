"""Per-query retrieval quality metrics for self-heal triggering."""
import re
import logging
from collections import Counter
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

METRICS_SCHEMA_VERSION = 1

# Stopwords for query_specificity (domain-agnostic)
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "this", "that", "these", "those", "it", "its", "as", "by", "with", "from",
    "into", "through", "during", "what", "which", "who", "when", "where", "why", "how", "explain", "summarize",
    "describe", "tell", "about", "source", "own", "words",
})


def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _query_specificity(query: str) -> float:
    """specificity = min(1.0, unique_nonstop_tokens / 6.0)."""
    tokens = re.findall(r"\w+", query.lower())
    nonstop = [t for t in tokens if t not in _STOPWORDS]
    unique = len(set(nonstop))
    return round(min(1.0, unique / 6.0), 4)


def compute_retrieval_metrics(
    top_k_results: List[Dict[str, Any]],
    query: str,
    chunk_texts: Dict[str, str],
    chunk_doc_ids: Dict[str, str],
    chunk_types: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Compute metrics from topK retrieved items.
    top_k_results: list of {chunk_id, vector_score?, lexical_score?, fused_score?, ...}
    chunk_types: optional chunk_id -> chunk_type for mean_chunk_len / mixed_chunk_types.
    """
    if not top_k_results:
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "max_vector_score": 0.0,
            "mean_vector_score_topk": 0.0,
            "max_lexical_score": 0.0,
            "mean_lexical_score_topk": 0.0,
            "max_fused_score": 0.0,
            "mean_fused_score_topk": 0.0,
            "lexical_overlap_ratio": 0.0,
            "concentration": 0.0,
            "score_separation": 0.0,
            "coverage": 0,
            "evidence_density": 0.0,
            "mean_chunk_len": 0,
            "mixed_chunk_types": False,
            "query_specificity": 0.0,
            "overlap_top3": 0.0,
            "max_vector_raw": 0.0,
            "max_lexical_raw": 0.0,
        }
    q_tokens = _tokenize(query)
    query_specificity = _query_specificity(query)

    vector_scores = [r.get("vector_score") or r.get("cosine") or r.get("score", 0.0) for r in top_k_results]
    lexical_scores = [r.get("lexical_score") or r.get("lexical", 0.0) for r in top_k_results]
    fused_scores = [r.get("fused_score") or r.get("score", 0.0) for r in top_k_results]
    vector_raw = [r.get("vector_raw", r.get("vector_score", 0.0)) for r in top_k_results]
    lexical_raw = [r.get("lexical_raw", r.get("lexical_score", 0.0)) for r in top_k_results]

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def safe_std(xs):
        if len(xs) < 2:
            return 0.0
        m = safe_mean(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        return var ** 0.5

    # If query has 0 non-stop tokens (e.g. "explain this"), set overlap_top3 to 0.0 to avoid misleading overlap
    tokens_list = re.findall(r"\w+", query.lower())
    unique_nonstop = len(set(t for t in tokens_list if t not in _STOPWORDS))
    if unique_nonstop == 0:
        overlap_top3 = 0.0
    else:
        # Overlap using ONLY top 3 chunks: |Q ∩ (T1∪T2∪T3)| / max(1, |Q|)
        top3 = top_k_results[:3]
        top3_tokens_union = set()
        for r in top_k_results:
            if r in top3:
                cid = r.get("chunk_id")
                text = chunk_texts.get(cid, "")
                top3_tokens_union |= _tokenize(text)
        overlap_top3 = len(q_tokens & top3_tokens_union) / max(1, len(q_tokens))
        overlap_top3 = round(overlap_top3, 4)
    overlap_per_chunk = []
    for r in top_k_results:
        cid = r.get("chunk_id")
        text = chunk_texts.get(cid, "")
        t_tokens = _tokenize(text)
        if q_tokens:
            overlap_per_chunk.append(len(q_tokens & t_tokens) / len(q_tokens))
    # Keep lexical_overlap_ratio = overlap_top3 for backward compat (diagnose/heal use it)
    lexical_overlap_ratio = overlap_top3

    # Concentration: fraction of topK from same doc
    doc_ids = [chunk_doc_ids.get(r.get("chunk_id"), "") for r in top_k_results]
    counts = Counter(doc_ids)
    most_common = counts.most_common(1)
    concentration = (most_common[0][1] / len(top_k_results)) if most_common and top_k_results else 0.0

    # Coverage: unique docs in topK
    coverage = len(set(d for d in doc_ids if d))

    # Evidence density: average unique token overlap per chunk
    evidence_density = safe_mean(overlap_per_chunk) if overlap_per_chunk else 0.0

    # Optional: mean chunk length and mixed chunk types (for CHUNK_TOO_COARSE diagnosis)
    mean_chunk_len = 0
    mixed_chunk_types = False
    if chunk_texts:
        lens = [len(chunk_texts.get(r.get("chunk_id"), "")) for r in top_k_results]
        mean_chunk_len = int(safe_mean(lens)) if lens else 0
    if chunk_types:
        types_in_top = [chunk_types.get(r.get("chunk_id"), "") for r in top_k_results if chunk_types.get(r.get("chunk_id"))]
        mixed_chunk_types = len(set(types_in_top)) > 1 if types_in_top else False

    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "max_vector_score": max(vector_scores) if vector_scores else 0.0,
        "mean_vector_score_topk": safe_mean(vector_scores),
        "max_lexical_score": max(lexical_scores) if lexical_scores else 0.0,
        "mean_lexical_score_topk": safe_mean(lexical_scores),
        "max_fused_score": max(fused_scores) if fused_scores else 0.0,
        "mean_fused_score_topk": safe_mean(fused_scores),
        "max_vector_raw": round(max(vector_raw), 4) if vector_raw else 0.0,
        "max_lexical_raw": round(max(lexical_raw), 4) if lexical_raw else 0.0,
        "lexical_overlap_ratio": round(lexical_overlap_ratio, 4),
        "concentration": round(concentration, 4),
        "score_separation": round(safe_std(fused_scores), 4),
        "coverage": coverage,
        "evidence_density": round(evidence_density, 4),
        "mean_chunk_len": mean_chunk_len,
        "mixed_chunk_types": mixed_chunk_types,
        "query_specificity": query_specificity,
        "overlap_top3": overlap_top3,
    }


def passes_thresholds(
    metrics: Dict[str, Any],
    min_max_fused: float,
    min_lexical_overlap: float,
    min_score_separation: float,
    min_query_specificity: float,
) -> bool:
    """True if all thresholds pass (no healing needed). Uses overlap_top3 for overlap, and query_specificity."""
    if metrics.get("max_fused_score", 0) < min_max_fused:
        return False
    if metrics.get("overlap_top3", metrics.get("lexical_overlap_ratio", 0)) < min_lexical_overlap:
        return False
    if metrics.get("score_separation", 0) < min_score_separation:
        return False
    if metrics.get("query_specificity", 1.0) < min_query_specificity:
        return False
    return True


def get_failed_thresholds(
    metrics: Dict[str, Any],
    min_max_fused: float,
    min_lexical_overlap: float,
    min_score_separation: float,
    min_query_specificity: float,
) -> List[Dict[str, Any]]:
    """Return list of {name, value, threshold} for each threshold that failed on attempt 1."""
    out: List[Dict[str, Any]] = []
    v = metrics.get("max_fused_score", 0)
    if v < min_max_fused:
        out.append({"name": "max_fused_score", "value": round(v, 4), "threshold": min_max_fused})
    v = metrics.get("overlap_top3", metrics.get("lexical_overlap_ratio", 0))
    if v < min_lexical_overlap:
        out.append({"name": "overlap_top3", "value": round(v, 4), "threshold": min_lexical_overlap})
    v = metrics.get("score_separation", 0)
    if v < min_score_separation:
        out.append({"name": "score_separation", "value": round(v, 4), "threshold": min_score_separation})
    v = metrics.get("query_specificity", 1.0)
    if v < min_query_specificity:
        out.append({"name": "query_specificity", "value": round(v, 4), "threshold": min_query_specificity})
    return out
