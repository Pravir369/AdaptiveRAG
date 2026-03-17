"""Small test for diagnose() rules: strong single-doc no CHUNK_TOO_COARSE; vague + weak overlap triggers QUERY_TOO_VAGUE/RETRIEVAL_MISS."""
from app.rag.diagnose import diagnose, extract_query_features


def test_single_doc_strong_metrics_no_chunk_too_coarse():
    """Single-doc query with strong metrics must NOT trigger CHUNK_TOO_COARSE (evidence = overlap + raw)."""
    metrics = {
        "max_fused_score": 0.6,
        "lexical_overlap_ratio": 0.5,
        "overlap_top3": 0.5,
        "max_vector_raw": 0.7,
        "max_lexical_raw": 1.2,
        "score_separation": 0.05,
        "concentration": 1.0,
        "coverage": 1,
        "mean_chunk_len": 1200,
        "mixed_chunk_types": True,
    }
    query_features = extract_query_features("What is the main finding in document X?")
    out = diagnose(metrics, query_features)
    labels = [d["label"] for d in out]
    assert "CHUNK_TOO_COARSE" not in labels, "Strong metrics + single doc should not trigger CHUNK_TOO_COARSE"


def test_vague_query_weak_overlap_triggers_vague_or_retrieval_miss():
    """Vague query with weak overlap should trigger QUERY_TOO_VAGUE and/or RETRIEVAL_MISS (not CHUNK_TOO_COARSE when query_specificity=0)."""
    metrics = {
        "max_fused_score": 0.15,
        "lexical_overlap_ratio": 0.08,
        "overlap_top3": 0.08,
        "max_vector_raw": 0.2,
        "max_lexical_raw": 0.1,
        "query_specificity": 0.0,
        "score_separation": 0.01,
    }
    query_features = extract_query_features("Explain the source in your own words")
    out = diagnose(metrics, query_features)
    labels = [d["label"] for d in out]
    assert query_features.get("has_generic_intent"), "Query should be detected as generic intent"
    assert "QUERY_TOO_VAGUE" in labels or "RETRIEVAL_MISS" in labels, "Should trigger QUERY_TOO_VAGUE or RETRIEVAL_MISS"
    assert "CHUNK_TOO_COARSE" not in labels, "When query_specificity=0 (vague query), prefer QUERY_TOO_VAGUE/RETRIEVAL_MISS over CHUNK_TOO_COARSE"


if __name__ == "__main__":
    test_single_doc_strong_metrics_no_chunk_too_coarse()
    test_vague_query_weak_overlap_triggers_vague_or_retrieval_miss()
    print("diagnose tests passed")
