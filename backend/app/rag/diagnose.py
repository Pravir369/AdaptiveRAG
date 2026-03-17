"""Rule-based failure diagnosis: query features + taxonomy labels with reason strings."""
import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

GENERIC_INTENT_WORDS = {"explain", "summarize", "describe", "tell", "about", "source", "own", "words", "what", "say"}
DIAGNOSIS_LABELS = [
    "QUERY_TOO_VAGUE",
    "RETRIEVAL_MISS",
    "RANKING_WEAK",
    "CHUNK_TOO_COARSE",
    "CHUNK_TOO_THIN",
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text)


def extract_query_features(query: str) -> Dict[str, Any]:
    """No LLM: tokenize, generic intent, entities (capitals, acronyms, years), lengths."""
    tokens = _tokenize(query)
    lower = [t.lower() for t in tokens]
    # Generic intent
    generic_count = sum(1 for t in lower if t in GENERIC_INTENT_WORDS)
    has_generic_intent = generic_count >= 2 or ("explain" in lower and "source" in lower)
    # Entities: capitalized, acronyms (2–5 all-caps), years (4 digits), numbers
    capitalized = [t for t in tokens if t and t[0].isupper() and not t.isupper()]
    acronyms = [t for t in tokens if 2 <= len(t) <= 5 and t.isalpha() and t.isupper()]
    years = re.findall(r"\b19\d{2}\b|\b20\d{2}\b", query)
    numbers = re.findall(r"\d+", query)
    return {
        "query_length": len(tokens),
        "unique_token_count": len(set(lower)),
        "has_generic_intent": has_generic_intent,
        "generic_intent_word_count": generic_count,
        "capitalized": capitalized,
        "acronyms": acronyms,
        "years": years,
        "numbers": numbers,
        "has_entities": bool(capitalized or acronyms or years or numbers),
        "is_conceptual": has_generic_intent and ("why" in lower or "explain" in lower or "summarize" in lower),
    }


def diagnose(
    metrics: Dict[str, Any],
    query_features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Deterministic rules -> list of {label, reason}.
    """
    out: List[Dict[str, Any]] = []

    # QUERY_TOO_VAGUE: generic intent and low overlap
    if query_features.get("has_generic_intent") and (metrics.get("lexical_overlap_ratio", 1) < 0.2):
        out.append({
            "label": "QUERY_TOO_VAGUE",
            "reason": "Query uses generic intent words (e.g. explain, summarize) and lexical overlap with top chunks is low.",
        })

    # RETRIEVAL_MISS: use raw evidence (overlap + raw maxima), not normalized max_fused
    overlap_val = metrics.get("overlap_top3", metrics.get("lexical_overlap_ratio", 0)) or 0
    max_vec_raw = metrics.get("max_vector_raw", 0) or 0
    max_lex_raw = metrics.get("max_lexical_raw", 0) or 0
    evidence_weak = overlap_val < 0.2 or (max_vec_raw < 0.3 and max_lex_raw < 0.5)
    if evidence_weak:
        out.append({
            "label": "RETRIEVAL_MISS",
            "reason": f"Evidence weak: overlap_top3={overlap_val:.3f}, max_vector_raw={max_vec_raw:.3f}, max_lexical_raw={max_lex_raw:.3f}; retrieval may have missed relevant content.",
        })
    if overlap_val < 0.1:
        out.append({
            "label": "RETRIEVAL_MISS",
            "reason": f"Lexical overlap with top chunks ({overlap_val:.3f}) is very low; few query terms appear in top chunks.",
        })

    # RANKING_WEAK: low score separation (scores too similar)
    if metrics.get("score_separation", 1) < 0.03:
        out.append({
            "label": "RANKING_WEAK",
            "reason": f"Score separation (stddev of fused scores) is {metrics.get('score_separation', 0):.3f}; ranking is weak and scores are too similar.",
        })

    # CHUNK_TOO_COARSE: only when weak evidence (overlap + raw) AND (large chunks or mixed types). Skip when query_specificity==0 (vague query -> prefer QUERY_TOO_VAGUE/RETRIEVAL_MISS).
    query_spec = metrics.get("query_specificity", 1.0) or 0
    weak_evidence = (overlap_val < 0.2) or (max_vec_raw < 0.3 and max_lex_raw < 0.5)
    large_or_mixed = (metrics.get("mean_chunk_len", 0) > 800) or metrics.get("mixed_chunk_types", False)
    if weak_evidence and large_or_mixed and query_spec > 0:
        out.append({
            "label": "CHUNK_TOO_COARSE",
            "reason": f"Weak evidence (overlap {overlap_val:.3f}, max_vector_raw {max_vec_raw:.3f}, max_lexical_raw {max_lex_raw:.3f}) with large mean chunk length ({metrics.get('mean_chunk_len', 0)}) or mixed chunk types; chunks may be too coarse.",
        })

    # CHUNK_TOO_THIN: low evidence density
    if metrics.get("evidence_density", 1) < 0.1 and query_features.get("query_length", 0) >= 3:
        out.append({
            "label": "CHUNK_TOO_THIN",
            "reason": f"Evidence density ({metrics.get('evidence_density', 0):.3f}) is low; chunks may be too thin or not matching query.",
        })

    return out
