"""Self-healing actions: generic query rewrite, weight adjustment, K increase. No LLM."""
import re
import math
import logging
from typing import List, Dict, Any, Tuple

from app.rag.diagnose import extract_query_features

logger = logging.getLogger(__name__)

# Expanded stopwords for salient-term extraction: pronouns, time/generic, common words
_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "this", "that", "these", "those", "it", "its", "as", "by", "with", "from",
    "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "my", "your", "our", "their", "his", "her", "present", "past", "current", "general",
    "information", "content", "document", "section", "page", "part", "include", "includes", "including",
})


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _tokenize_keep_case(text: str) -> List[str]:
    """Tokenize preserving case for acronym/capital detection."""
    return re.findall(r"\w+", text)


def extract_salient_terms(chunks_texts: List[str], top_n: int = 6) -> List[str]:
    """
    Extract top_n salient keywords from top 3 chunks only. Ignores tokens len < 3 and expanded stopwords.
    Prefers acronyms/capitalized tokens and numbers/years; then TF*IDF among remaining.
    """
    if not chunks_texts or top_n <= 0:
        return []
    # Use top 3 chunks only
    chunks_texts = chunks_texts[:3]
    N = len(chunks_texts)
    doc_tokens: List[set] = []
    all_tf: List[Dict[str, int]] = []
    for text in chunks_texts:
        tokens_lower = [t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP]
        tokens_raw = _tokenize_keep_case(text)
        tf: Dict[str, int] = {}
        for t in tokens_lower:
            tf[t] = tf.get(t, 0) + 1
        all_tf.append(tf)
        doc_tokens.append(set(tf.keys()))
    # Prefer: acronyms (2–5 all-caps), capitalized, years, numbers
    preferred: List[str] = []
    for text in chunks_texts:
        for t in _tokenize_keep_case(text):
            if len(t) >= 2 and t not in _STOP:
                if 2 <= len(t) <= 5 and t.isalpha() and t.isupper():
                    preferred.append(t.lower())
                elif t[0].isupper() and len(t) >= 3:
                    preferred.append(t.lower())
                elif re.match(r"^(19|20)\d{2}$", t) or (t.isdigit() and len(t) <= 5):
                    preferred.append(t)
    # TF*IDF over tokens (len >= 3, not stop)
    df: Dict[str, int] = {}
    for doc in doc_tokens:
        for t in doc:
            df[t] = df.get(t, 0) + 1
    scores: Dict[str, float] = {}
    for tf in all_tf:
        for t, count in tf.items():
            idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0
            scores[t] = scores.get(t, 0.0) + count * idf
    # Boost preferred terms
    for p in preferred:
        scores[p] = scores.get(p, 0.0) + 2.0
    sorted_terms = sorted(scores.keys(), key=lambda x: -scores[x])
    return sorted_terms[:top_n]


def query_rewrites(
    query: str,
    query_features: Dict[str, Any],
    salient_terms: List[str] | None = None,
) -> List[str]:
    """
    Generate up to 3 domain-agnostic rewrites (no LLM).
    A) Grounding: "Based on the provided documents, <intent> with key points and important details."
    B) Structured: "Summarize the main sections/topics and their key takeaways."
    C) Keyword-anchored: original query + salient terms from top chunks (if provided).
    Entities (acronyms, numbers, quoted phrases) are incorporated when present.
    """
    rewrites: List[str] = []
    lower = query.lower().strip()
    entities = []
    if query_features.get("acronyms"):
        entities.extend(query_features["acronyms"])
    if query_features.get("numbers"):
        entities.extend(query_features["numbers"][:3])
    if query_features.get("capitalized"):
        entities.extend(query_features["capitalized"][:2])
    entity_suffix = " " + " ".join(entities) if entities else ""

    # Rewrite A: grounding
    intent = query.strip()
    if len(intent) > 80:
        intent = intent[:77] + "..."
    rewrites.append(f"Based on the provided documents, {intent} with key points and important details.{entity_suffix}".strip())
    if len(rewrites) >= 3:
        return rewrites[:3]

    # Rewrite B: structured output
    rewrites.append("Summarize the main sections/topics and their key takeaways." + entity_suffix)
    if len(rewrites) >= 3:
        return rewrites[:3]

    # Rewrite C: keyword-anchored (if we have salient terms from Attempt 1 chunks)
    if salient_terms:
        kw = " ".join(salient_terms[:6])
        rewrites.append(f"{query} {kw}".strip())
    else:
        rewrites.append(query + " key points main topics summary" + entity_suffix)
    return rewrites[:3]


def quality_score_for_attempt(
    metrics: Dict[str, Any],
) -> float:
    """Single scalar: max_fused + overlap_top3 + separation + 0.25*query_specificity (incorporates specificity when low)."""
    m = metrics.get("max_fused_score", 0) or 0
    o = metrics.get("overlap_top3", metrics.get("lexical_overlap_ratio", 0)) or 0
    s = min(1.0, (metrics.get("score_separation", 0) or 0) * 10)
    spec = metrics.get("query_specificity", 0) or 0
    return m + o + s + 0.25 * spec


def suggest_weights(
    query_features: Dict[str, Any],
    diagnoses: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """
    Suggest (w_vec, w_lex) based on query and diagnoses.
    Digits/acronyms/explicit keywords -> increase lexical.
    Conceptual/longer -> increase vector.
    """
    w_vec = 0.7
    w_lex = 0.3
    if query_features.get("has_entities") or query_features.get("acronyms") or query_features.get("numbers"):
        w_lex = 0.5
        w_vec = 0.5
    if query_features.get("is_conceptual") and query_features.get("query_length", 0) >= 4:
        w_vec = 0.8
        w_lex = 0.2
    total = w_vec + w_lex
    return w_vec / total, w_lex / total


def suggest_k_increase(current_k: int) -> int:
    """Last resort: increase K (e.g. 12 -> 20)."""
    return min(current_k + 8, 20)
