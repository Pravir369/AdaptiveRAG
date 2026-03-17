"""Hybrid retrieval (vector + lexical), metrics, threshold check, and self-heal loop."""
import re
import logging
from typing import List, Dict, Any, Tuple

# Intent-aware boost: prioritize achievement-related chunks over generic activity (post-fusion, before context selection)
ACHIEVEMENT_TOKENS = frozenset({
    "award", "awards", "won", "win", "winner", "place", "1st", "2nd", "3rd",
    "first", "second", "third", "champion",
})
# Generic verbs we do not use for boosting (ignore in intent boost)
GENERIC_VERBS = frozenset({
    "created", "worked", "managed", "designed", "developed", "led", "built",
    "implemented", "supported", "coordinated", "handled", "responsible",
})
INTENT_BOOST_ACHIEVEMENT = 0.15  # per achievement match, cap 3
INTENT_BOOST_ORG = 0.08         # per org (ALL CAPS 2-5) match, cap 3
INTENT_BOOST_TOTAL_CAP = 0.45

# Contact-block detection for context selection (penalize unless query asks for contact)
CONTACT_QUERY_TOKENS = frozenset({"contact", "email", "phone", "address", "reach", "call", "e-mail"})
EMAIL_PHONE_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b|\b(?:\+\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{2,4}[- ]?\d{2,4}\b")

# Generic Summary Mode: when query_specificity below this, treat as vague (no keyword-anchored rewrite C; diversity selection)
VAGUE_QUERY_SPECIFICITY_THRESHOLD = 0.25
VAGUE_SUMMARY_MAX_CHUNKS = 5
CHUNK_TYPE_ORDER = {"heading": 0, "list": 1, "bullets": 1, "paragraph": 2}

from app.db import Chunk
from app.db.session import SessionLocal
from app.rag.index import get_vector_index
from app.rag.embeddings import get_embedding_provider
from app.rag.hybrid_lexical import get_lexical_index, normalize_scores_to_unit
from app.rag.metrics import compute_retrieval_metrics, passes_thresholds, get_failed_thresholds
from app.rag.diagnose import extract_query_features, diagnose
from app.rag.heal import query_rewrites, quality_score_for_attempt, suggest_weights, suggest_k_increase, extract_salient_terms
from app.core import (
    MIN_MAX_FUSED,
    MIN_LEXICAL_OVERLAP,
    MIN_SCORE_SEPARATION,
    MIN_QUERY_SPECIFICITY,
    MAX_SELF_HEAL_ATTEMPTS,
    DEFAULT_TOP_K,
    HYBRID_W_VEC,
    HYBRID_W_LEX,
)

logger = logging.getLogger(__name__)


def _get_chunk_texts_and_doc_ids(chunk_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    db = SessionLocal()
    try:
        chunk_texts = {}
        chunk_doc_ids = {}
        for c in db.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids)).all():
            chunk_texts[c.chunk_id] = c.text
            chunk_doc_ids[c.chunk_id] = c.doc_id
        return chunk_texts, chunk_doc_ids
    finally:
        db.close()


def _enrich_with_section_and_type(chunk_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    db = SessionLocal()
    try:
        section_title = {}
        chunk_type = {}
        for c in db.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids)).all():
            section_title[c.chunk_id] = c.section_title or ""
            chunk_type[c.chunk_id] = c.chunk_type or ""
        return section_title, chunk_type
    finally:
        db.close()


def hybrid_retrieve(
    query: str,
    top_k: int,
    w_vec: float,
    w_lex: float,
    embedding_provider_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """
    Vector + lexical fusion. Returns (reranked list with vector_score, lexical_score, fused_score), chunk_texts, chunk_doc_ids.
    """
    provider = get_embedding_provider()
    vindex = get_vector_index()
    lex_index = get_lexical_index()
    # Over-fetch from vector
    fetch_k = min(top_k * 3, 50)
    if len(vindex.chunk_ids()) == 0:
        return [], {}, {}
    q_emb = provider.embed_query(query)
    vector_hits = vindex.search(q_emb, top_k=fetch_k)
    if not vector_hits:
        return [], {}, {}
    candidate_ids = [r["chunk_id"] for r in vector_hits]
    chunk_texts, chunk_doc_ids = _get_chunk_texts_and_doc_ids(candidate_ids)
    # Lexical scores for candidates (raw BM25; may be empty if lexical index empty)
    lex_scores = lex_index.score_chunks(query, candidate_ids) if lex_index.chunk_ids() else {cid: 0.0 for cid in candidate_ids}
    # Retain raw scores before normalization for metrics/diagnosis
    v_max = max((r["score"] for r in vector_hits), default=1.0) or 1.0
    for r in vector_hits:
        r["vector_raw"] = r["score"]
        r["vector_score"] = r["score"] / v_max
    lex_list = [{"chunk_id": cid, "score": lex_scores.get(cid, 0)} for cid in candidate_ids]
    for x in lex_list:
        x["raw"] = x["score"]
    normalize_scores_to_unit(lex_list, key="score")
    lex_norm = {r["chunk_id"]: r["score"] for r in lex_list}
    lex_raw = {r["chunk_id"]: r["raw"] for r in lex_list}
    # Fuse
    fused = []
    for r in vector_hits:
        cid = r["chunk_id"]
        vs = r["vector_score"]
        ls = lex_norm.get(cid, 0.0)
        r["lexical_score"] = ls
        r["lexical_raw"] = lex_raw.get(cid, 0.0)
        r["fused_score"] = w_vec * vs + w_lex * ls
        fused.append(r)
    fused.sort(key=lambda x: -x["fused_score"])
    top = fused[:top_k]
    # Intent-aware boost: achievement + org tokens from query (rewrite); ignore generic verbs
    q_tokens_raw = re.findall(r"\w+", query)
    q_tokens_lower = set(t.lower() for t in q_tokens_raw)
    achievement_tokens = q_tokens_lower & ACHIEVEMENT_TOKENS
    org_tokens = set(
        t for t in q_tokens_raw
        if 2 <= len(t) <= 5 and t.isalpha() and t.isupper() and t.lower() not in GENERIC_VERBS
    )
    if (achievement_tokens or org_tokens) and chunk_texts:
        for r in top:
            text_lower = chunk_texts.get(r["chunk_id"], "").lower()
            chunk_tokens = set(re.findall(r"\w+", text_lower))
            achievement_matches = min(len(achievement_tokens & chunk_tokens), 3)
            org_matches = min(sum(1 for o in org_tokens if o.lower() in chunk_tokens), 3)
            boost = min(
                INTENT_BOOST_ACHIEVEMENT * achievement_matches + INTENT_BOOST_ORG * org_matches,
                INTENT_BOOST_TOTAL_CAP,
            )
            r["fused_score"] = r["fused_score"] + boost
        top.sort(key=lambda x: -x["fused_score"])
    return top, chunk_texts, chunk_doc_ids


def select_chunks_for_context(
    reranked: List[Dict[str, Any]],
    chunk_texts: Dict[str, str],
    query: str,
    query_specificity: float | None = None,
    overlap_top3: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Deterministic coverage heuristic: prioritize chunks with highest overlap with query tokens;
    penalize contact blocks (email/phone) unless query mentions contact/email/phone.
    If query_specificity > 0 and overlap_top3 > 0.5, restrict to top 3 chunks only (strong rewrite).
    Returns list of {chunk_id, score} for generator; citations still use chunk_id.
    """
    if not reranked:
        return []
    if query_specificity is not None and overlap_top3 is not None and query_specificity > 0 and overlap_top3 > 0.5:
        reranked = reranked[:3]
    q_tokens = set(re.findall(r"\w+", query.lower()))
    query_wants_contact = bool(q_tokens & CONTACT_QUERY_TOKENS)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in reranked:
        cid = r["chunk_id"]
        base_score = r.get("fused_score", 0.0)
        text = chunk_texts.get(cid, "")
        chunk_tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(q_tokens & chunk_tokens) / max(1, len(q_tokens))
        contact_penalty = 0.0
        if not query_wants_contact and EMAIL_PHONE_RE.search(text):
            contact_penalty = 0.25
        selection_score = base_score + 0.15 * overlap - contact_penalty
        scored.append((selection_score, {"chunk_id": cid, "score": r.get("fused_score", base_score)}))
    scored.sort(key=lambda x: -x[0])
    return [item[1] for item in scored]


def select_chunks_for_vague_summary(
    reranked: List[Dict[str, Any]],
    chunk_texts: Dict[str, str],
    query: str,
    chunk_doc_ids: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    For vague queries: select up to N=5 chunks maximizing diversity.
    Prefer different section_title/doc_id; prefer chunk_type heading > list > paragraph;
    strong penalty to contact-heavy chunks unless query mentions contact.
    """
    if not reranked:
        return []
    chunk_ids = [r["chunk_id"] for r in reranked]
    section_title, chunk_type = _enrich_with_section_and_type(chunk_ids)
    q_tokens = set(re.findall(r"\w+", query.lower()))
    query_wants_contact = bool(q_tokens & CONTACT_QUERY_TOKENS)
    candidates: List[Dict[str, Any]] = []
    for r in reranked:
        cid = r["chunk_id"]
        text = chunk_texts.get(cid, "")
        contact_penalty = 0.5 if (not query_wants_contact and EMAIL_PHONE_RE.search(text)) else 0.0
        ct = (chunk_type.get(cid) or "").lower()
        type_rank = CHUNK_TYPE_ORDER.get(ct, 3)
        candidates.append({
            "chunk_id": cid,
            "score": r.get("fused_score", 0.0),
            "fused_score": r.get("fused_score", 0.0),
            "section_title": section_title.get(cid, ""),
            "doc_id": chunk_doc_ids.get(cid, ""),
            "chunk_type": ct,
            "contact_penalty": contact_penalty,
            "type_rank": type_rank,
        })
    candidates.sort(key=lambda c: (c["contact_penalty"], c["type_rank"], -c["fused_score"]))
    selected: List[Dict[str, Any]] = []
    seen_section_key: set = set()
    for c in candidates:
        if len(selected) >= VAGUE_SUMMARY_MAX_CHUNKS:
            break
        key = (c["doc_id"], c["section_title"] or "(no section)")
        if key in seen_section_key and len(selected) >= 3:
            continue
        selected.append({"chunk_id": c["chunk_id"], "score": c["score"]})
        seen_section_key.add(key)
    return selected


def single_attempt(
    query: str,
    top_k: int,
    w_vec: float,
    w_lex: float,
    embedding_provider_name: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """
    One retrieval + metrics + diagnoses.
    Returns (reranked, selected, chunk_texts, chunk_doc_ids, metrics, diagnoses).
    """
    reranked, chunk_texts, chunk_doc_ids = hybrid_retrieve(
        query, top_k, w_vec, w_lex, embedding_provider_name
    )
    _, chunk_type = _enrich_with_section_and_type([r["chunk_id"] for r in reranked]) if reranked else ({}, {})
    metrics = compute_retrieval_metrics(reranked, query, chunk_texts, chunk_doc_ids, chunk_types=chunk_type or None)
    selected = select_chunks_for_context(
        reranked, chunk_texts, query,
        query_specificity=metrics.get("query_specificity"),
        overlap_top3=metrics.get("overlap_top3", metrics.get("lexical_overlap_ratio")),
    )
    query_features = extract_query_features(query)
    diagnoses = diagnose(metrics, query_features)
    return reranked, selected, chunk_texts, chunk_doc_ids, metrics, diagnoses


def run_self_heal_loop(
    query: str,
    top_k: int,
    embedding_provider_name: str,
    failed_thresholds: List[Dict[str, Any]] | None = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    int,
    str,
]:
    """
    Run retrieval; if thresholds fail, apply heals. If query_specificity failed, QUERY_REWRITE runs first.
    When vague (query_specificity < 0.25), do not choose keyword-anchored rewrite C; use diversity selection.
    Returns (final_reranked, final_selected, chunk_texts, attempts, heals_applied, best_attempt_no, mode).
    """
    w_vec, w_lex = HYBRID_W_VEC, HYBRID_W_LEX
    attempts = []
    heals_applied = []
    best_reranked: List[Dict[str, Any]] = []
    best_selected: List[Dict[str, Any]] = []
    best_chunk_texts: Dict[str, str] = {}
    best_quality = -1.0
    best_attempt_no = 1
    query_used = query
    current_k = top_k
    failed_names = {t["name"] for t in (failed_thresholds or [])}
    query_rewrite_already_tried = False
    vague_mode = False

    for attempt_no in range(1, MAX_SELF_HEAL_ATTEMPTS + 1):
        reranked, selected, chunk_texts, chunk_doc_ids, metrics, diagnoses = single_attempt(
            query_used, current_k, w_vec, w_lex, embedding_provider_name
        )
        if not reranked:
            attempt_record = {
                "schema_version": 1,
                "attempt_no": attempt_no,
                "query_used": query_used,
                "weights": {"w_vec": w_vec, "w_lex": w_lex},
                "top_k": current_k,
                "retrieved": [],
                "metrics": metrics,
                "diagnoses": diagnoses,
                "mode": "vague_summary" if vague_mode else "normal",
            }
            attempts.append(attempt_record)
            break
        # Build attempt record for trace
        section_title, chunk_type = _enrich_with_section_and_type([r["chunk_id"] for r in reranked]) if reranked else ({}, {})
        retrieved_for_trace = []
        for r in reranked:
            cid = r["chunk_id"]
            snippet = (chunk_texts.get(cid, "")[:200] + "…") if len(chunk_texts.get(cid, "")) > 200 else chunk_texts.get(cid, "")
            retrieved_for_trace.append({
                "chunk_id": cid,
                "doc_id": chunk_doc_ids.get(cid, ""),
                "vector_score": r.get("vector_score"),
                "lexical_score": r.get("lexical_score"),
                "fused_score": r.get("fused_score"),
                "snippet": snippet,
                "section_title": section_title.get(cid, ""),
                "chunk_type": chunk_type.get(cid, ""),
            })

        if attempt_no == 1:
            vague_mode = (metrics.get("query_specificity", 1) or 0) < VAGUE_QUERY_SPECIFICITY_THRESHOLD
        if vague_mode:
            selected = select_chunks_for_vague_summary(reranked, chunk_texts, query_used, chunk_doc_ids)

        attempt_record = {
            "schema_version": 1,
            "attempt_no": attempt_no,
            "query_used": query_used,
            "weights": {"w_vec": w_vec, "w_lex": w_lex},
            "top_k": current_k,
            "retrieved": retrieved_for_trace,
            "metrics": metrics,
            "diagnoses": diagnoses,
            "mode": "vague_summary" if vague_mode else "normal",
        }
        attempts.append(attempt_record)

        q_score = quality_score_for_attempt(metrics)
        if q_score > best_quality:
            best_quality = q_score
            best_reranked = reranked
            best_selected = selected
            best_chunk_texts = chunk_texts
            best_attempt_no = attempt_no

        if passes_thresholds(metrics, MIN_MAX_FUSED, MIN_LEXICAL_OVERLAP, MIN_SCORE_SEPARATION, MIN_QUERY_SPECIFICITY):
            break

        # Heal: determine trigger reasons; query_specificity failed -> QUERY_REWRITE first
        next_query = query_used
        next_w_vec, next_w_lex = w_vec, w_lex
        next_k = current_k
        heal_type = None
        heal_params: Dict[str, Any] = {}
        heal_reason = "; ".join(d.get("reason", "") for d in diagnoses[:2])
        overlap_val = metrics.get("overlap_top3", metrics.get("lexical_overlap_ratio", 0)) or 0
        max_fused_val = metrics.get("max_fused_score", 0) or 0

        # 1) QUERY_REWRITE: when query_specificity failed (always first) or vague/low overlap
        should_rewrite = (
            "query_specificity" in failed_names
            or any(d.get("label") == "QUERY_TOO_VAGUE" for d in diagnoses)
            or overlap_val < MIN_LEXICAL_OVERLAP
        )
        if should_rewrite and not query_rewrite_already_tried:
            query_features = extract_query_features(query_used)
            top_chunk_texts = [chunk_texts.get(r["chunk_id"], "") for r in reranked[:3]]
            salient_terms = extract_salient_terms(top_chunk_texts, 6) if top_chunk_texts else []
            rewrites = query_rewrites(query_used, query_features, salient_terms=salient_terms)
            best_rewrite = query_used
            best_rewrite_quality = -1.0
            rewrite_scores: List[Dict[str, Any]] = []
            for i, rq in enumerate(rewrites):
                _, _, _, _, m, _ = single_attempt(rq, current_k, w_vec, w_lex, embedding_provider_name)
                qq = quality_score_for_attempt(m)
                rewrite_scores.append({
                    "rewrite": rq,
                    "quality_score": round(qq, 4),
                    "max_fused": round((m.get("max_fused_score") or 0), 4),
                    "overlap_top3": round((m.get("overlap_top3", m.get("lexical_overlap_ratio")) or 0), 4),
                    "separation": round((m.get("score_separation") or 0), 4),
                })
                if vague_mode and i == 2:
                    continue
                if qq > best_rewrite_quality:
                    best_rewrite_quality = qq
                    best_rewrite = rq
            query_rewrite_already_tried = True
            heal_type = "QUERY_REWRITE"
            heal_params = {
                "rewrites_tried": rewrites,
                "chosen_rewrite": best_rewrite,
                "rewrite_scores": rewrite_scores,
            }
            next_query = best_rewrite
            heals_applied.append({
                "schema_version": 1,
                "type": heal_type,
                "params": heal_params,
                "reason": heal_reason,
                "before_metrics": metrics,
                "after_metrics": None,
                "delta": None,
            })
            query_used = next_query
            continue

        # 2) Weight adjustment
        if not heal_type and (any(d.get("label") == "RANKING_WEAK" for d in diagnoses) or (metrics.get("score_separation") or 0) < MIN_SCORE_SEPARATION):
            query_features = extract_query_features(query_used)
            next_w_vec, next_w_lex = suggest_weights(query_features, diagnoses)
            heal_type = "DYNAMIC_WEIGHT_ADJUSTMENT"
            heal_params = {"w_vec": next_w_vec, "w_lex": next_w_lex}
            w_vec, w_lex = next_w_vec, next_w_lex
            heals_applied.append({
                "schema_version": 1,
                "type": heal_type,
                "params": heal_params,
                "reason": heal_reason,
                "before_metrics": metrics,
                "after_metrics": None,
                "delta": None,
            })
            continue

        # 3) K_INCREASE only if max_fused or overlap_top3 still below threshold, and top_k < 20, and not duplicate
        next_k = suggest_k_increase(current_k)
        fused_or_overlap_still_low = max_fused_val < MIN_MAX_FUSED or overlap_val < MIN_LEXICAL_OVERLAP
        can_increase_k = current_k < 20 and next_k > current_k
        last_heal = heals_applied[-1] if heals_applied else None
        duplicate_k = last_heal and last_heal.get("type") == "K_INCREASE" and last_heal.get("params", {}).get("after_k") == next_k
        if fused_or_overlap_still_low and can_increase_k and not duplicate_k:
            heal_type = "K_INCREASE"
            heal_params = {"before_k": current_k, "after_k": next_k}
            heals_applied.append({
                "schema_version": 1,
                "type": heal_type,
                "params": heal_params,
                "reason": heal_reason,
                "before_metrics": metrics,
                "after_metrics": None,
                "delta": None,
            })
            current_k = next_k
        # If we couldn't apply any heal (e.g. already at k=20), break to avoid infinite loop
        if not heal_type:
            break

        query_used = next_query
        w_vec, w_lex = next_w_vec, next_w_lex

    # Fill after_metrics/delta for last heal if we have one more attempt after it
    for i, h in enumerate(heals_applied):
        if i + 1 < len(attempts):
            h["after_metrics"] = attempts[i + 1]["metrics"]
            b = h.get("before_metrics") or {}
            a = h["after_metrics"]
            h["delta"] = {
                "max_fused": round((a.get("max_fused_score") or 0) - (b.get("max_fused_score") or 0), 4),
                "lexical_overlap": round((a.get("lexical_overlap_ratio") or 0) - (b.get("lexical_overlap_ratio") or 0), 4),
                "overlap_top3": round(((a.get("overlap_top3", a.get("lexical_overlap_ratio")) or 0) - ((b.get("overlap_top3", b.get("lexical_overlap_ratio")) or 0))), 4),
                "score_separation": round((a.get("score_separation") or 0) - (b.get("score_separation") or 0), 4),
            }

    return best_reranked, best_selected, best_chunk_texts, attempts, heals_applied, best_attempt_no, ("vague_summary" if vague_mode else "normal")


def retrieve_and_rerank(
    query: str,
    top_k: int,
    embedding_provider_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]], bool, List[Dict[str, Any]], int, str]:
    """
    Entry: single attempt if thresholds pass, else self-heal loop.
    Returns (reranked, selected, chunk_texts, attempts, heals_applied, self_heal_triggered, trigger_failed_thresholds, best_attempt_no, mode).
    """
    reranked, selected, chunk_texts, chunk_doc_ids, metrics, diagnoses = single_attempt(
        query, top_k, HYBRID_W_VEC, HYBRID_W_LEX, embedding_provider_name
    )
    vague_mode = (metrics.get("query_specificity", 1) or 0) < VAGUE_QUERY_SPECIFICITY_THRESHOLD
    if vague_mode:
        selected = select_chunks_for_vague_summary(reranked, chunk_texts, query, chunk_doc_ids)
    mode = "vague_summary" if vague_mode else "normal"
    if passes_thresholds(metrics, MIN_MAX_FUSED, MIN_LEXICAL_OVERLAP, MIN_SCORE_SEPARATION, MIN_QUERY_SPECIFICITY):
        section_title, chunk_type = _enrich_with_section_and_type([r["chunk_id"] for r in reranked]) if reranked else ({}, {})
        attempt1 = {
            "schema_version": 1,
            "attempt_no": 1,
            "query_used": query,
            "weights": {"w_vec": HYBRID_W_VEC, "w_lex": HYBRID_W_LEX},
            "top_k": top_k,
            "retrieved": [
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": chunk_doc_ids.get(r["chunk_id"], ""),
                    "vector_score": r.get("vector_score"),
                    "lexical_score": r.get("lexical_score"),
                    "fused_score": r.get("fused_score"),
                    "snippet": (chunk_texts.get(r["chunk_id"], "")[:200] + "…") if len(chunk_texts.get(r["chunk_id"], "")) > 200 else chunk_texts.get(r["chunk_id"], ""),
                    "section_title": section_title.get(r["chunk_id"], ""),
                    "chunk_type": chunk_type.get(r["chunk_id"], ""),
                }
                for r in reranked
            ],
            "metrics": metrics,
            "diagnoses": diagnoses,
            "mode": mode,
        }
        return reranked, selected, chunk_texts, [attempt1], [], False, [], 1, mode
    # Thresholds failed on attempt 1: compute failed thresholds and run self-heal loop
    failed_thresholds = get_failed_thresholds(
        metrics, MIN_MAX_FUSED, MIN_LEXICAL_OVERLAP, MIN_SCORE_SEPARATION, MIN_QUERY_SPECIFICITY
    )
    best_reranked, best_selected, best_chunk_texts, attempts, heals_applied, best_attempt_no, mode = run_self_heal_loop(
        query, top_k, embedding_provider_name, failed_thresholds=failed_thresholds
    )
    return best_reranked, best_selected, best_chunk_texts, attempts, heals_applied, True, failed_thresholds, best_attempt_no, mode
