# app/reranker.py
# Lightweight reranker used after hybrid retrieval.
# This gives us a dedicated reranking stage without introducing a heavyweight
# model dependency right now. It can be replaced later with a cross-encoder.

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "about", "into",
    "your", "their", "them", "these", "those", "document", "pdf",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", _normalize(text)) if t and t not in STOPWORDS]


def _phrase_bonus(query: str, content: str) -> float:
    q = _normalize(query)
    c = _normalize(content)
    if not q or not c:
        return 0.0
    if q in c:
        return 2.5

    # Reward shorter overlapping phrases of 2-4 tokens.
    q_tokens = _tokens(q)
    bonus = 0.0
    for n in range(4, 1, -1):
        for i in range(0, max(0, len(q_tokens) - n + 1)):
            phrase = " ".join(q_tokens[i : i + n])
            if phrase and phrase in c:
                bonus = max(bonus, 0.35 * n)
    return bonus


def _token_overlap_bonus(query: str, content: str) -> float:
    q_tokens = _tokens(query)
    c_tokens = set(_tokens(content))
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = sum(1 for t in q_tokens if t in c_tokens)
    coverage = overlap / max(1, len(set(q_tokens)))
    return coverage * 3.0


def _length_penalty(content: str) -> float:
    # Slightly penalize very short chunks because they often lack enough answer context.
    length = len((content or "").strip())
    if length <= 80:
        return -0.35
    if length <= 160:
        return -0.15
    if length >= 1800:
        return -0.10
    return 0.0


def rerank_results(
    query: str,
    results: Iterable[Dict[str, Any]],
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """
    Re-score hybrid retrieval candidates.

    This stage intentionally sits between retrieval and generation so Q&A is:
    retrieve -> rerank -> generate
    instead of sending raw similarity-ranked chunks directly to the LLM.
    """
    reranked: List[Dict[str, Any]] = []

    for rank, item in enumerate(results, start=1):
        content = item.get("content") or ""
        base_score = float(item.get("combined_score") or item.get("score") or 0.0)
        overlap = _token_overlap_bonus(query, content)
        phrase = _phrase_bonus(query, content)
        length_penalty = _length_penalty(content)
        initial_rank_bonus = max(0.0, 0.35 - (rank - 1) * 0.01)

        rerank_score = base_score + overlap + phrase + length_penalty + initial_rank_bonus

        copy = dict(item)
        copy["retrieval_score"] = base_score
        copy["rerank_score"] = round(rerank_score, 6)
        copy["method"] = "hybrid+rereank" if item.get("method") == "hybrid" else f"{item.get('method', 'retrieval')}+rerank"
        reranked.append(copy)

    reranked.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
    return reranked[: max(1, int(top_n))]
