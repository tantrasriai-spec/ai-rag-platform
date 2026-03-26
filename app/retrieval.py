# app/retrieval.py
# Retrieval stage for production-style RAG:
# 1) BM25 retrieval for lexical precision
# 2) Vector retrieval for semantic recall
# 3) Hybrid merge
# 4) Neighbor expansion for chunk-boundary safety

from typing import Any, Dict, List, Optional, Tuple

from db import get_conn
from embeddings_vertex import embed_query
from opensearch_client import get_os_client
from search_index import INDEX_NAME, ensure_index


def search_chunks_bm25(query: str, k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
    ensure_index()
    client = get_os_client()
    filters = []
    if document_id:
        filters.append({"term": {"document_id": document_id}})

    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": filters,
                "should": [
                    {"match_phrase": {"content": {"query": query, "boost": 4}}},
                    {"match": {"content": {"query": query, "boost": 2, "operator": "and"}}},
                    {"match": {"content": {"query": query, "boost": 1}}},
                ],
                "minimum_should_match": 1,
            }
        },
        "highlight": {
            "fields": {
                "content": {"fragment_size": 160, "number_of_fragments": 2}
            }
        },
    }

    resp = client.search(index=INDEX_NAME, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    return [
        {
            "score": h.get("_score"),
            "document_id": (h.get("_source") or {}).get("document_id"),
            "chunk_index": (h.get("_source") or {}).get("chunk_index"),
            "page_number": (h.get("_source") or {}).get("page_number"),
            "content": (h.get("_source") or {}).get("content"),
            "highlights": (h.get("highlight") or {}).get("content", []),
            "method": "bm25",
        }
        for h in hits
    ]


def search_chunks_vector(query: str, k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
    ensure_index()
    client = get_os_client()
    qvec = embed_query(query)
    filters = []
    if document_id:
        filters.append({"term": {"document_id": document_id}})

    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": filters,
                "must": [
                    {"knn": {"embedding": {"vector": qvec, "k": k}}}
                ],
            }
        },
    }

    resp = client.search(index=INDEX_NAME, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    return [
        {
            "score": h.get("_score"),
            "document_id": (h.get("_source") or {}).get("document_id"),
            "chunk_index": (h.get("_source") or {}).get("chunk_index"),
            "page_number": (h.get("_source") or {}).get("page_number"),
            "content": (h.get("_source") or {}).get("content"),
            "highlights": [],
            "method": "vector",
        }
        for h in hits
    ]


def search_chunks_hybrid(
    query: str,
    k: int = 5,
    document_id: Optional[str] = None,
    bm25_k: Optional[int] = None,
    vector_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if bm25_k is None:
        bm25_k = max(k, 10)
    if vector_k is None:
        vector_k = max(k, 10)

    bm25_results = search_chunks_bm25(query=query, k=bm25_k, document_id=document_id)
    vector_results = search_chunks_vector(query=query, k=vector_k, document_id=document_id)

    merged: Dict[str, Dict[str, Any]] = {}

    def key_of(r: Dict[str, Any]) -> str:
        return f"{r.get('document_id')}::{r.get('chunk_index')}"

    for r in bm25_results:
        key = key_of(r)
        merged[key] = dict(r)
        merged[key]["bm25_score"] = float(r.get("score") or 0.0)
        merged[key]["vector_score"] = 0.0

    for r in vector_results:
        key = key_of(r)
        if key in merged:
            merged[key]["vector_score"] = float(r.get("score") or 0.0)
            merged[key]["method"] = "hybrid"
        else:
            merged[key] = dict(r)
            merged[key]["bm25_score"] = 0.0
            merged[key]["vector_score"] = float(r.get("score") or 0.0)
            merged[key]["method"] = "hybrid"

    bm25_max = max([x["bm25_score"] for x in merged.values()] or [1.0]) or 1.0
    vector_max = max([x["vector_score"] for x in merged.values()] or [1.0]) or 1.0

    bm25_weight = 0.55
    vector_weight = 0.45

    for item in merged.values():
        bm25_norm = item["bm25_score"] / bm25_max
        vector_norm = item["vector_score"] / vector_max
        item["combined_score"] = (bm25_weight * bm25_norm) + (vector_weight * vector_norm)

    return sorted(merged.values(), key=lambda r: r.get("combined_score", 0.0), reverse=True)[:k]


def search_chunks(query: str, k: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return search_chunks_hybrid(query=query, k=k, document_id=document_id)


def fetch_adjacent_chunks(document_id: str, chunk_index: int, window: int = 1) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, chunk_index, page_number, content
                FROM chunks
                WHERE document_id = %s
                  AND chunk_index BETWEEN %s AND %s
                ORDER BY chunk_index ASC
                """,
                (document_id, chunk_index - window, chunk_index + window),
            )
            rows = cur.fetchall()

    return [
        {
            "document_id": r[0],
            "chunk_index": r[1],
            "page_number": r[2],
            "content": r[3],
            "method": "adjacent",
            "score": 1.0,
            "highlights": [],
        }
        for r in rows
    ]


def expand_results_with_neighbors(results: List[Dict[str, Any]], window: int = 1) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, int]] = set()
    expanded: List[Dict[str, Any]] = []

    for r in results:
        doc_id = r.get("document_id")
        chunk_idx = r.get("chunk_index")
        if doc_id is None or chunk_idx is None:
            continue
        for n in fetch_adjacent_chunks(str(doc_id), int(chunk_idx), window=window):
            key = (n["document_id"], n["chunk_index"])
            if key not in seen:
                seen.add(key)
                expanded.append(n)

    expanded.sort(key=lambda x: (x["document_id"], x["chunk_index"]))
    return expanded
