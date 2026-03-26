# app/main.py
# FastAPI API service for production-style RAG:
# - /upload -> save PDF to shared volume + queue Celery ingestion
# - /documents -> list docs from Postgres
# - /query -> hybrid retrieval (with Redis retrieval cache)
# - /query-vector -> vector-only retrieval for debugging
# - /answer -> route either to Q&A pipeline or document-task pipeline

import hashlib
import json
import os
import uuid







from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from psycopg.errors import UniqueViolation
from cache_keys import answer_key, doc_prefix, retrieval_key, summary_key
from db import get_conn
from document_store import fetch_all_chunk_texts
from llm_vertex import generate_answer_with_usage
from redis_client import get_redis
from reranker import rerank_results
from retrieval import expand_results_with_neighbors, search_chunks, search_chunks_vector
from summarize import detect_doc_task, run_doc_task
from tasks import ingest_document
from typing import Any, Dict, List, Optional


# -----------------------------
# App init
# -----------------------------
app = FastAPI(title="RAG API")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
RETRIEVAL_CACHE_TTL = 20 * 60
ANSWER_CACHE_TTL = 60 * 60
SUMMARY_CACHE_TTL = 24 * 60 * 60


# -----------------------------
# Request Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 5
    document_id: Optional[str] = None


class AnswerRequest(BaseModel):
    question: str
    k: int = 6
    document_id: Optional[str] = None
    include_sources: bool = False
    mode: Optional[str] = None
    doc_task: Optional[str] = None


# -----------------------------
# DB init (runs once on startup)
# -----------------------------
@app.on_event("startup")
def init_db():
    print("Initializing database schema...")
    with open("schema.sql", "r", encoding="utf-8") as f:
        sql = f.read()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Database ready.")


# -----------------------------
# Helpers
# -----------------------------
def _zero_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _cost_from_usage(usage: Dict[str, int]) -> Dict[str, float]:
    in_cost_per_1k = float(os.getenv("VERTEX_INPUT_COST_PER_1K", "0") or "0")
    out_cost_per_1k = float(os.getenv("VERTEX_OUTPUT_COST_PER_1K", "0") or "0")
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    cost = {
        "currency": "USD",
        "input_cost_per_1k": in_cost_per_1k,
        "output_cost_per_1k": out_cost_per_1k,
        "input_cost": (prompt_tokens / 1000.0) * in_cost_per_1k,
        "output_cost": (completion_tokens / 1000.0) * out_cost_per_1k,
    }
    cost["total_cost"] = cost["input_cost"] + cost["output_cost"]
    return cost


def _doc_task_from_request(req: AnswerRequest):
    """
    Decide which document pipeline to run.

    Priority:
    1. Explicit doc_task from UI
    2. Detect from question text
    """

    if getattr(req, "mode", None) == "doc_task" and req.doc_task:
        return req.doc_task

    # fallback detection (safety)
    return detect_doc_task(req.question)


# -----------------------------
# Health endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/db-check")
def db_check():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
    return {"documents_count": count}


@app.get("/documents")
def list_documents() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, status
                FROM documents
                ORDER BY created_at DESC NULLS LAST, id DESC
                """
            )
            rows = cur.fetchall()
    return [{"id": r[0], "filename": r[1], "status": r[2]} for r in rows]


# -----------------------------
# Upload PDF -> queue ingestion
# -----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, status FROM documents WHERE sha256=%s LIMIT 1;",
                (file_hash,),
            )
            row = cur.fetchone()

    if row:
        existing_id, existing_filename, existing_status = row
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_uploads (document_id, uploaded_filename, sha256)
                    VALUES (%s, %s, %s)
                    """,
                    (existing_id, file.filename, file_hash),
                )
            conn.commit()
        return {
            "message": "Duplicate file detected (sha256 match). Upload recorded.",
            "already_exists": True,
            "doc_id": existing_id,
            "original_filename": existing_filename,
            "uploaded_filename": file.filename,
            "status": existing_status,
            "task_id": None,
        }

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (id, filename, sha256, status)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (doc_id, file.filename, file_hash, "processing"),
                )
                cur.execute(
                    """
                    INSERT INTO document_uploads (document_id, uploaded_filename, sha256)
                    VALUES (%s, %s, %s)
                    """,
                    (doc_id, file.filename, file_hash),
                )
            conn.commit()
    except UniqueViolation:
        try:
            os.remove(file_path)
        except OSError:
            pass
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, filename, status FROM documents WHERE sha256=%s LIMIT 1;",
                    (file_hash,),
                )
                row = cur.fetchone()
        existing_id, existing_filename, existing_status = row
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_uploads (document_id, uploaded_filename, sha256)
                    VALUES (%s, %s, %s)
                    """,
                    (existing_id, file.filename, file_hash),
                )
            conn.commit()
        return {
            "message": "Duplicate detected during insert (race). Upload recorded.",
            "already_exists": True,
            "doc_id": existing_id,
            "original_filename": existing_filename,
            "uploaded_filename": file.filename,
            "status": existing_status,
            "task_id": None,
        }

    job = ingest_document.delay(doc_id, file_path, file.filename)
    return {
        "message": "Upload successful. Ingestion queued.",
        "already_exists": False,
        "doc_id": doc_id,
        "uploaded_filename": file.filename,
        "sha256": file_hash,
        "task_id": job.id,
    }


# -----------------------------
# Retrieval endpoints
# -----------------------------
@app.post("/query")
def query(req: QueryRequest):
    r = get_redis()
    key = retrieval_key(req.query, req.k, req.document_id)

    cached = r.get(key)
    if cached:
        return {
            "query": req.query,
            "document_id": req.document_id,
            "cache": "HIT",
            "results": json.loads(cached),
        }

    retrieval_k = max(req.k, 30)
    results = search_chunks(req.query, retrieval_k, req.document_id)
    results = rerank_results(req.query, results, top_n=req.k)

    r.setex(key, RETRIEVAL_CACHE_TTL, json.dumps(results))

    return {
        "query": req.query,
        "document_id": req.document_id,
        "cache": "MISS",
        "results": results,
    }

@app.post("/query-vector")
def query_vector(req: QueryRequest):
    results = search_chunks_vector(req.query, req.k, req.document_id)
    return {"query": req.query, "document_id": req.document_id, "results": results}


# -----------------------------
# Core answer route
# -----------------------------
@app.post("/answer")
def answer(req: AnswerRequest):
    r = get_redis()

    # Route document-wide tasks away from the Q&A retrieval path.
    task = _doc_task_from_request(req) if req.document_id else None
    if req.document_id and task:
        return _document_task_answer(req, task)

    akey = answer_key(req.question, req.k, req.document_id)
    cached_answer = r.get(akey)
    if cached_answer:
        payload = json.loads(cached_answer)
        payload["cache"] = "HIT"
        payload["retrieval_cache"] = "SKIPPED"
        return payload

    # Retrieval cache stores the raw hybrid candidates only.
    retrieval_k = max(req.k, 30)
    rkey = retrieval_key(req.question, retrieval_k, req.document_id)
    cached_results = r.get(rkey)
    if cached_results:
        hybrid_candidates = json.loads(cached_results)
        retrieval_cache = "HIT"
    else:
        hybrid_candidates = search_chunks(req.question, retrieval_k, req.document_id)
        r.setex(rkey, RETRIEVAL_CACHE_TTL, json.dumps(hybrid_candidates))
        retrieval_cache = "MISS"

    # Dedicated reranking layer for production precision.
    reranked = rerank_results(req.question, hybrid_candidates, top_n=max(req.k, 6))
    results = expand_results_with_neighbors(reranked[: min(len(reranked), 3)], window=1)

    if not results:
        payload = {
            "question": req.question,
            "document_id": req.document_id,
            "answer": "I don't know.",
            "usage": _zero_usage(),
            "cost": _cost_from_usage(_zero_usage()),
            "cache": "MISS",
            "retrieval_cache": retrieval_cache,
            "sources": [] if req.include_sources else None,
        }
        if not req.include_sources:
            payload.pop("sources", None)
        r.setex(akey, 2 * 60, json.dumps(payload))
        return payload

    context = "\n\n---\n\n".join(
        f"[doc={s['document_id']} chunk={s['chunk_index']} page={s.get('page_number')}]\n{s['content']}"
        for s in results
    )

    final_answer, usage = generate_answer_with_usage(req.question, context)
    payload = {
        "question": req.question,
        "document_id": req.document_id,
        "answer": final_answer,
        "usage": usage,
        "cost": _cost_from_usage(usage),
        "cache": "MISS",
        "retrieval_cache": retrieval_cache,
    }
    if req.include_sources:
        payload["sources"] = results

    r.setex(akey, ANSWER_CACHE_TTL, json.dumps(payload))
    return payload

def _build_cost(usage: Dict[str, int]) -> Dict[str, float]:
    in_cost_per_1k = float(os.getenv("VERTEX_INPUT_COST_PER_1K", "0") or "0")
    out_cost_per_1k = float(os.getenv("VERTEX_OUTPUT_COST_PER_1K", "0") or "0")
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)

    cost = {
        "currency": "USD",
        "input_cost_per_1k": in_cost_per_1k,
        "output_cost_per_1k": out_cost_per_1k,
        "input_cost": (prompt_tokens / 1000.0) * in_cost_per_1k,
        "output_cost": (completion_tokens / 1000.0) * out_cost_per_1k,
    }
    cost["total_cost"] = cost["input_cost"] + cost["output_cost"]
    return cost

# -----------------------------
# Document-task pipeline
# -----------------------------
def _document_task_answer(req: AnswerRequest, task: str) -> dict:
    r = get_redis()
    skey = summary_key(req.document_id, task)

    cached = r.get(skey)
    if cached:
        payload = json.loads(cached)
        payload["cache"] = "HIT"
        payload["retrieval_cache"] = "SKIPPED"
        return payload

    chunk_texts = fetch_all_chunk_texts(
        document_id=req.document_id,
        page_size=500,
        max_chunks=4000,
    )

    if not chunk_texts:
        payload = {
            "question": req.question,
            "document_id": req.document_id,
            "answer": "No document content found.",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost": _build_cost({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "cache": "MISS",
            "retrieval_cache": "SKIPPED",
        }
        if req.include_sources:
            payload["sources"] = []
        r.setex(skey, SUMMARY_CACHE_TTL, json.dumps(payload))
        return payload

    final_text, usage = run_doc_task(task, chunk_texts)

    payload = {
        "question": req.question,
        "document_id": req.document_id,
        "answer": final_text,
        "usage": usage,
        "cost": _build_cost(usage),
        "cache": "MISS",
        "retrieval_cache": "SKIPPED",
    }

    if req.include_sources:
        payload["sources"] = []

    r.setex(skey, SUMMARY_CACHE_TTL, json.dumps(payload))
    return payload
# -----------------------------
# Cache maintenance
# -----------------------------
@app.delete("/cache/document/{document_id}")
def clear_cache_for_document(document_id: str):
    r = get_redis()
    pattern = doc_prefix(document_id)
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=200)
        if keys:
            deleted += r.delete(*keys)
        if cursor == 0:
            break
    return {"document_id": document_id, "deleted_keys": deleted}


@app.get("/documents/{doc_id}/uploads")
def list_uploads(doc_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uploaded_filename, uploaded_at
                FROM document_uploads
                WHERE document_id = %s
                ORDER BY uploaded_at DESC
                """,
                (doc_id,),
            )
            rows = cur.fetchall()
    return [{"uploaded_filename": r[0], "uploaded_at": str(r[1])} for r in rows]
