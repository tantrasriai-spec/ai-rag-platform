# app/cache_keys.py
# Stable cache keys + hashing to keep keys short and safe.

import hashlib
from typing import Optional


CACHE_VERSION = "v2"
SUMMARY_VERSION = "s1"
RETRIEVAL_VERSION = "r2"
ANSWER_VERSION = "a2"


def _hash_text(text: str) -> str:
    norm = " ".join((text or "").strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def retrieval_key(question: str, k: int, document_id: Optional[str]) -> str:
    doc = document_id or "ALL"
    qh = _hash_text(question)
    return f"retrieval:{CACHE_VERSION}:{RETRIEVAL_VERSION}:doc={doc}:k={k}:q={qh}"


def answer_key(question: str, k: int, document_id: Optional[str]) -> str:
    doc = document_id or "ALL"
    qh = _hash_text(question)
    return f"answer:{CACHE_VERSION}:{ANSWER_VERSION}:doc={doc}:k={k}:q={qh}"


def summary_key(document_id: str, task: str) -> str:
    return f"summary:{CACHE_VERSION}:{SUMMARY_VERSION}:doc={document_id}:task={_hash_text(task)}"


def doc_prefix(document_id: str) -> str:
    return f"*doc={document_id}*"
