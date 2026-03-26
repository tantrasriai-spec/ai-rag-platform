# app/document_store.py

from dataclasses import dataclass
from typing import List, Optional

from db import get_conn


@dataclass
class StoredChunk:
    chunk_index: int
    page_number: Optional[int]
    content: str


def fetch_chunks_for_document(
    document_id: str,
    limit: int = 500,
    offset: int = 0,
) -> List[StoredChunk]:
    if not document_id:
        return []

    limit = max(1, min(int(limit), 2000))
    offset = max(0, int(offset))

    sql = """
        SELECT chunk_index, page_number, content
        FROM chunks
        WHERE document_id = %s
        ORDER BY chunk_index ASC
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (document_id, limit, offset))
            rows = cur.fetchall() or []

    out: List[StoredChunk] = []
    for r in rows:
        out.append(
            StoredChunk(
                chunk_index=int(r[0]),
                page_number=(int(r[1]) if r[1] is not None else None),
                content=str(r[2] or ""),
            )
        )
    return out


def fetch_all_chunk_texts(
    document_id: str,
    page_size: int = 500,
    max_chunks: int = 4000,
) -> List[str]:
    all_texts: List[str] = []
    offset = 0

    while len(all_texts) < max_chunks:
        batch = fetch_chunks_for_document(
            document_id=document_id,
            limit=page_size,
            offset=offset,
        )
        if not batch:
            break

        for chunk in batch:
            text = (chunk.content or "").strip()
            if text:
                all_texts.append(text)
                if len(all_texts) >= max_chunks:
                    break

        if len(batch) < page_size:
            break

        offset += page_size

    return all_texts