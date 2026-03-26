# app/tasks.py

import os
import hashlib
from celery_app import celery
from db import get_conn

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from opensearch_client import get_os_client
from search_index import ensure_index, INDEX_NAME
from embeddings_vertex import embed_texts


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@celery.task(name="tasks.ingest_document")
def ingest_document(doc_id: str, file_path: str, filename: str = "uploaded.pdf"):
    """
    Worker ingestion task:
    1) Load PDF from shared volume
    2) Split into chunks
    3) Ensure document row exists
    4) Insert chunks into Postgres
    5) Embed chunks using Vertex embeddings
    6) Bulk index chunks+embeddings into OpenSearch
    7) Mark document as ingested
    """

    print(f"[worker] Ingest started doc_id={doc_id}")
    print(f"[worker] File path: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    # Compute sha for dedupe/tracking
    file_hash = sha256_file(file_path)

    # -----------------------------
    # 1) Load PDF
    # -----------------------------
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"[worker] Loaded pages: {len(pages)}")

    # -----------------------------
    # 2) Chunk PDF
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300
   )
    chunks = splitter.split_documents(pages)
    print(f"[worker] Created chunks: {len(chunks)}")

    # -----------------------------
    # 3) Write to Postgres (documents + chunks)
    # -----------------------------
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Ensure document exists
            cur.execute(
                """
                INSERT INTO documents (id, filename, sha256, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (doc_id, filename, file_hash, "processing")
            )

            # Insert chunks
            for i, ch in enumerate(chunks):
                page_number = ch.metadata.get("page", 0)
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, page_number, content)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index) DO NOTHING
                    """,
                    (doc_id, i, page_number, ch.page_content)
                )

            # Mark as processing for now (will flip to ingested after OpenSearch)
            cur.execute(
                "UPDATE documents SET status=%s WHERE id=%s",
                ("processing", doc_id)
            )

        conn.commit()

    # -----------------------------
    # 4) Ensure OpenSearch index exists
    # -----------------------------
    ensure_index()
    client = get_os_client()

    # -----------------------------
    # 5) Generate embeddings (Vertex)
    # -----------------------------
    texts = [ch.page_content for ch in chunks]
    vectors = embed_texts(texts)  # list[list[float]]
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count")

    # -----------------------------
    # 6) Bulk index into OpenSearch (content + embedding)
    # -----------------------------
    bulk_body = []
    for i, (ch, vec) in enumerate(zip(chunks, vectors)):
        page_number = ch.metadata.get("page", 0)

        doc = {
            "document_id": doc_id,
            "chunk_index": i,
            "page_number": page_number,
            "content": ch.page_content,
            "embedding": vec,  # ✅ vector field
        }

        bulk_body.append({"index": {"_index": INDEX_NAME, "_id": f"{doc_id}::{i}"}})
        bulk_body.append(doc)

    resp = client.bulk(body=bulk_body)

    # Count success/errors in bulk response
    errors = resp.get("errors", False)
    items = resp.get("items", [])
    success = 0
    fail = 0
    for it in items:
        action = it.get("index", {})
        status = action.get("status", 0)
        if 200 <= status < 300:
            success += 1
        else:
            fail += 1

    print(f"[worker] OpenSearch bulk indexed: success={success} errors={fail} errors_flag={errors}")

    if errors or fail > 0:
        # If some docs failed, keep status as error for visibility
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status=%s WHERE id=%s",
                    ("error_indexing", doc_id)
                )
            conn.commit()
        raise RuntimeError("OpenSearch bulk indexing had errors. Check worker logs.")

    # -----------------------------
    # 7) Mark document as ingested
    # -----------------------------
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status=%s WHERE id=%s",
                ("ingested", doc_id)
            )
        conn.commit()

    print(f"[worker] Ingest finished doc_id={doc_id}")

    return {
        "doc_id": doc_id,
        "filename": filename,
        "sha256": file_hash,
        "pages": len(pages),
        "chunks": len(chunks),
        "indexed": success,
        "status": "ingested"
    }