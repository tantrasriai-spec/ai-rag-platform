# app/search_index.py
#
# Creates/ensures the OpenSearch index used for chunk storage + vector search.
# We use a NEW index name (v2) because adding knn_vector fields to an existing
# index is not straightforward.

from opensearch_client import get_os_client

# ✅ New index with vector field
INDEX_NAME = "rag_chunks_v2"

# ✅ text-embedding-004 returns 768-dim vectors (your test confirmed)
EMBED_DIM = 768

INDEX_BODY = {
    "settings": {
        "index": {
            "knn": True,
            # Optional: tune shard/replica for local dev
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "document_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "page_number": {"type": "integer"},
            "content": {"type": "text"},
            # ✅ Vector field for semantic search
            "embedding": {
                "type": "knn_vector",
                "dimension": EMBED_DIM
            },
        }
    },
}


def ensure_index() -> None:
    """
    Create the OpenSearch index if it does not exist.
    Safe to call multiple times.
    """
    client = get_os_client()

    # `exists()` may differ across versions; this works for opensearch-py 2.x
    exists = client.indices.exists(INDEX_NAME)

    # Some versions return dict; normalize to bool
    if isinstance(exists, dict):
        # {'status': 200} style is rare; keep safe
        exists = bool(exists.get("status") == 200)

    if not exists:
        client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
        print(f"[opensearch] Created index: {INDEX_NAME}")
    else:
        # Index already exists; do nothing
        # print(f"[opensearch] Index exists: {INDEX_NAME}")
        pass