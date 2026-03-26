import os
from langchain_google_vertexai import VertexAIEmbeddings

def get_embedder() -> VertexAIEmbeddings:
    # Uses GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION and ADC credentials
    model = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004")
    return VertexAIEmbeddings(model_name=model)

def embed_texts(texts: list[str]) -> list[list[float]]:
    emb = get_embedder()
    return emb.embed_documents(texts)

def embed_query(text: str) -> list[float]:
    emb = get_embedder()
    return emb.embed_query(text)