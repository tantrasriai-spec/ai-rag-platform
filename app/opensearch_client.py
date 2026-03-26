import os
from opensearchpy import OpenSearch

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch:9200")

def get_os_client() -> OpenSearch:
    # Security plugin is disabled in docker-compose, so no auth needed
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        verify_certs=False,
    )