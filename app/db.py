import os
import psycopg

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://rag:ragpass@postgres:5432/ragdb")

def get_conn():
    return psycopg.connect(POSTGRES_URL)