# app/redis_client.py
# Creates a Redis client using REDIS_URL from docker-compose

import os
import redis

_redis = None  # module-level singleton


def get_redis() -> redis.Redis:
    global _redis

    if _redis is not None:
        return _redis

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # decode_responses=True returns strings instead of bytes (easier)
    _redis = redis.Redis.from_url(redis_url, decode_responses=True)

    return _redis