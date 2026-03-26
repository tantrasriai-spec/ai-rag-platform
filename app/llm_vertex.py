# app/llm_vertex.py

import os
import time
from typing import Any, Dict, Tuple, Optional, List

from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    GoogleAPICallError,
)

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Config (env-driven)
# -----------------------------
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")

# -----------------------------
# Prompts
# -----------------------------
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer ONLY using the provided context. "
            "If the answer is not in the context, say exactly: I don't know.",
        ),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]
)

DOC_TASK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert document analyst.\n"
            "Always produce the requested output using ONLY the provided text.\n"
            "Do not invent facts.\n"
            "Do not say 'I don't know'.\n"
            "If the text is limited, produce the best possible output from the available text.\n"
            "Keep the response structured and concise.",
        ),
        ("human", "TEXT:\n{text}\n\nTASK:\n{task}"),
    ]
)

# -----------------------------
# LLM Builder
# -----------------------------
def _build_llm(
    max_output_tokens: int = 512,
    temperature: float = 0.2,
) -> ChatVertexAI:
    return ChatVertexAI(
        project=PROJECT,
        location=LOCATION,
        model=MODEL,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def get_llm(
    max_output_tokens: int = 512,
    temperature: float = 0.2,
) -> ChatVertexAI:
    return _build_llm(max_output_tokens=max_output_tokens, temperature=temperature)


# -----------------------------
# Token usage extraction
# -----------------------------
def _extract_usage(response_metadata: Dict[str, Any]) -> Dict[str, int]:
    meta = response_metadata or {}

    for key in ("usage_metadata", "usage", "token_usage"):
        maybe = meta.get(key)
        if isinstance(maybe, dict):
            meta = maybe
            break

    def pick_int(*names: str) -> int:
        for n in names:
            v = meta.get(n)
            if isinstance(v, int):
                return v
        return 0

    prompt_tokens = pick_int(
        "prompt_token_count",
        "prompt_tokens",
        "input_tokens",
        "input_token_count",
    )
    completion_tokens = pick_int(
        "candidates_token_count",
        "completion_tokens",
        "output_tokens",
        "output_token_count",
    )
    total_tokens = pick_int("total_token_count", "total_tokens")

    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


# -----------------------------
# Retry wrapper
# -----------------------------
def _invoke_with_backoff(
    llm: ChatVertexAI,
    messages: List[BaseMessage],
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 10.0,
):
    delay = float(initial_delay)
    last_exc: Optional[Exception] = None

    for attempt in range(1, int(max_attempts) + 1):
        try:
            return llm.invoke(messages)
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded, GoogleAPICallError) as e:
            last_exc = e
            if attempt >= max_attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, float(max_delay))

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown error in _invoke_with_backoff")


# -----------------------------
# Helpers
# -----------------------------
def _safe_llm_invoke(
    prompt: ChatPromptTemplate,
    variables: Dict[str, str],
    *,
    max_output_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, int]]:
    llm = get_llm(max_output_tokens=max_output_tokens, temperature=temperature)
    messages: List[BaseMessage] = prompt.format_messages(**variables)

    try:
        resp = _invoke_with_backoff(llm, messages, max_attempts=3)
        usage = _extract_usage(getattr(resp, "response_metadata", {}) or {})
        return str(getattr(resp, "content", "") or "").strip(), usage
    except ResourceExhausted:
        return (
            "Vertex AI rate limit (429). Please wait 30–60 seconds and try again.",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    except Exception as e:
        return (
            f"LLM error: {type(e).__name__}: {e}",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )


# -----------------------------
# Public APIs
# -----------------------------
def generate_answer_with_usage(
    question: str,
    context: str,
    max_output_tokens: int = 512,
    temperature: float = 0.2,
) -> Tuple[str, Dict[str, int]]:
    return _safe_llm_invoke(
        QA_PROMPT,
        {"context": context, "question": question},
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def generate_doc_task_with_usage(
    task: str,
    text: str,
    max_output_tokens: int = 700,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, int]]:
    return _safe_llm_invoke(
        DOC_TASK_PROMPT,
        {"text": text, "task": task},
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )