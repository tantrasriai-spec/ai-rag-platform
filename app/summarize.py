# app/summarize.py
"""
Document Task Pipeline (Production Grade)

This module handles document-wide operations that should NOT use
retrieval-based Q&A pipelines.

Supported tasks:
- Full document summary
- Highlights and lowlights
- Definitions extraction

Design goals:
1. Separate document analysis from retrieval-based Q&A
2. Process large documents safely with map-reduce
3. Keep outputs deterministic for production
4. Keep prompts structured for better UI rendering
5. Support backward-compatible task aliases
"""

# Import typing helpers used throughout this module.
from typing import Dict, Iterable, List, Literal, Optional, Tuple

# Import the shared LLM wrapper used for document-task generation.
from llm_vertex import generate_doc_task_with_usage


# ------------------------------------------------------------
# Supported document task types
# ------------------------------------------------------------

# Restrict supported task names for clarity and safer routing.
DocTask = Literal[
    "summary",
    "highlights_lowlights",
    "definitions_only",
]


# ------------------------------------------------------------
# Natural-language trigger phrases
# Used when the UI sends plain text instead of explicit task IDs
# ------------------------------------------------------------

# Trigger phrases for summary requests.
SUMMARY_TRIGGERS = (
    "summarize",
    "summarise",
    "summary",
    "summarize this document",
    "summarize the document",
    "summary of the document",
    "tldr",
    "tl;dr",
)

# Trigger phrases for highlights/lowlights requests.
HIGHLIGHTS_TRIGGERS = (
    "highlights",
    "lowlights",
    "highlights and lowlights",
    "give highlights and lowlights",
    "give highlights and lowlights for this document",
    "key takeaways",
)

# Trigger phrases for definitions requests.
DEFINITION_TRIGGERS = (
    "definitions",
    "definition",
    "glossary",
    "key terms",
    "terminology",
    "definitions only",
    "compile all definitions",
)


# ------------------------------------------------------------
# Detect which document task the user requested
# ------------------------------------------------------------

def detect_doc_task(question: str) -> Optional[DocTask]:
    """
    Infer the document task from natural-language question text.

    Returns:
        - "definitions_only"
        - "highlights_lowlights"
        - "summary"
        - None if no document task is detected
    """
    # Normalize the incoming question for matching.
    q = (question or "").strip().lower()

    # If the question is empty, no task can be inferred.
    if not q:
        return None

    # Definitions takes priority if the question mentions glossary-style terms.
    if any(t in q for t in DEFINITION_TRIGGERS):
        return "definitions_only"

    # Highlights/lowlights takes priority next.
    if any(t in q for t in HIGHLIGHTS_TRIGGERS):
        return "highlights_lowlights"

    # Summary is checked last.
    if any(t in q for t in SUMMARY_TRIGGERS):
        return "summary"

    # No matching task found.
    return None


# ------------------------------------------------------------
# Utility: Split list into batches
# ------------------------------------------------------------

def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """
    Yield chunks of a list for batch processing.
    """
    # Ensure batch size is always at least 1.
    batch_size = max(1, int(batch_size))

    # Yield slices of the input list.
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ------------------------------------------------------------
# Utility: Join chunks safely with size limits
# ------------------------------------------------------------

def _join_chunks(chunks: List[str], max_chars: int) -> str:
    """
    Combine text chunks without exceeding a character limit.
    """
    # Collect accepted chunks here.
    out: List[str] = []

    # Track total character count.
    total = 0

    # Process each chunk in order.
    for c in chunks:
        # Normalize chunk text.
        c = (c or "").strip()

        # Skip empty chunks.
        if not c:
            continue

        # Defensively trim very large chunks so one chunk does not dominate the prompt.
        if len(c) > 3000:
            c = c[:3000]

        # Stop before exceeding the prompt size budget.
        if total + len(c) + 2 > max_chars:
            break

        # Keep the chunk.
        out.append(c)

        # Account for the chunk and separator characters.
        total += len(c) + 2

    # Join accepted chunks using blank lines.
    return "\n\n".join(out)


# ------------------------------------------------------------
# Utility: Accumulate token usage across batches
# ------------------------------------------------------------

def _add_usage(total: Dict[str, int], part: Dict[str, int]) -> None:
    """
    Aggregate token usage from multiple LLM calls.
    """
    # Add prompt token usage.
    total["prompt_tokens"] += int(part.get("prompt_tokens", 0) or 0)

    # Add completion token usage.
    total["completion_tokens"] += int(part.get("completion_tokens", 0) or 0)

    # Add total token usage.
    total["total_tokens"] += int(part.get("total_tokens", 0) or 0)


# ------------------------------------------------------------
# Core Map-Reduce Pipeline
# ------------------------------------------------------------

def _map_reduce(
    chunk_texts: List[str],
    map_task: str,
    reduce_task: str,
    *,
    batch_size: int = 8,
    max_chars_per_batch: int = 12000,
    map_output_tokens: int = 400,
    reduce_output_tokens: int = 800,
) -> Tuple[str, Dict[str, int]]:
    """
    Execute a Map-Reduce style LLM workflow.
    """
    # Track total token usage across the full workflow.
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Store intermediate map outputs here.
    partials: List[str] = []

    # -------------------------
    # Map phase
    # -------------------------
    for batch in batched(chunk_texts, batch_size):
        # Build one prompt-safe text block from this batch.
        text = _join_chunks(batch, max_chars=max_chars_per_batch)

        # Skip empty text.
        if not text.strip():
            continue

        # Run the map-stage prompt for this batch.
        part, usage = generate_doc_task_with_usage(
            task=map_task,
            text=text,
            max_output_tokens=map_output_tokens,
            temperature=0.0,  # deterministic for production consistency
        )

        # Aggregate usage.
        _add_usage(usage_total, usage)

        # Keep non-empty map output.
        if part.strip():
            partials.append(part.strip())

    # If nothing was extracted, return a safe fallback.
    if not partials:
        return "No content available for this document task.", usage_total

    # -------------------------
    # Reduce phase
    # -------------------------

    # Combine partial summaries/insights into one reduce prompt.
    combined = _join_chunks(partials, max_chars=20000)

    # Run the reduce-stage prompt.
    final_text, final_usage = generate_doc_task_with_usage(
        task=reduce_task,
        text=combined,
        max_output_tokens=reduce_output_tokens,
        temperature=0.0,
    )

    # Aggregate reduce usage.
    _add_usage(usage_total, final_usage)

    # Return the final text and full token usage.
    return final_text.strip(), usage_total


# ------------------------------------------------------------
# Public entry point for document tasks
# ------------------------------------------------------------

def run_doc_task(task: str, chunk_texts: List[str]) -> Tuple[str, Dict[str, int]]:
    """
    Main dispatcher for document-wide operations.
    """
    # Guard against empty document content.
    if not chunk_texts:
        return (
            "No document content found.",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # Normalize requested task name.
    task = (task or "").strip().lower()

    # Backward-compatible aliases so older UI payloads still work.
    aliases = {
        "summarize": "summary",
        "doc_summary": "summary",
        "highlights": "highlights_lowlights",
        "lowlights": "highlights_lowlights",
        "definitions": "definitions_only",
        "glossary": "definitions_only",
        "definitions_highlights": "definitions_only",
    }

    # Resolve aliases to canonical task names.
    task = aliases.get(task, task)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    if task == "summary":
        return _map_reduce(
            chunk_texts,

            # Map prompt: extract partial summaries with structure and full sentences.
            map_task=(
                "Read this document text and create a strong partial summary.\n\n"
                "Return exactly:\n"
                "SUMMARY\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n\n"
                "Rules:\n"
                "- Use only the document text\n"
                "- Focus on the most important ideas, themes, and concepts\n"
                "- Keep every bullet as a complete sentence\n"
                "- Do not truncate sentences\n"
                "- Do not say 'I don't know'"
            ),

            # Reduce prompt: merge all partial summaries into one fuller summary.
            reduce_task=(
                "Combine the partial summaries into one final structured summary.\n\n"
                "Return exactly:\n"
                "SUMMARY\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n\n"
                "TL;DR\n"
                "1 short paragraph.\n\n"
                "Rules:\n"
                "- Keep all important ideas found across the partial summaries\n"
                "- Remove duplicates\n"
                "- Keep every sentence complete\n"
                "- Do not truncate the output\n"
                "- Use only document content"
            ),

            # Increase tokens so summary has enough room to be complete.
            map_output_tokens=500,
            reduce_output_tokens=1200,
        )

    # --------------------------------------------------------
    # HIGHLIGHTS + LOWLIGHTS
    # Keep this version strong because you said this style was better.
    # --------------------------------------------------------
    if task == "highlights_lowlights":
        return _map_reduce(
            chunk_texts,

            # Map prompt: extract both positives and negatives from each batch.
            map_task=(
                "Analyze this document text and extract two sections.\n\n"
                "Return exactly:\n"
                "HIGHLIGHTS\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n\n"
                "LOWLIGHTS\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n\n"
                "Rules:\n"
                "- Use only the document text\n"
                "- Highlights should be key ideas, benefits, capabilities, or important concepts\n"
                "- Lowlights should be limitations, risks, concerns, drawbacks, or challenges explicitly stated in the text\n"
                "- Each bullet must be a full sentence\n"
                "- Do not include incomplete bullets\n"
                "- Do not say 'I don't know'\n"
                "- If the text has no real lowlights, write: • No explicit lowlights were stated in the document"
            ),

            # Reduce prompt: preserve the stronger full-section format.
            reduce_task=(
                "Merge the partial analyses into one final answer.\n\n"
                "Return exactly:\n"
                "HIGHLIGHTS\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n\n"
                "LOWLIGHTS\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n"
                "• complete sentence\n\n"
                "Rules:\n"
                "- Remove duplicates\n"
                "- Keep every bullet as a full sentence\n"
                "- Use only document content\n"
                "- Do not say 'I don't know'\n"
                "- Do not include incomplete bullets\n"
                "- Keep as many meaningful points as the document supports"
            ),

            # Preserve a richer output budget so the answer does not collapse to one bullet.
            map_output_tokens=500,
            reduce_output_tokens=1200,
        )

    # --------------------------------------------------------
    # DEFINITIONS ONLY
    # Keep unchanged from your working definitions direction.
    # --------------------------------------------------------
    if task == "definitions_only":
        return _map_reduce(
            chunk_texts,
            map_task=(
                "Extract ALL explicit definitions, term explanations, concept descriptions, "
                "model descriptions, technique descriptions, risk definitions, governance terms, "
                "privacy terms, and regulation descriptions from this text.\n\n"
                "Return exactly:\n"
                "DEFINITIONS\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n\n"
                "Rules:\n"
                "- Extract as many definitions as the text supports\n"
                "- Use only terms present in the document\n"
                "- Prefer explicit definitions and clear concept descriptions\n"
                "- Include model names, AI techniques, AI risks, privacy concepts, compliance terms, and governance terms\n"
                "- Each item must be a complete sentence\n"
                "- Do not add a HIGHLIGHTS section\n"
                "- Do not add a GLOSSARY section\n"
                "- Do not say 'I don't know'"
            ),
            reduce_task=(
                "Merge all extracted definitions into one complete final output.\n\n"
                "Return exactly:\n"
                "DEFINITIONS\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n"
                "• TERM: complete definition sentence\n\n"
                "Rules:\n"
                "- Keep all important definitions found across batches\n"
                "- Deduplicate repeated terms\n"
                "- Prefer maximum definition coverage over brevity\n"
                "- Keep each item as a complete sentence\n"
                "- Do not add a HIGHLIGHTS section\n"
                "- Do not add a GLOSSARY section\n"
                "- Use only document content"
            ),
            batch_size=6,
            max_chars_per_batch=14000,
            map_output_tokens=900,
            reduce_output_tokens=2600,
        )

    # Fallback for unknown task names.
    return (
        f"Unsupported document task: {task}",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )