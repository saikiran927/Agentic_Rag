from typing import TypedDict


class Chunk(TypedDict):
    id: int
    source: str
    chunk_index: int
    content: str


class AgentState(TypedDict):
    """State threaded through every node of the graph."""

    original_query: str      # the user's question, never overwritten
    query: str                # the query actually sent to retrieval (may be rewritten)
    context: list[Chunk]       # chunks gathered so far
    sufficient: bool           # verdict from the grade node
    grade_reasoning: str       # why the grader made that call (useful for tracing/debug)
    retry_count: int
    max_retries: int
    used_web_search: bool
    answer: str
    sources: list[dict]
