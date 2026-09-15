"""
Retrieval capabilities used by the graph.

These are plain functions today, called directly by graph nodes over HTTP.
When the MCP tool server exists, only this file changes -- swap the HTTP
call for an MCP client call. The graph logic in graph.py stays the same.
"""
import os

import requests
from loguru import logger
from tavily import TavilyClient

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:7000")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Call Rag_service's /search endpoint for raw, ungenerated context chunks."""
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        logger.info(f"retrieve_context: {len(results)} chunks for query={query!r}")
        return results
    except requests.RequestException as e:
        logger.error(f"retrieve_context failed: {e}")
        return []


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Escalation tool used when internal retrieval is not sufficient."""
    if not TAVILY_API_KEY:
        logger.warning("web_search called but TAVILY_API_KEY is not set; skipping")
        return []

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)
        results = [
            {
                "id": -1,
                "source": item.get("url", "web"),
                "chunk_index": idx,
                "content": item.get("content", ""),
            }
            for idx, item in enumerate(response.get("results", []))
        ]
        logger.info(f"web_search: {len(results)} results for query={query!r}")
        return results
    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return []
