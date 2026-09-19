import os

from dotenv import load_dotenv

load_dotenv()  # must run before agent.graph imports the LLM client

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from agent.graph import run as run_agent
from agent.tools import ingest_url

app = FastAPI(title="Agentic RAG Service")


class AskRequest(BaseModel):
    query: str
    max_retries: int = 2


class IngestRequest(BaseModel):
    url: str
    chunk_size: int = 1000
    chunk_overlap: int = 100


@app.post("/ingest", summary="Ingest a URL into the RAG knowledge base (proxies Rag_service's /ingest)")
async def ingest(request: IngestRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        result = ingest_url(url, chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
    except requests.HTTPError as e:
        detail = str(e)
        status_code = 502
        if e.response is not None:
            status_code = e.response.status_code
            try:
                detail = e.response.json().get("detail", detail)
            except ValueError:
                pass
        logger.error(f"Rag_service rejected ingest for url={url!r}: {detail}")
        raise HTTPException(status_code=status_code, detail=detail)
    except requests.RequestException as e:
        logger.exception(f"Could not reach Rag_service for ingest: {e}")
        raise HTTPException(status_code=502, detail=f"Could not reach RAG service: {e}")

    return JSONResponse(content=result)


@app.post("/ask", summary="Answer a question using the corrective-RAG agent")
async def ask(request: AskRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = run_agent(query, max_retries=request.max_retries)
    except Exception as e:
        logger.exception(f"Agent run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "retries": result["retry_count"],
        "used_web_search": result["used_web_search"],
    })


def _cli():
    print("=" * 60)
    print("Agentic RAG -- type 'exit' to quit")
    print("=" * 60)
    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "exit":
            break
        result = run_agent(query)
        print(f"\n[Agent]: {result['answer']}")
        print(f"[retries={result['retry_count']} used_web_search={result['used_web_search']}]")


if __name__ == "__main__":
    import sys

    if "--cli" in sys.argv:
        _cli()
    else:
        import uvicorn
        port = int(os.getenv("PORT", 7100))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
