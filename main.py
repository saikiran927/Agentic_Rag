from dotenv import load_dotenv

load_dotenv()  # must run before agent.graph imports the LLM client

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from agent.graph import run as run_agent

app = FastAPI(title="Agentic RAG Service")


class AskRequest(BaseModel):
    query: str
    max_retries: int = 2


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
        uvicorn.run("main:app", host="localhost", port=7100, reload=False)
