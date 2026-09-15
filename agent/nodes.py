import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.tools import retrieve_context, web_search

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=GOOGLE_API_KEY)


GENERATE_SYSTEM_PROMPT = """You are a Retrieval-Augmented Generation (RAG) assistant. Answer ONLY using the context provided below.
Rules:
- If the answer is not present in the context, respond with: "I don't have enough information to answer this question."
- Do NOT use any external knowledge outside the context.
- Keep your answer short, clear and on point.
- Do not repeat the question back.
- Do not make up or assume any information.
"""


class GradeResult(BaseModel):
    sufficient: bool = Field(description="True if the context is enough to answer the question")
    reasoning: str = Field(description="One short sentence explaining the verdict")
    rewritten_query: str = Field(
        default="",
        description="If not sufficient, a rewritten/refined search query that would retrieve better context. Empty otherwise.",
    )


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(c.get("content", "") for c in chunks) if chunks else ""


def _extract_text(content) -> str:
    """Normalize an AIMessage.content value into plain text.

    langchain_google_genai sometimes returns a plain string and sometimes a
    list of content blocks (e.g. [{'type': 'text', 'text': '...'}]) depending
    on the response shape Gemini sends back. Handle both.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return str(content)


def retrieve_node(state: AgentState) -> dict:
    chunks = retrieve_context(state["query"])
    return {"context": state["context"] + chunks}


def grade_node(state: AgentState) -> dict:
    context_text = format_context(state["context"])
    grader = llm.with_structured_output(GradeResult)

    result: GradeResult = grader.invoke([
        SystemMessage(content=(
            "You judge whether retrieved context is sufficient to answer a user's question. "
            "Be strict: if the context does not directly address the question, mark it insufficient "
            "and propose a better search query."
        )),
        HumanMessage(content=f"Question: {state['original_query']}\n\nRetrieved context:\n{context_text or '(empty)'}"),
    ])

    logger.info(f"grade_node: sufficient={result.sufficient} reasoning={result.reasoning}")

    return {
        "sufficient": result.sufficient,
        "grade_reasoning": result.reasoning,
        "query": result.rewritten_query or state["query"],
        "retry_count": state["retry_count"] + 1,
    }


def web_search_node(state: AgentState) -> dict:
    results = web_search(state["original_query"])
    return {"context": state["context"] + results, "used_web_search": True}


def generate_node(state: AgentState) -> dict:
    context_text = format_context(state["context"])

    if not context_text:
        return {
            "answer": "I don't have enough information to answer this question.",
            "sources": [],
        }

    prompt = f"""
User Question: {state['original_query']}
Context:
{context_text}
"""
    response = llm.invoke([
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    sources = [
        {"id": c.get("id"), "source": c.get("source"), "chunk_index": c.get("chunk_index")}
        for c in state["context"]
    ]

    return {"answer": _extract_text(response.content).strip(), "sources": sources}


def route_after_grade(state: AgentState) -> str:
    if state["sufficient"]:
        return "generate"
    if state["retry_count"] < state["max_retries"]:
        return "retrieve"
    if not state["used_web_search"]:
        return "web_search"
    return "generate"
