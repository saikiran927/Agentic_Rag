from langgraph.graph import END, StateGraph

from agent.nodes import generate_node, grade_node, retrieve_node, route_after_grade, web_search_node
from agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


agentic_rag_graph = build_graph()


def run(query: str, max_retries: int = 2) -> dict:
    initial_state: AgentState = {
        "original_query": query,
        "query": query,
        "context": [],
        "sufficient": False,
        "grade_reasoning": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "used_web_search": False,
        "answer": "",
        "sources": [],
    }
    return agentic_rag_graph.invoke(initial_state)
