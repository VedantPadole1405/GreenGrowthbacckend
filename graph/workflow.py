from langgraph.graph import StateGraph, END
from graph.state import ReviewState
from graph.nodes import (
    extract_receipt_node,
    retrieve_policy_node,
    review_receipt_node,
    move_next_receipt_node,
    summarize_node,
    should_continue,
)


def build_review_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("extract_receipt", extract_receipt_node)
    graph.add_node("retrieve_policy", retrieve_policy_node)
    graph.add_node("review_receipt", review_receipt_node)
    graph.add_node("move_next_receipt", move_next_receipt_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("extract_receipt")

    graph.add_edge("extract_receipt", "retrieve_policy")
    graph.add_edge("retrieve_policy", "review_receipt")
    graph.add_edge("review_receipt", "move_next_receipt")

    graph.add_conditional_edges(
        "move_next_receipt",
        should_continue,
        {
            "continue": "extract_receipt",
            "done": "summarize",
        },
    )

    graph.add_edge("summarize", END)

    return graph.compile()