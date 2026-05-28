from graph.state import ReviewState
from services.extractor import extract_receipt
from services.retriever import retrieve_policy_clauses
from services.reviewer import review_receipt


def extract_receipt_node(state: ReviewState) -> ReviewState:
    index = state["current_index"]
    receipt = state["receipts"][index]

    extracted = extract_receipt(
        file_path=receipt["file_path"],
        file_type=receipt["file_type"],
    )

    state["receipts"][index]["extracted"] = extracted
    return state


def retrieve_policy_node(state: ReviewState) -> ReviewState:
    index = state["current_index"]
    receipt = state["receipts"][index]
    extracted = receipt["extracted"]

    query = f"""
    Employee: {state["employee"].name}
    Grade: {state["employee"].grade}
    Department: {state["employee"].department}
    Trip purpose: {state["employee"].trip_purpose}

    Receipt text:
    {extracted.raw_text if extracted else ""}
    """

    clauses = retrieve_policy_clauses(query)
    state["receipts"][index]["clauses"] = clauses
    return state


def review_receipt_node(state: ReviewState) -> ReviewState:
    index = state["current_index"]
    receipt = state["receipts"][index]

    decision = review_receipt(
        receipt=receipt["extracted"],
        employee=state["employee"],
        clauses=receipt["clauses"],
    )

    state["receipts"][index]["decision"] = decision
    return state


def move_next_receipt_node(state: ReviewState) -> ReviewState:
    state["current_index"] += 1
    return state


def summarize_node(state: ReviewState) -> ReviewState:
    verdict_counts = {
        "compliant": 0,
        "flagged": 0,
        "rejected": 0,
        "needs_review": 0,
    }

    for receipt in state["receipts"]:
        decision = receipt["decision"]
        if decision:
            verdict_counts[decision.verdict] += 1

    state["summary"] = verdict_counts
    return state


def should_continue(state: ReviewState) -> str:
    if state["current_index"] < len(state["receipts"]):
        return "continue"

    return "done"