from typing import List, Optional, TypedDict
from graph.schemas import EmployeeInfo, ExtractedReceipt, PolicyClause, ReviewDecision


class ReceiptState(TypedDict):
    file_name: str
    file_path: str
    file_type: str
    extracted: Optional[ExtractedReceipt]
    clauses: List[PolicyClause]
    decision: Optional[ReviewDecision]


class ReviewState(TypedDict):
    submission_id: str
    employee: EmployeeInfo
    receipts: List[ReceiptState]
    current_index: int
    errors: List[str]
    summary: Optional[dict]