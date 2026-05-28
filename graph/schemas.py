from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EmployeeInfo(BaseModel):
    employee_id: str
    name: str
    department: str
    grade: str
    manager: Optional[str] = None
    trip_purpose: Optional[str] = None
    trip_start_date: Optional[str] = None
    trip_end_date: Optional[str] = None


class ExtractedReceipt(BaseModel):
    merchant: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    currency: str = "USD"
    category: Optional[str] = None
    raw_text: str


class PolicyClause(BaseModel):
    policy_id: str
    section: Optional[str] = None
    quote: str
    source_file: str
    score: float


class ReviewDecision(BaseModel):
    verdict: Literal["compliant", "flagged", "rejected", "needs_review"]
    reasoning: str
    reviewer_action: str
    confidence: float = Field(ge=0, le=1)
    citations: List[PolicyClause]