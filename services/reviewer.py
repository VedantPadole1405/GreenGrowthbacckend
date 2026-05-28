import os
import json
from typing import List

from dotenv import load_dotenv
from groq import Groq

from graph.schemas import ExtractedReceipt, EmployeeInfo, PolicyClause, ReviewDecision

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def review_receipt(
    receipt: ExtractedReceipt,
    employee: EmployeeInfo,
    clauses: List[PolicyClause],
) -> ReviewDecision:

    if receipt is None:
        return ReviewDecision(
            verdict="needs_review",
            reasoning="Receipt extraction failed, so a human reviewer must inspect this item.",
            reviewer_action="Ask the employee to re-upload a clearer receipt or manually inspect the document.",
            confidence=0.1,
            citations=[],
        )

    if not clauses:
        return ReviewDecision(
            verdict="needs_review",
            reasoning="No relevant policy clauses were retrieved, so the system cannot make a grounded decision.",
            reviewer_action="Send this item to manual review because the system could not find supporting policy evidence.",
            confidence=0.2,
            citations=[],
        )

    policy_context = "\n\n".join(
        [
            f"Policy Index: {index}\n"
            f"Policy ID: {clause.policy_id}\n"
            f"Section: {clause.section}\n"
            f"Quote: {clause.quote}\n"
            f"Source: {clause.source_file}\n"
            for index, clause in enumerate(clauses)
        ]
    )

    prompt = f"""
You are an AI finance policy reviewer.

Your job is to help a human finance reviewer pre-review one expense receipt.

Rules:
1. Use ONLY the provided policy clauses.
2. Do not invent company policy.
3. If the receipt is missing important fields, return needs_review.
4. If the policy evidence is weak, return needs_review.
5. The reviewer_action must be practical and specific.
6. citation_indexes must refer to the Policy Index numbers shown below.

Employee:
Name: {employee.name}
Department: {employee.department}
Grade: {employee.grade}
Trip purpose: {employee.trip_purpose}

Receipt:
Merchant: {receipt.merchant}
Date: {receipt.date}
Amount: {receipt.amount}
Currency: {receipt.currency}
Category: {receipt.category}

Raw receipt text:
{receipt.raw_text}

Policy clauses:
{policy_context}

Return ONLY valid JSON with this exact schema:
{{
  "verdict": "compliant | flagged | rejected | needs_review",
  "reasoning": "short explanation of why this verdict was chosen",
  "reviewer_action": "clear next step the human finance reviewer should take",
  "confidence": 0.0,
  "citation_indexes": [0]
}}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a careful finance compliance reviewer. Output only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    data = json.loads(content)

    citation_indexes = data.get("citation_indexes", [])
    selected_citations = []

    for index in citation_indexes:
        if isinstance(index, int) and 0 <= index < len(clauses):
            selected_citations.append(clauses[index])

    return ReviewDecision(
        verdict=data.get("verdict", "needs_review"),
        reasoning=data.get("reasoning", "The model did not provide reasoning."),
        reviewer_action=data.get(
            "reviewer_action",
            "Escalate this item to a human reviewer."
        ),
        confidence=float(data.get("confidence", 0.3)),
        citations=selected_citations or clauses[:1],
    )