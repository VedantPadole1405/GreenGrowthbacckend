import os
import json

from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader

from graph.schemas import ExtractedReceipt

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_receipt(file_path: str, file_type: str) -> ExtractedReceipt:

    if file_type == "application/pdf":
        raw_text = extract_pdf_text(file_path)

    elif file_type == "text/plain":
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

    else:
        raw_text = "Image receipt uploaded."

    prompt = f"""
You are an expert receipt extraction AI.

Extract structured information from this receipt.

Return ONLY valid JSON:

{{
  "merchant": "",
  "date": "",
  "amount": 0,
  "currency": "USD",
  "category": "",
  "raw_text": ""
}}

Receipt text:
{raw_text}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Extract receipt information and output only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content

    data = json.loads(content)

    return ExtractedReceipt(
        merchant=data.get("merchant"),
        date=data.get("date"),
        amount=data.get("amount"),
        currency=data.get("currency", "USD"),
        category=data.get("category"),
        raw_text=data.get("raw_text", raw_text),
    )