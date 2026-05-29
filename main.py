import os
import uuid
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq

from graph.workflow import build_review_graph
from graph.schemas import EmployeeInfo

load_dotenv()

app = FastAPI()

# Fallback mechanism: Uses environment variable if found, otherwise falls back to your hardcoded string
GROQ_KEY = os.environ.get("GROQ_API_KEY") or "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"
groq_client = Groq(api_key=GROQ_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://green-growth-ui.vercel.app",  # Allowed your live Vercel application link
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "backend running"}


@app.get("/ping")
def ping():
    return {"message": "backend alive"}


@app.get("/stream-test")
def stream_test():
    def generate():
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Explain why this expense review system is useful in 5 bullet points.",
                }
            ],
            temperature=0.2,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/review")
async def review_submission(
    employee_name: str = Form(...),
    department: str = Form(...),
    grade: str = Form(...),
    files: List[UploadFile] = File(...),
):
    # Build graph lazily so the host can start the server before loading heavy dependencies
    review_graph = build_review_graph()

    submission_id = str(uuid.uuid4())
    upload_dir = f"uploads/{submission_id}"
    os.makedirs(upload_dir, exist_ok=True)

    receipts = []

    for uploaded_file in files:
        file_path = f"{upload_dir}/{uploaded_file.filename}"

        with open(file_path, "wb") as file:
            file.write(await uploaded_file.read())

        receipts.append(
            {
                "file_name": uploaded_file.filename,
                "file_path": file_path,
                "file_type": uploaded_file.content_type,
                "extracted": None,
                "clauses": [],
                "decision": None,
            }
        )

    employee = EmployeeInfo(
        employee_id=str(uuid.uuid4()),
        name=employee_name,
        department=department,
        grade=grade,
    )

    initial_state = {
        "submission_id": submission_id,
        "employee": employee,
        "receipts": receipts,
        "current_index": 0,
        "errors": [],
        "summary": None,
    }

    result = review_graph.invoke(initial_state)

    return result
