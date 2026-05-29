import os
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
from dotenv import load_dotenv

from graph.workflow import build_review_graph
from graph.schemas import EmployeeInfo

load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Create a placeholder dictionary to hold your graph
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs right AFTER the server binds to the port
    print("Loading review graph...")
    app_state["review_graph"] = build_review_graph()
    print("Review graph loaded successfully!")
    yield
    # Clean up on shutdown if needed
    app_state.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/stream-test")
def stream_test():
    def generate():
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Explain why this expense review system is useful in 5 bullet points."
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://green-growth-ui.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "backend running"}


@app.post("/review")
async def review_submission(
    employee_name: str = Form(...),
    department: str = Form(...),
    grade: str = Form(...),
    files: List[UploadFile] = File(...),
):
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

    # Pull the review graph from the app state instead of the global scope
    graph = app_state.get("review_graph")
    if not graph:
        return {"error": "Graph not initialized yet"}

    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
