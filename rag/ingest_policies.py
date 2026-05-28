import os
import uuid
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


POLICY_FOLDER = "policies"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "northwind_policies"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def ingest_policies():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    for file_name in os.listdir(POLICY_FOLDER):
        if not file_name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(POLICY_FOLDER, file_name)
        text = read_pdf(file_path)
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()

            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source_file": file_name,
                        "chunk_index": index,
                        "policy_id": file_name.replace(".pdf", ""),
                    }
                ],
            )

        print(f"Ingested {file_name} with {len(chunks)} chunks")

    print("Policy ingestion complete.")


if __name__ == "__main__":
    ingest_policies()