import chromadb
from sentence_transformers import SentenceTransformer
from typing import List

from graph.schemas import PolicyClause


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "northwind_policies"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve_policy_clauses(query: str, top_k: int = 5) -> List[PolicyClause]:
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    clauses = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        score = 1 / (1 + distance)

        clauses.append(
            PolicyClause(
                policy_id=metadata.get("policy_id", "UNKNOWN"),
                section=str(metadata.get("chunk_index", "")),
                quote=document,
                source_file=metadata.get("source_file", "unknown"),
                score=score,
            )
        )

    return clauses