import json
import os
from collections.abc import Mapping

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# Database & Embedding Configuration
DB_URI = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:your_secure_password_here@localhost:5432/sovereign_x",
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
MIN_CONFIDENCE = 0.35


def search_knowledge_base(query: str, top_k: int = 5) -> dict:
    """Return the nearest document chunks and their source metadata."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")

    query_embedding = embedding_model.encode(
        query, normalize_embeddings=True
    ).tolist()
    query_vector = Vector(query_embedding)

    with psycopg.connect(DB_URI) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    filename,
                    page,
                    section,
                    document_version,
                    access_permission,
                    chunk_text,
                    embedding <=> %s AS cosine_distance
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vector, query_vector, top_k),
            )
            rows = cursor.fetchall()

    results = []
    for row in rows:
        confidence = max(0.0, 1.0 - float(row[6]))
        if confidence < MIN_CONFIDENCE:
            continue
        results.append(
            {
                "claim": row[5],
                "claim_context": row[5],
                "source_document": row[0],
                "page": row[1],
                "section": row[2],
                "document_version": row[3],
                "access_permission": row[4],
                "cosine_distance": round(float(row[6]), 4),
                "confidence": round(confidence, 4),
            }
        )

    if not results:
        return {
            "status": "insufficient_evidence",
            "query": query,
            "message": "Local documents do not contain sufficient evidence to support this claim.",
            "evidence_count": 0,
            "evidence": [],
        }

    return {
        "status": "success",
        "query": query,
        "evidence_count": len(results),
        "evidence": results,
    }


def search(request: Mapping[str, object]) -> dict:
    """Search using a request such as {"query": "...", "top_k": 5}."""
    query = request.get("query")
    top_k = request.get("top_k", 5)
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    return search_knowledge_base(query, top_k)


if __name__ == "__main__":
    # Test execution
    test_query = "What is the inspection limit for pump vibration?"
    print(f"Running test query: '{test_query}'\n")

    response = search({"query": test_query, "top_k": 5})
    print(json.dumps(response, indent=2))
