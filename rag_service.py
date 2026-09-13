import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "./local_rag_db"

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_or_create_collection(
    name="industrial_knowledge", embedding_function=embedding_func  # type: ignore[arg-type]
)


def search_knowledge_base(
    query: str, top_k: int = 3, distance_threshold: float = 1.2
) -> dict:
    """Searches local vector DB and returns grounded evidence or insufficient_evidence status."""
    results = collection.query(query_texts=[query], n_results=top_k)

    # Check if results exist
    if (
        not results
        or not results.get("documents")
        or not results["documents"]
    ):
        return {
            "status": "insufficient_evidence",
            "message": "No local documents found matching the query.",
            "evidence": [],
        }

    document_groups = results.get("documents") or []
    metadata_groups = results.get("metadatas") or []
    distance_groups = results.get("distances") or []
    documents = document_groups[0] if document_groups else []
    metadatas = metadata_groups[0] if metadata_groups else [{} for _ in documents]
    distances = distance_groups[0] if distance_groups else [0.5 for _ in documents]

    # Filter by distance threshold
    filtered_evidence = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist <= distance_threshold:
            # Confidence score calculation based on distance
            confidence = round(max(0.0, min(1.0, 1.0 - (dist / 2.0))), 2)

            filtered_evidence.append(
                {
                    "claim_context": doc,
                    "source_document": (meta or {}).get("filename", "Unknown"),
                    "page": (meta or {}).get("page", 1),
                    "section": (meta or {}).get("section", "General"),
                    "confidence": confidence,
                }
            )

    # If all items were filtered out due to low distance/confidence
    if not filtered_evidence:
        return {
            "status": "insufficient_evidence",
            "message": "Local sources do not contain sufficient evidence to answer this query safely.",
            "evidence": [],
        }

    return {
        "status": "success",
        "query": query,
        "evidence_count": len(filtered_evidence),
        "evidence": filtered_evidence,
    }


if __name__ == "__main__":
    # Quick Test Execution
    test_query = "What is the inspection limit for pump vibration?"
    print(f"🔎 Querying local RAG for: '{test_query}'\n")
    search_res = search_knowledge_base(test_query, top_k=2)

    import json

    print(json.dumps(search_res, indent=2))