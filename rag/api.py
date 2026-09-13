"""HTTP API for semantic search over the local industrial knowledge base."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .rag_service import search_knowledge_base
except ImportError:
    from rag_service import search_knowledge_base


app = FastAPI(title="Sovereign_X Industrial Multimodal RAG API")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Question to search for")
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    status: str
    query: str
    evidence_count: int = 0
    evidence: list[dict[str, Any]]


@app.post("/api/search", response_model=SearchResponse)
def semantic_search_api(payload: SearchRequest) -> SearchResponse:
    """Return the nearest chunks and their source metadata."""
    try:
        result = search_knowledge_base(payload.query, payload.top_k)
        return SearchResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
