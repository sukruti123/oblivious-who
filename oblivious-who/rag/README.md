# Backend

The backend exposes the multimodal RAG search API through FastAPI.

## Run locally

From the repository root:

```powershell
python -m uvicorn rag.api:app --reload
```

The search endpoint is available at `POST /api/search` and accepts a JSON body with `query` and optional `top_k` fields.
