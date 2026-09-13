from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ingestion_pipeline import IngestionPipeline
from rag import RAGStore

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in a .env file before running the multimodal RAG demo.")
    return OpenAI(api_key=api_key)


def embed_texts(values: list[str]) -> list[list[float]]:
    if not values:
        return []
    response = openai_client().embeddings.create(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        input=values,
    )
    return [item.embedding for item in response.data]


def describe_image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")

    response = openai_client().responses.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image for retrieval. Keep it factual and concise."},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}", "detail": "high"},
                ],
            }
        ],
    )
    return response.output_text.strip()


def ask_question(question: str, top_k: int = 5) -> dict:
    store = RAGStore(ROOT / "rag.sqlite3")
    hits = store.search(embed_texts([question])[0], top_k=top_k)
    if not hits:
        return {"answer": "No indexed content yet. Ingest a file before asking a question.", "sources": []}

    context = "\n\n".join(
        f"[Source: {hit.source}; page: {hit.page or 'n/a'}; modality: {hit.modality}]\n{hit.content}"
        for hit in hits
    )

    answer = openai_client().responses.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        input=f"Question: {question}\n\nRetrieved context:\n{context}",
    )
    return {
        "answer": answer.output_text,
        "sources": [
            {
                "source": hit.source,
                "page": hit.page,
                "modality": hit.modality,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ],
    }


def ingest_path(path: str | Path) -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    payload = source.read_bytes()
    db = RAGStore(ROOT / "rag.sqlite3")
    pipeline = IngestionPipeline(db, UPLOAD_DIR, describe_image, embed_texts)
    registration = pipeline.register(source.name, payload)
    pipeline.run(registration.job_id)

    return {
        "job_id": registration.job_id,
        "document_id": registration.document_id,
        "duplicate": registration.duplicate,
    }


if __name__ == "__main__":
    print("Multimodal RAG demo")
    print("1) Add a .txt, .md, .pdf, or image file to the uploads directory")
    print("2) Run: python multimodal_rag.py --ingest path/to/file.txt")
    print("3) Then ask a question with: python multimodal_rag.py --ask \"What is in the document?\"")
