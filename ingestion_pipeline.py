"""The durable, observable ingestion pipeline for text, PDFs, and images."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

try:
    from .rag import RAGStore, chunk_text
except ImportError:  # pragma: no cover - supports running as a top-level script
    from rag import RAGStore, chunk_text

SUPPORTED_TYPES = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class IngestionError(ValueError):
    """Raised when an input cannot safely enter the pipeline."""


@dataclass(frozen=True)
class IngestionRegistration:
    job_id: str
    document_id: str
    duplicate: bool


@dataclass(frozen=True)
class ExtractedRecord:
    modality: str
    content: str | None
    page: int | None
    image_path: str | None


CaptionImage = Callable[[Path], str]
EmbedTexts = Callable[[list[str]], list[list[float]]]


class IngestionPipeline:
    def __init__(self, store: RAGStore, upload_directory: Path, caption_image: CaptionImage,
                 embed_texts: EmbedTexts, max_upload_bytes: int = MAX_UPLOAD_BYTES) -> None:
        self.store = store
        self.upload_directory = upload_directory
        self.caption_image = caption_image
        self.embed_texts = embed_texts
        self.max_upload_bytes = max_upload_bytes

    def register(self, filename: str, payload: bytes) -> IngestionRegistration:
        safe_name = Path(filename).name or "upload"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_TYPES:
            raise IngestionError("Supported formats: .txt, .md, .pdf, .png, .jpg, .jpeg, .webp")
        if not payload:
            raise IngestionError("The uploaded file is empty.")
        if len(payload) > self.max_upload_bytes:
            raise IngestionError(f"Uploads must be no larger than {self.max_upload_bytes // (1024 * 1024)} MB.")

        content_hash = hashlib.sha256(payload).hexdigest()
        existing = self.store.document_by_hash(content_hash)
        job_id = uuid.uuid4().hex
        if existing:
            document_id = str(existing["id"])
            self.store.create_job(job_id, document_id, "completed", "deduplicated", self._document_chunk_count(document_id))
            return IngestionRegistration(job_id, document_id, True)

        document_id = uuid.uuid4().hex
        destination_dir = self.upload_directory / document_id
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / safe_name
        destination.write_bytes(payload)
        self.store.create_document(document_id, safe_name, content_hash, suffix.lstrip("."), str(destination))
        self.store.create_job(job_id, document_id)
        return IngestionRegistration(job_id, document_id, False)

    def run(self, job_id: str) -> None:
        job = self.store.job(job_id)
        if not job:
            raise IngestionError("Ingestion job does not exist.")
        if job["status"] == "completed":
            return
        document_id = str(job["document_id"])
        document = self.store.document(document_id)
        if not document:
            raise IngestionError("Document does not exist.")
        try:
            self.store.update_document(document_id, "processing")
            self.store.update_job(job_id, status="processing", stage="extracting")
            extracted = self._extract(Path(str(document["stored_path"])))
            records = self._normalize_records(extracted)
            if not records:
                raise IngestionError("No readable text or supported images were found in this file.")

            self.store.update_job(job_id, status="processing", stage="embedding")
            embeddings = self._embed_in_batches([record.content or "" for record in records])
            if len(embeddings) != len(records):
                raise RuntimeError("Embedding provider returned an unexpected vector count.")

            self.store.clear_document_chunks(document_id)
            for record, embedding in zip(records, embeddings):
                self.store.add(
                    document_id=document_id,
                    source=str(document["source_name"]),
                    modality=record.modality,
                    content=record.content or "",
                    page=record.page,
                    image_path=record.image_path,
                    embedding=embedding,
                )
            self.store.update_document(document_id, "completed")
            self.store.update_job(job_id, status="completed", stage="completed", chunks_created=len(records))
        except Exception as exc:
            self.store.update_document(document_id, "failed")
            self.store.update_job(job_id, status="failed", stage="failed", error=str(exc))

    def _extract(self, path: Path) -> list[ExtractedRecord]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return [ExtractedRecord("text", path.read_text(encoding="utf-8", errors="replace"), None, None)]
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return [ExtractedRecord("image", None, None, str(path))]
        return self._extract_pdf(path)

    def _extract_pdf(self, path: Path) -> list[ExtractedRecord]:
        records: list[ExtractedRecord] = []
        image_directory = path.parent / "extracted-images"
        for page_number, page in enumerate(PdfReader(path).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append(ExtractedRecord("text", text, page_number, None))
            # pypdf exposes PDF image objects as ImageFile values with `name` and `data`.
            for image_number, image in enumerate(page.images):
                extension = Path(image.name).suffix.lower()
                if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                image_directory.mkdir(exist_ok=True)
                image_path = image_directory / f"page-{page_number}-image-{image_number}{extension}"
                image_path.write_bytes(image.data)
                records.append(ExtractedRecord("image", None, page_number, str(image_path)))
        return records

    def _normalize_records(self, extracted: list[ExtractedRecord]) -> list[ExtractedRecord]:
        records: list[ExtractedRecord] = []
        for record in extracted:
            if record.modality == "text" and record.content:
                records.extend(ExtractedRecord("text", part, record.page, None) for part in chunk_text(record.content))
            elif record.modality == "image" and record.image_path:
                caption = self.caption_image(Path(record.image_path)).strip()
                if caption:
                    records.append(ExtractedRecord("image", caption, record.page, record.image_path))
        return records

    def _embed_in_batches(self, values: list[str], batch_size: int = 96) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(values), batch_size):
            vectors.extend(self.embed_texts(values[start:start + batch_size]))
        return vectors

    def _document_chunk_count(self, document_id: str) -> int:

        with self.store._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)).fetchone()[0])

