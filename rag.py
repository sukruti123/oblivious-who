from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120


@dataclass
class SearchHit:
    document_id: str
    source: str
    modality: str
    content: str
    page: int | None
    score: float
    image_path: str | None = None


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + chunk_size - overlap, end - overlap)
    return chunks


class RAGStore:
    def __init__(self, db_path: str | Path = "rag.sqlite3") -> None:
        self.db_path = str(db_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    stage TEXT NOT NULL DEFAULT 'queued',
                    chunks_created INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    image_path TEXT,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
            )
            conn.commit()

    def create_document(self, document_id: str, source_name: str, content_hash: str, extension: str, stored_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents(id, source_name, content_hash, extension, stored_path, status) VALUES (?, ?, ?, ?, ?, 'queued')",
                (document_id, source_name, content_hash, extension, stored_path),
            )
            conn.commit()

    def document_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE content_hash = ? LIMIT 1", (content_hash,)).fetchone()
        return row

    def document(self, document_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()

    def create_job(self, job_id: str, document_id: str, status: str = "queued", stage: str = "queued", chunks_created: int = 0) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs(id, document_id, status, stage, chunks_created) VALUES (?, ?, ?, ?, ?)",
                (job_id, document_id, status, stage, chunks_created),
            )
            conn.commit()

    def update_document(self, document_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, document_id))
            conn.commit()

    def update_job(self, job_id: str, status: str | None = None, stage: str | None = None, chunks_created: int | None = None, error: str | None = None) -> None:
        if status is None and stage is None and chunks_created is None and error is None:
            return
        with self._connect() as conn:
            assignments: list[str] = []
            values: list[Any] = []
            if status is not None:
                assignments.append("status = ?")
                values.append(status)
            if stage is not None:
                assignments.append("stage = ?")
                values.append(stage)
            if chunks_created is not None:
                assignments.append("chunks_created = ?")
                values.append(chunks_created)
            if error is not None:
                assignments.append("error = ?")
                values.append(error)
            values.append(job_id)
            conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", values)
            conn.commit()

    def clear_document_chunks(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.commit()

    def add(self, document_id: str, source: str, modality: str, content: str, page: int | None, image_path: str | None, embedding: Iterable[float]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chunks(document_id, source, modality, content, page, image_path, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, source, modality, content, page, image_path, json.dumps(list(embedding))),
            )
            conn.commit()

    def job(self, job_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def document_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchHit]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks").fetchall()
        stored = []
        for row in rows:
            vector = json.loads(row["embedding"]) if row["embedding"] else []
            score = self._cosine_similarity(query_embedding, vector)
            stored.append((score, row))
        stored.sort(key=lambda item: item[0], reverse=True)
        hits: list[SearchHit] = []
        for score, row in stored[:top_k]:
            hits.append(
                SearchHit(
                    document_id=row["document_id"],
                    source=row["source"],
                    modality=row["modality"],
                    content=row["content"],
                    page=row["page"],
                    score=float(score),
                    image_path=row["image_path"],
                )
            )
        return hits

    @staticmethod
    def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
        a_list = list(a)
        b_list = list(b)
        if not a_list or not b_list or len(a_list) != len(b_list):
            return 0.0
        dot = sum(x * y for x, y in zip(a_list, b_list))
        norm_a = sum(x * x for x in a_list) ** 0.5
        norm_b = sum(x * x for x in b_list) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


__all__ = ["RAGStore", "chunk_text", "SearchHit"]
