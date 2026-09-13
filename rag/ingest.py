import os
import re
from hashlib import md5
from typing import cast

import pymupdf
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
try:
    from .multimodal_pipeline import process_multimodal_document
except ImportError:
    from multimodal_pipeline import process_multimodal_document

# ------------------------------------------------------------------
# Configuration & Database Setup
# ------------------------------------------------------------------
DB_URI = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:your_secure_password_here@localhost:5432/sovereign_x",
)
KNOWLEDGE_DIR = "knowledge"

# Initialize local embedding model (384 dimensions)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = 384


def get_db_connection():
    """Connects to local PostgreSQL database and registers pgvector."""
    conn = psycopg.connect(DB_URI, autocommit=True)
    register_vector(conn)
    return conn


def initialize_schema(conn):
    """Create the pgvector storage schema if it does not exist."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                page INTEGER NOT NULL,
                section TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                document_version TEXT NOT NULL,
                access_permission TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,
                UNIQUE (filename, page, section, chunk_hash)
            );
            ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_hash TEXT;
            UPDATE document_chunks
            SET chunk_hash = md5(chunk_text)
            WHERE chunk_hash IS NULL;
            ALTER TABLE document_chunks ALTER COLUMN chunk_hash SET NOT NULL;
            ALTER TABLE document_chunks
            DROP CONSTRAINT IF EXISTS document_chunks_filename_page_section_chunk_text_key;
            CREATE UNIQUE INDEX IF NOT EXISTS document_chunks_identity_idx
            ON document_chunks (filename, page, section, chunk_hash);
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
            """
        )


def clean_text(text: str) -> str:
    """Normalize extracted PDF/OCR text before chunking."""
    return re.sub(r"\s+", " ", text).strip()


def store_chunk(conn, filename: str, page: int, section: str, chunk: str) -> None:
    """Embed and upsert one chunk into PostgreSQL/pgvector."""
    embedding = embedding_model.encode(chunk, normalize_embeddings=True).tolist()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document_chunks
                 (filename, page, section, chunk_text, chunk_hash,
                  document_version, access_permission, embedding)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
              ON CONFLICT (filename, page, section, chunk_hash) DO UPDATE
            SET embedding = EXCLUDED.embedding
            """,
            (
                filename,
                page,
                section,
                chunk,
                md5(chunk.encode("utf-8")).hexdigest(),
                "1.0",
                "Engineer",
                embedding,
            ),
        )


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Splits document text into overlapping paragraph chunks."""
    paragraphs = [p.strip() for p in clean_text(text).split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            words = para.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])
                chunks.append(chunk)
    return chunks


def ingest_documents():
    """Walks knowledge base, extracts text/OCR, chunks, embeds, and stores in pgvector."""
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"❌ Directory '{KNOWLEDGE_DIR}' does not exist.")
        return

    conn = get_db_connection()
    initialize_schema(conn)
    total_chunks = 0

    print("🚀 Starting Document Ingestion Pipeline into PostgreSQL/pgvector...\n")

    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            category = os.path.basename(root)

            print(f"📂 Processing [{category}]: {file}")

            # ------------------------------------------------------
            # 1. Parse PDF Files
            # ------------------------------------------------------
            if file.lower().endswith(".pdf"):
                doc = pymupdf.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = clean_text(cast(str, page.get_text("text")))

                    # Fallback to 2-tier OCR if page is a scanned image
                    if not page_text:
                        print(f"  ⚠️ Page {page_num + 1} has no text layer. Running OCR...")
                        pix = page.get_pixmap()
                        tmp_img = f"tmp_page_{page_num+1}.png"
                        pix.save(tmp_img)
                        ocr_res = process_multimodal_document(tmp_img)
                        page_text = clean_text(ocr_res.get("extracted_text", ""))
                        if os.path.exists(tmp_img):
                            os.remove(tmp_img)

                    if not page_text:
                        continue

                    chunks = chunk_text(page_text)

                    for idx, chunk in enumerate(chunks):
                        store_chunk(conn, file, page_num + 1, f"Section {idx + 1}", chunk)
                        total_chunks += 1

            # ------------------------------------------------------
            # 2. Parse Standalone Images / Scans
            # ------------------------------------------------------
            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                ocr_res = process_multimodal_document(file_path)
                extracted_text = clean_text(ocr_res.get("extracted_text", ""))

                if extracted_text:
                    chunks = chunk_text(extracted_text)
                    for idx, chunk in enumerate(chunks):
                        store_chunk(conn, file, 1, f"OCR Extraction {idx + 1}", chunk)
                        total_chunks += 1

    conn.close()
    print(f"\n✅ Ingestion Complete! Total chunks stored in pgvector: {total_chunks}")


if __name__ == "__main__":
    ingest_documents()