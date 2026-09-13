"""PostgreSQL/pgvector storage for industrial multimodal RAG chunks."""

import os
from collections.abc import Sequence

import psycopg
from pgvector.psycopg import register_vector


DB_CONNECTION_STRING = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:your_secure_password_here@localhost:5432/sovereign_x",
)
EMBEDDING_DIMENSIONS = 1536


def create_knowledge_chunks_table() -> None:
    """Create the knowledge_chunks table when it does not already exist."""
    with psycopg.connect(DB_CONNECTION_STRING) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id SERIAL PRIMARY KEY,
                    content_type VARCHAR(50) NOT NULL
                        CHECK (content_type IN ('text', 'image_summary')),
                    raw_content TEXT NOT NULL,
                    embedding VECTOR(1536) NOT NULL
                )
                """
            )


def insert_chunk(
    content_type: str,
    raw_content: str,
    embedding_list: Sequence[float],
) -> int:
    """Insert a text or image-summary chunk and return its generated id."""
    if content_type not in {"text", "image_summary"}:
        raise ValueError("content_type must be 'text' or 'image_summary'")
    if len(embedding_list) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"embedding_list must contain {EMBEDDING_DIMENSIONS} values"
        )

    create_knowledge_chunks_table()
    with psycopg.connect(DB_CONNECTION_STRING) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO knowledge_chunks
                    (content_type, raw_content, embedding)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (content_type, raw_content, list(embedding_list)),
            )
            inserted_id = cursor.fetchone()[0]

    return inserted_id


def similarity_search(
    query_embedding: Sequence[float], limit: int = 5
) -> list[dict[str, object]]:
    """Return the closest chunks using pgvector cosine distance."""
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"query_embedding must contain {EMBEDDING_DIMENSIONS} values"
        )
    if limit < 1:
        raise ValueError("limit must be at least 1")

    create_knowledge_chunks_table()
    with psycopg.connect(DB_CONNECTION_STRING) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    content_type,
                    raw_content,
                    embedding <=> %s AS cosine_distance
                FROM knowledge_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (list(query_embedding), list(query_embedding), limit),
            )
            columns = [column.name for column in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


if __name__ == "__main__":
    create_knowledge_chunks_table()
    print("knowledge_chunks table is ready.")
