import os
from typing import Any, cast

import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
from multimodal_pipeline import process_multimodal_document

# Persistence directory for local offline vector database
CHROMA_DATA_PATH = "./local_rag_db"

# Initialize local embedding function (no cloud APIs)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Connect to local persistent ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_or_create_collection(
    name="industrial_knowledge", embedding_function=cast(Any, embedding_func)
)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Splits document text into overlapping paragraph chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # Sub-split long paragraphs
            words = para.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])
                chunks.append(chunk)
    return chunks


def ingest_knowledge_base(knowledge_root: str = "knowledge"):
    """Walks knowledge folder, parses files, and adds vectors with metadata."""
    if not os.path.exists(knowledge_root):
        print(f"❌ Knowledge directory '{knowledge_root}' does not exist.")
        return

    doc_counter = 0

    for root, _, files in os.walk(knowledge_root):
        for file in files:
            file_path = os.path.join(root, file)
            category = os.path.basename(root)

            print(f"\n📂 Processing [{category}]: {file}")

            # 1. Handle PDF Documents
            if file.endswith(".pdf"):
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = cast(str, page.get_text("text")).strip()

                    # Fallback to OCR if page has no selectable text
                    if not page_text:
                        print(
                            f"  ⚠️ Page {page_num + 1} has no text. Running OCR..."
                        )
                        # Temporary image render for OCR
                        pix = page.get_pixmap()
                        tmp_img_path = f"tmp_page_{page_num}.png"
                        pix.save(tmp_img_path)
                        ocr_res = process_multimodal_document(tmp_img_path)
                        page_text = str(ocr_res.get("extracted_text", ""))
                        if os.path.exists(tmp_img_path):
                            os.remove(tmp_img_path)

                    if not page_text:
                        continue

                    chunks = chunk_text(page_text)

                    for idx, chunk in enumerate(chunks):
                        doc_id = f"{file}_p{page_num+1}_c{idx+1}"
                        collection.add(
                            documents=[chunk],
                            metadatas=[
                                {
                                    "filename": file,
                                    "page": page_num + 1,
                                    "section": f"Section {idx + 1}",
                                    "document_version": "1.0",
                                    "access_permission": "Engineer",
                                }
                            ],
                            ids=[doc_id],
                        )
                        doc_counter += 1

            # 2. Handle Standalone Scanned Images
            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                ocr_res = process_multimodal_document(file_path)
                extracted_text = ocr_res["extracted_text"]

                if extracted_text:
                    chunks = chunk_text(extracted_text)
                    for idx, chunk in enumerate(chunks):
                        doc_id = f"{file}_img_c{idx+1}"
                        collection.add(
                            documents=[chunk],
                            metadatas=[
                                {
                                    "filename": file,
                                    "page": 1,
                                    "section": f"OCR Extraction {idx + 1}",
                                    "document_version": "1.0",
                                    "access_permission": "Engineer",
                                }
                            ],
                            ids=[doc_id],
                        )
                        doc_counter += 1

    print(
        f"\n✅ Ingestion complete! Successfully stored {doc_counter} text chunks in ChromaDB."
    )


if __name__ == "__main__":
    ingest_knowledge_base()