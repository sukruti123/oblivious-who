import os
from typing import cast

import pymupdf


def extract_pdf_text(pdf_path: str):
    """Extracts and prints page-by-page text from a target PDF file."""
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    print(f"📄 Testing text extraction for: {pdf_path}\n" + "=" * 50)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f"Total Pages: {total_pages}\n")

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = cast(str, page.get_text("text")).strip()
        print(f"--- [Page {page_num + 1}] ---")
        if text:
            print(text[:300] + ("..." if len(text) > 300 else ""))
        else:
            print("⚠️ [Warning: No text found. Might be a scanned page/image.]")
        print("-" * 50)


def find_knowledge_pdfs(knowledge_root: str = "knowledge") -> list[str]:
    """Find every PDF in the knowledge folders, including nested folders."""
    pdf_paths = []
    for root, _, files in os.walk(knowledge_root):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file_name))
    return sorted(pdf_paths)


if __name__ == "__main__":
    knowledge_pdfs = find_knowledge_pdfs()

    if not knowledge_pdfs:
        print("No PDF files found under the knowledge directory.")
    else:
        print(f"Found {len(knowledge_pdfs)} PDF file(s) to test.\n")
        for pdf_path in knowledge_pdfs:
            category = os.path.basename(os.path.dirname(pdf_path))
            print(f"### Category: {category} ###")
            extract_pdf_text(pdf_path)