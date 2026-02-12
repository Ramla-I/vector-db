"""PDF parsing and chunking."""

from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF

from chunking import split_text_recursive, add_overlap


class PDFProcessor:
    """Process PDFs: extract text and create chunks with metadata.

    Uses page-level chunking (no section awareness or preprocessing).
    For markdown files with section headers, use TextProcessor instead.
    """

    def extract_text_with_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with page numbers."""
        pages = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append({"page_num": page_num + 1, "text": text})

        doc.close()
        return pages

    def process_pdf(
        self, pdf_path: Path, extra_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Process a PDF and return chunks with metadata."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = self.extract_text_with_pages(pdf_path)
        all_chunks = []
        chunk_index = 0

        for page_data in pages:
            page_num = page_data["page_num"]
            text = page_data["text"]

            page_chunks = split_text_recursive(text)
            page_chunks = add_overlap(page_chunks)

            for chunk_text in page_chunks:
                metadata = {
                    "source": pdf_path.name,
                    "page": page_num,
                    "chunk_index": chunk_index,
                }
                if extra_metadata:
                    metadata.update(extra_metadata)

                all_chunks.append({"text": chunk_text, "metadata": metadata})
                chunk_index += 1

        return all_chunks
