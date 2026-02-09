"""PDF parsing and chunking."""

from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import tiktoken

import config


class PDFProcessor:
    """Process PDFs: extract text and create chunks with metadata."""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

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

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def split_text_recursive(
        self, text: str, separators: List[str] = None
    ) -> List[str]:
        """Recursively split text by separators (paragraph, sentence, word)."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]

        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            test_chunk = (
                current_chunk + separator + split if current_chunk else split
            )
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if self.count_tokens(split) > self.chunk_size:
                    sub_chunks = self.split_text_recursive(
                        split, remaining_separators
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def create_chunks_with_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap from both preceding and succeeding chunks for context continuity."""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []
        half_overlap = self.chunk_overlap // 2

        for i, chunk in enumerate(chunks):
            parts = []

            # Add overlap from preceding chunk
            if i > 0:
                prev_tokens = self.tokenizer.encode(chunks[i - 1])
                overlap_tokens = prev_tokens[-half_overlap:]
                overlap_text = self.tokenizer.decode(overlap_tokens).strip()
                if overlap_text:
                    parts.append(f"[...] {overlap_text}")

            # Add main chunk content
            parts.append(chunk)

            # Add overlap from succeeding chunk
            if i < len(chunks) - 1:
                next_tokens = self.tokenizer.encode(chunks[i + 1])
                overlap_tokens = next_tokens[:half_overlap]
                overlap_text = self.tokenizer.decode(overlap_tokens).strip()
                if overlap_text:
                    parts.append(f"{overlap_text} [...]")

            overlapped_chunks.append("\n\n".join(parts))

        return overlapped_chunks

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

            page_chunks = self.split_text_recursive(text)
            page_chunks = self.create_chunks_with_overlap(page_chunks)

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
