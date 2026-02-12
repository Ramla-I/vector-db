"""Shared chunking utilities used by both PDF and text processors."""

from typing import List

import tiktoken

import config


# Shared tokenizer instance (cl100k_base is used by OpenAI models)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_tokenizer.encode(text))


def split_text_recursive(
    text: str, chunk_size: int = None, separators: List[str] = None
) -> List[str]:
    """Recursively split text by separators (paragraph -> line -> sentence -> word).

    Tries each separator level in order, accumulating text into chunks
    that fit within chunk_size tokens. Oversized segments recurse with
    finer-grained separators.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE

    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if not separators or count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_separators = separators[1:]

    splits = text.split(separator)
    chunks = []
    current_chunk = ""

    for split in splits:
        test_chunk = (
            current_chunk + separator + split if current_chunk else split
        )
        if count_tokens(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if count_tokens(split) > chunk_size:
                sub_chunks = split_text_recursive(
                    split, chunk_size, remaining_separators
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = split

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def add_overlap(chunks: List[str], chunk_overlap: int = None) -> List[str]:
    """Add bidirectional overlap between adjacent chunks for context continuity.

    Each chunk gets ±half_overlap tokens from its neighbors, marked with [...].
    """
    if len(chunks) <= 1:
        return chunks

    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    overlapped = []
    half_overlap = chunk_overlap // 2

    for i, chunk in enumerate(chunks):
        parts = []

        # Prepend tail of preceding chunk
        if i > 0:
            prev_tokens = _tokenizer.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-half_overlap:]
            overlap_text = _tokenizer.decode(overlap_tokens).strip()
            if overlap_text:
                parts.append(f"[...] {overlap_text}")

        parts.append(chunk)

        # Append head of succeeding chunk
        if i < len(chunks) - 1:
            next_tokens = _tokenizer.encode(chunks[i + 1])
            overlap_tokens = next_tokens[:half_overlap]
            overlap_text = _tokenizer.decode(overlap_tokens).strip()
            if overlap_text:
                parts.append(f"{overlap_text} [...]")

        overlapped.append("\n\n".join(parts))

    return overlapped
