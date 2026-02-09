"""Text/Markdown processing with preprocessing and chunking."""

import re
from pathlib import Path
from typing import List, Dict, Any

import tiktoken

import config


class TextProcessor:
    """Process text/markdown files: clean, chunk, and add metadata."""

    # Patterns for STM reference manual headers/footers
    HEADER_PATTERNS = [
        r'^\d+/\d+\s+RM\d+\s+Rev\s+\d+\s*$',  # "612/709 RM0041 Rev 6"
        r'^RM\d+\s+Rev\s+\d+\s+\d+/\d+\s*$',  # "RM0041 Rev 6 612/709"
        r'^RM\d+\s+[A-Za-z].*$',               # "RM0041 Universal synchronous..."
    ]

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self._header_regexes = [re.compile(p, re.MULTILINE) for p in self.HEADER_PATTERNS]

    def clean_text(self, text: str) -> str:
        """Clean text by removing headers/footers and normalizing whitespace."""
        # Remove page headers/footers
        for regex in self._header_regexes:
            text = regex.sub('', text)

        # Normalize multiple blank lines to double newline
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace from lines
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def split_by_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split markdown by headers, preserving section context."""
        # Split on markdown headers (# ## ### etc)
        section_pattern = r'^(#{1,4})\s+(.+)$'
        lines = text.split('\n')

        sections = []
        current_section = {
            'header': '',
            'level': 0,
            'content': []
        }

        for line in lines:
            match = re.match(section_pattern, line)
            if match:
                # Save previous section if it has content
                if current_section['content']:
                    sections.append(current_section)

                level = len(match.group(1))
                header = match.group(2).strip()
                current_section = {
                    'header': header,
                    'level': level,
                    'content': []
                }
            else:
                current_section['content'].append(line)

        # Don't forget the last section
        if current_section['content']:
            sections.append(current_section)

        return sections

    def chunk_section(self, section: Dict[str, Any]) -> List[str]:
        """Chunk a section's content while preserving the header context."""
        header = section['header']
        content = '\n'.join(section['content']).strip()

        if not content:
            return []

        # Skip TOC entries (header + just a page number or very short content)
        # Also skip entries with dot leaders like "Section name . . . . . . 123"
        content_without_numbers = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE).strip()
        content_without_toc = re.sub(r'\.[\s.]+\d+\s*$', '', content, flags=re.MULTILINE).strip()
        if len(content_without_numbers) < 50 or len(content_without_toc) < 50:
            return []

        # If section fits in one chunk, return it with header and key terms
        full_text = f"# {header}\n\n{content}" if header else content
        key_terms = self.extract_key_terms(full_text, header)
        full_text_with_keys = key_terms + full_text

        if self.count_tokens(full_text_with_keys) <= self.chunk_size:
            return [full_text_with_keys] if full_text_with_keys.strip() else []

        # Otherwise, split content and prepend header to each chunk
        chunks = self._split_text_recursive(content)

        # Add header context and key terms to each chunk
        result = []
        for chunk in chunks:
            if header:
                chunk_with_header = f"# {header}\n\n{chunk}"
            else:
                chunk_with_header = chunk

            # Add key terms for this specific chunk
            chunk_key_terms = self.extract_key_terms(chunk_with_header, header)
            final_chunk = chunk_key_terms + chunk_with_header

            if final_chunk.strip():
                result.append(final_chunk)

        return result

    def _split_text_recursive(
        self, text: str, separators: List[str] = None
    ) -> List[str]:
        """Recursively split text by separators."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]

        if not separators or self.count_tokens(text) <= self.chunk_size:
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
            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if self.count_tokens(split) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(
                        split, remaining_separators
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def has_markdown_table(self, text: str) -> bool:
        """Check if text contains a markdown table."""
        # Look for table separator row pattern: |---|---| or | --- | --- |
        table_sep_pattern = r'^\|[\s\-:]+\|[\s\-:|]+$'
        return bool(re.search(table_sep_pattern, text, re.MULTILINE))

    def extract_key_terms(self, text: str, header: str) -> str:
        """Extract key terms from chunk to improve reranking."""
        terms = []
        register_title = ""

        # Check for table - indicates register bit field definitions
        has_table = self.has_markdown_table(text)
        if has_table:
            terms.append("TABLE:register_bitfields")

        # Extract register names (e.g., AFIO_MAPR, GPIOx_CRL, USART_BRR)
        register_pattern = r'\b([A-Z]{2,}[x]?_[A-Z0-9_]+)\b'
        registers = re.findall(register_pattern, text)
        if registers:
            # Dedupe and take first few
            unique_regs = list(dict.fromkeys(registers))[:5]

            # If this looks like a register definition (has table + offset), make register name prominent
            if has_table and 'Address offset:' in text:
                register_title = f"REGISTER DEFINITION: {unique_regs[0]} - Complete bit field specification\n"
                terms.extend(unique_regs)
            elif len(unique_regs) >= 4:
                # Many registers mentioned without definition - mark as overview (don't add register names)
                terms.append("OVERVIEW:register_list")
            else:
                terms.extend(unique_regs)

        # Extract address offset
        offset_match = re.search(r'Address offset:\s*(0x[0-9A-Fa-f]+)', text)
        if offset_match:
            terms.append(f"offset:{offset_match.group(1)}")

        # Extract reset value
        reset_match = re.search(r'Reset value:\s*(0x[0-9A-Fa-f]+)', text)
        if reset_match:
            terms.append(f"reset:{reset_match.group(1)}")

        # Extract bit field names (e.g., "Bit 7 EVOE:", "Bits 3:0 PIN[3:0]")
        field_pattern = r'Bits?\s+\d+(?::\d+)?\s+([A-Z][A-Z0-9_\[\]]+):'
        fields = re.findall(field_pattern, text)
        if fields:
            unique_fields = list(dict.fromkeys(fields))[:8]
            terms.append(f"fields:{','.join(unique_fields)}")

        if terms:
            return f"{register_title}[KEY: {' | '.join(terms)}]\n\n"
        return ""

    def add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap from both preceding and succeeding chunks for context continuity."""
        if len(chunks) <= 1:
            return chunks

        overlapped = []
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

            overlapped.append("\n\n".join(parts))

        return overlapped

    def process_file(
        self, file_path: Path, extra_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Process a text/markdown file and return chunks with metadata."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read and clean text
        text = file_path.read_text(encoding='utf-8')
        text = self.clean_text(text)

        # Split by sections and chunk each section
        sections = self.split_by_sections(text)

        all_chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self.chunk_section(section)
            section_chunks = self.add_overlap(section_chunks)

            for chunk_text in section_chunks:
                metadata = {
                    "source": file_path.name,
                    "section": section['header'][:100] if section['header'] else "",
                    "chunk_index": chunk_index,
                }
                if extra_metadata:
                    metadata.update(extra_metadata)

                all_chunks.append({"text": chunk_text, "metadata": metadata})
                chunk_index += 1

        return all_chunks
