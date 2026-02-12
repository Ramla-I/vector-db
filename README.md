# Vector DB

A CLI tool for semantic search over PDF and Markdown documents using ChromaDB. Supports both local (FastEmbed) and OpenAI embeddings - can run fully offline and free.

## Features

- **Multi-format support**: Ingest PDF, Markdown, and plain text files
- **Smart chunking**: Section-aware chunking with bidirectional overlap
- **Multiple databases**: Create isolated databases for different document collections
- **Semantic search**: Find relevant content using natural language queries
- **Hybrid search**: Combine semantic similarity with keyword matching
- **Multiple reranking options**: Cohere API, FlashRank (local), or BGE (local with torch)
- **Metadata filtering**: Filter results by source, page, or custom attributes
- **Technical document optimization**: Special preprocessing for register definitions, tables, and technical specs
- **Fully offline mode**: Local embeddings (FastEmbed) + local reranking (FlashRank) - no API calls needed

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │  PDF/MD  │───▶│  Text Processor   │───▶│   Chunker    │───▶│ Embedder │ │
│  │   File   │    │  - Clean text     │    │  - Sections  │    │(local or │ │
│  └──────────┘    │  - Extract tables │    │  - Overlap   │    │ OpenAI)│ │
│                  │  - Add key terms  │    │  - Metadata  │    └────┬─────┘ │
│                  └───────────────────┘    └──────────────┘         │       │
│                                                                     ▼       │
│                                                              ┌──────────┐   │
│                                                              │ ChromaDB │   │
│                                                              │ (Vector  │   │
│                                                              │  Store)  │   │
│                                                              └──────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               SEARCH PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐  │
│  │  Query  │───▶│ Embedder │───▶│ ChromaDB │───▶│ Reranker │───▶│Keyword│  │
│  │         │    │(local or │    │  Search  │    │(optional)│    │ Boost │  │
│  │         │    │ OpenAI)  │    │          │    │          │    │       │  │
│  └─────────┘    └──────────┘    └────┬─────┘    └────┬─────┘    └───┬───┘  │
│                                      │               │              │      │
│                                      ▼               ▼              ▼      │
│                               ┌─────────────────────────────────────────┐  │
│                               │            Top-K Results               │  │
│                               │  - Score, Source, Page/Section         │  │
│                               │  - Matching text snippet               │  │
│                               └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
cd vector_db

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to choose your embedding provider (local or openai)
```

### Embedding Providers

**Local (default, recommended)** - Free, offline, no API key needed:
```bash
# .env
EMBEDDING_PROVIDER=local
```
Uses FastEmbed with `BAAI/bge-small-en-v1.5` (ONNX, ~67MB model downloaded on first use).

**OpenAI** - Requires API key:
```bash
# .env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-key-here
```

> **Note:** Databases are tied to their embedding provider. A database created with local
> embeddings (384 dimensions) is not compatible with OpenAI embeddings (1536 dimensions).
> Create separate databases if you want to compare providers.

## Quick Start

```bash
# 1. Create a database
python main.py create-db mydb

# 2. Ingest a document
python main.py ingest mydb document.pdf

# 3. Search
python main.py search mydb "your search query"

# 4. Search with better accuracy (recommended)
python main.py search mydb "your search query" --keyword-boost
```

## CLI Commands

### Database Management

```bash
# Create a new database
python main.py create-db <db_name>

# List all databases
python main.py list-dbs

# Delete a database
python main.py delete-db <db_name>
```

### Document Management

```bash
# Ingest a file (PDF, Markdown, or text)
python main.py ingest <db_name> <file_path>

# Ingest with custom metadata
python main.py ingest <db_name> <file_path> --meta category=manual --meta version=2.0

# List documents in a database
python main.py list-docs <db_name>

# Delete a document
python main.py delete-doc <db_name> <doc_name>
```

### Search

```bash
# Basic search
python main.py search <db_name> "your query"

# Search with options
python main.py search <db_name> "your query" [OPTIONS]
```

#### Search Options

| Option | Description | Speed | Accuracy |
|--------|-------------|-------|----------|
| (none) | Raw semantic search | Fast | Good |
| `--keyword-boost` | Hybrid semantic + keyword matching | **Fast** | **Best** |
| `--rerank-local` | FlashRank MiniLM reranking | Medium | Good |
| `--rerank-local --keyword-boost` | FlashRank + keyword boost | Medium | **Best** |
| `--rerank-bge` | BGE reranker (requires torch) | Slow | Good |
| `--rerank-bge --keyword-boost` | BGE + keyword boost | Slow | **Best** |
| `--rerank` | Cohere API reranking | Medium | **Best** |

#### Other Search Options

```bash
--top-k N           # Number of results (default: 5)
--filter key=value  # Filter by metadata (can be repeated)
```

#### Examples

```bash
# Recommended: Fast and accurate
python main.py search mydb "AFIO_MAPR2 register definition" --keyword-boost

# Filter by source file
python main.py search mydb "interrupt handling" --filter source=rm0041.pdf

# Filter by page
python main.py search mydb "GPIO configuration" --filter page=45

# Get more results
python main.py search mydb "DMA transfer" --top-k 10 --keyword-boost
```

## Search Strategy Recommendations

### For Technical Documentation (Datasheets, Reference Manuals)

```bash
# Best accuracy for register lookups, API references, etc.
python main.py search mydb "REGISTER_NAME definition" --keyword-boost
```

The `--keyword-boost` option uses word-boundary matching to distinguish similar terms (e.g., `AFIO_MAPR` vs `AFIO_MAPR2`).

### For General Documents

```bash
# Semantic search is usually sufficient
python main.py search mydb "how does authentication work"
```

### When You Need Maximum Accuracy

```bash
# Cohere reranker (requires API key)
python main.py search mydb "complex technical query" --rerank

# Or local reranking + keyword boost (free)
python main.py search mydb "complex technical query" --rerank-local --keyword-boost
```

## Configuration

Edit `.env` to configure:

```bash
# Embedding provider: "local" (free, offline) or "openai" (API)
EMBEDDING_PROVIDER=local

# OpenAI API key (only needed if EMBEDDING_PROVIDER=openai)
OPENAI_API_KEY=your-api-key-here

# Local embedding model (only used if EMBEDDING_PROVIDER=local)
# Options: BAAI/bge-small-en-v1.5 (default, 384d), BAAI/bge-base-en-v1.5 (768d)
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Optional: Cohere API key for --rerank option
COHERE_API_KEY=your-cohere-key

# Chunking settings
CHUNK_SIZE=500      # Target tokens per chunk
CHUNK_OVERLAP=50    # Overlap tokens between chunks

# Search settings
TOP_K_RESULTS=5     # Default number of results
```

## Project Structure

```
vector_db/
├── main.py              # CLI entry point
├── config.py            # Configuration management
├── embeddings.py        # Embedding providers (FastEmbed local + OpenAI)
├── pdf_processor.py     # PDF parsing and chunking
├── text_processor.py    # Markdown/text processing with preprocessing
├── vector_store.py      # ChromaDB operations + keyword boost
├── reranker.py          # Cohere, FlashRank, and BGE rerankers
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment configuration
└── databases/           # Database storage (created automatically)
    └── <db_name>/       # Each database is isolated
```

## Ingestion Pipeline

The ingestion pipeline converts raw documents into semantically enriched, fixed-size text chunks stored as vectors.

```
Raw document (.pdf / .md / .txt)
    │
    ▼
1. Text Cleaning          Strip page headers/footers, normalize whitespace
    │
    ▼
2. Section Splitting      Split on markdown headers (# ## ### ####)
    │
    ▼
3. TOC Filtering          Discard table-of-contents entries (< 50 chars of real content)
    │
    ▼
4. Chunking               Recursive splitting at 500-token limit
    │                      (split at: paragraphs → lines → sentences → words)
    ▼
5. Key Term Extraction    Classify and annotate each chunk:
    │                      • Table + offset → "REGISTER DEFINITION: AFIO_MAPR2"
    │                      • 4+ register names → "OVERVIEW:register_list"
    │                      • Otherwise → "[KEY: AFIO_MAPR | offset:0x04]"
    ▼
6. Bidirectional Overlap  Add ±25 tokens from neighboring chunks
    │
    ▼
7. Embedding              FastEmbed (384-dim, local) or OpenAI (1536-dim, API)
    │
    ▼
8. Storage                ChromaDB with cosine similarity index
```

### Text Cleaning
Removes document-specific artifacts such as page headers (`"612/709 RM0041 Rev 6"`), collapses multiple blank lines into a single paragraph break, and strips trailing whitespace. This ensures consistent delimiter behavior during chunking.

### Section Splitting
The markdown text is split on heading delimiters (levels 1-4). Each section retains its heading text. When a section must be split into multiple chunks, the heading is prepended to every sub-chunk so that context is preserved.

### TOC Filtering
Table-of-contents entries like `"GPIO . . . . . . 45"` are detected by stripping page numbers and dot-leader patterns. Sections with fewer than 50 characters of remaining content are discarded.

### Chunking
A recursive strategy splits text at the most natural boundary available, trying paragraph breaks first (`\n\n`), then line breaks (`\n`), then sentence boundaries (`. `), and finally word boundaries (` `) as a last resort. Each chunk targets 500 tokens (measured by the cl100k_base tokenizer).

### Key Term Extraction
Each chunk is classified based on structural cues and annotated before embedding:

- **Register definitions** — chunks containing both a markdown table and an `Address offset:` field receive a prominent title (e.g., `REGISTER DEFINITION: AFIO_MAPR2 - Complete bit field specification`) and their register names, offsets, reset values, and bit field names are extracted into a `[KEY: ...]` prefix.
- **Overview sections** — chunks mentioning 4+ distinct register names without containing a definition are tagged as `OVERVIEW:register_list`, and individual register names are excluded from the key terms to prevent these sections from dominating retrieval for specific register queries.
- **Regular chunks** — register names found in the text are added to the key terms.

Example annotated chunk:
```
REGISTER DEFINITION: AFIO_MAPR2 - Complete bit field specification
[KEY: TABLE:register_bitfields | AFIO_MAPR2 | offset:0x1C | reset:0x0000 | fields:MISC_REMAP,TIM12_REMAP]

# 7.4.7 AF remap and debug I/O configuration register (AFIO_MAPR2)
Address offset: 0x1C
Reset value: 0x0000 0000
...
```

### Bidirectional Overlap
Each chunk receives the final 25 tokens of its predecessor and the first 25 tokens of its successor, demarcated with `[...]` markers. This prevents information loss at chunk boundaries.

### PDF vs Markdown
Both formats use the same recursive chunking and overlap strategy. The key differences:
- **PDF processor** splits by pages and stores page numbers as metadata. It has no preprocessing or key term extraction, and table structure is lost during text extraction.
- **Text processor** splits by markdown headers and stores section names as metadata. It includes full preprocessing (table detection, key terms, overview detection) and preserves table structure.

Markdown is strongly preferred for technical documentation.

## Search Pipeline

The search pipeline retrieves and refines relevant chunks through semantic search followed by optional post-retrieval refinement.

```
Query: "AFIO_MAPR2 register definition" --keyword-boost --top-k 1
    │
    ▼
1. Expand candidate pool     top_k × 5 = 5 candidates
    │
    ▼
2. Embed query                FastEmbed: ~6ms / OpenAI: ~1300ms
    │
    ▼
3. ChromaDB cosine search     ~5ms, returns 5 nearest vectors
    │
    ▼
4. Reranking (optional)       Cross-encoder re-scores all candidates
    │
    ▼
5. Keyword boost (optional)   Word-boundary matching boosts exact terms
    │
    ▼
6. Truncate to top_k          Return top 1 result
```

### Candidate Expansion
When reranking or keyword boosting is enabled, the initial retrieval fetches 5x the requested result count to provide a large enough candidate pool for downstream refinement.

### Reranking
Three cross-encoder reranking backends are supported:

| Backend | Model | Size | Speed | Requires |
|---------|-------|------|-------|----------|
| `--rerank` | Cohere rerank-v3.5 | Cloud | ~130ms | API key |
| `--rerank-local` | MiniLM-L-12-v2 | ~22MB | ~200ms | Nothing |
| `--rerank-bge` | bge-reranker-v2-m3 | ~568MB | ~15s | PyTorch |

Cross-encoders process the query and each candidate as a pair, enabling deeper semantic comparison than the initial bi-encoder retrieval. However, they often produce near-identical scores for candidates containing similar terminology.

### Keyword Boost
A hybrid search stage that combines semantic scores with exact lexical matching:

1. **Term extraction** — technical identifiers (e.g., `AFIO_MAPR2`) are extracted from the query using regex
2. **Word-boundary matching** — each candidate is scanned using `\bAFIO_MAPR2\b`, which matches `AFIO_MAPR2` but not `AFIO_MAPR`
3. **Tiered boosting** — additive score boost based on match context:
   - `+0.20` for match in a `REGISTER DEFINITION:` title
   - `+0.10` for match in `[KEY: ...]` terms
   - `+0.05` for match elsewhere in the text
4. **Re-sort** — candidates are re-sorted by boosted scores (not capped at 1.0)

When both reranking and keyword boost are enabled, keyword boost is applied **after** reranking. This ordering is important because cross-encoders often produce tied scores among top candidates — the keyword boost breaks these ties in favor of chunks containing the exact query term.

### Why Keyword Boost Works
Dense embedding models encode meaning at a sub-word level and cannot reliably distinguish identifiers that differ by a single character (e.g., `AFIO_MAPR` vs `AFIO_MAPR2`). Word-boundary regex matching provides the lexical precision that semantic search lacks.

## Performance

Tested on STM32 reference manual (~700 pages, 1867 chunks):

### Search Latency

| Embedding Provider | Search Method | Avg Time |
|-------------------|---------------|----------|
| **Local (FastEmbed)** | + keyword-boost | **~800ms** |
| OpenAI API | + keyword-boost | ~2000ms |
| OpenAI File Search | (built-in) | ~1600ms |

### Operation Times

| Operation | Local Embeddings | OpenAI Embeddings |
|-----------|-----------------|-------------------|
| Ingestion (1867 chunks) | ~30s | ~60s |
| Search (basic) | ~750ms | ~1.3s |
| Search + keyword-boost | ~800ms | ~2.0s |
| Search + rerank-local | ~1.5s | ~2.7s |
| Search + rerank (Cohere) | ~1.6s | ~2.8s |

### Accuracy Benchmarks

Tested with AFIO register queries (EVCR, MAPR, MAPR2):

| Method | Local Embeddings | OpenAI Embeddings | OpenAI File Search |
|--------|-----------------|-------------------|--------------------|
| Raw search | 2/3 | 2/3 | 0/3 |
| `--keyword-boost` | **3/3** | **3/3** | N/A |
| `--rerank-local --keyword-boost` | **3/3** | **3/3** | N/A |
| `--rerank` (Cohere) | **3/3** | **3/3** | N/A |

### Cost Per Query

| Method | Cost |
|--------|------|
| **Local embeddings** | **Free** |
| OpenAI embeddings | ~$0.00002 |
| OpenAI File Search | ~$0.0025 |

## Requirements

- Python 3.11+
- No API keys needed for local mode (FastEmbed + FlashRank)
- Optional: OpenAI API key (for OpenAI embeddings)
- Optional: Cohere API key (for `--rerank`)
- Optional: PyTorch + Python 3.12 (for `--rerank-bge`)

## License

MIT
