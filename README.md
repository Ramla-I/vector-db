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

## Preprocessing Features

The text processor includes several optimizations for technical documents:

### Table Detection
Chunks containing markdown tables are tagged with `TABLE:register_bitfields` to help identify register definitions.

### Register Definition Markers
Chunks with both a table and an address offset get a prominent title:
```
REGISTER DEFINITION: AFIO_MAPR2 - Complete bit field specification
```

### Key Term Extraction
Important terms are extracted and prepended to chunks:
```
[KEY: TABLE:register_bitfields | AFIO_MAPR2 | offset:0x1C | reset:0x0000 | fields:MISC_REMAP,TIM12_REMAP]
```

### Overview Detection
Sections mentioning many registers without definitions are marked as overviews to prevent them from matching specific register queries.

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
