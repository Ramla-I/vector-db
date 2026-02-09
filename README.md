# Vector DB

A CLI tool for semantic search over PDF and Markdown documents using ChromaDB and OpenAI embeddings.

## Features

- **Multi-format support**: Ingest PDF, Markdown, and plain text files
- **Smart chunking**: Section-aware chunking with bidirectional overlap
- **Multiple databases**: Create isolated databases for different document collections
- **Semantic search**: Find relevant content using natural language queries
- **Hybrid search**: Combine semantic similarity with keyword matching
- **Multiple reranking options**: Cohere API, FlashRank (local), or BGE (local with torch)
- **Metadata filtering**: Filter results by source, page, or custom attributes
- **Technical document optimization**: Special preprocessing for register definitions, tables, and technical specs

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │  PDF/MD  │───▶│  Text Processor   │───▶│   Chunker    │───▶│ Embedder │ │
│  │   File   │    │  - Clean text     │    │  - Sections  │    │ (OpenAI) │ │
│  └──────────┘    │  - Extract tables │    │  - Overlap   │    └────┬─────┘ │
│                  │  - Add key terms  │    │  - Metadata  │         │       │
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
│  │         │    │ (OpenAI) │    │  Search  │    │(optional)│    │ Boost │  │
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

# Create virtual environment (Python 3.11 or 3.12 recommended for all features)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For BGE reranker support (optional, requires Python 3.12)
pip install torch 'transformers<4.45'

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

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
# Required: OpenAI API key for embeddings
OPENAI_API_KEY=your-api-key-here

# Optional: Cohere API key for --rerank option
COHERE_API_KEY=your-cohere-key

# Embedding provider (default: openai)
EMBEDDING_PROVIDER=openai

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
├── embeddings.py        # OpenAI embedding provider
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

| Operation | Time |
|-----------|------|
| Ingestion | ~60s (with embedding) |
| Search (basic) | ~1.3s |
| Search + keyword-boost | ~1.6s |
| Search + rerank-local | ~2.7s |
| Search + rerank (Cohere) | ~2.8s |
| Search + rerank-bge | ~16s |

## Accuracy Benchmarks

Tested with AFIO register queries (EVCR, MAPR, MAPR2):

| Method | Accuracy |
|--------|----------|
| Raw ChromaDB | 2/3 |
| `--keyword-boost` | **3/3** |
| `--rerank-local` | 2/3 |
| `--rerank-local --keyword-boost` | **3/3** |
| `--rerank` (Cohere) | **3/3** |

## Requirements

- Python 3.11+ (3.12 recommended for BGE reranker)
- OpenAI API key (for embeddings)
- Optional: Cohere API key (for `--rerank`)
- Optional: PyTorch (for `--rerank-bge`)

## License

MIT
