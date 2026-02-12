#!/usr/bin/env python3
"""CLI interface for vector_db semantic search."""

import argparse
import sys
import time
from pathlib import Path

from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from vector_store import (
    VectorStore,
    create_database,
    list_databases,
    delete_database,
    database_exists,
)


def cmd_create_db(args):
    """Create a new database."""
    if database_exists(args.db_name):
        print(f"Error: Database '{args.db_name}' already exists.")
        return 1

    if create_database(args.db_name):
        print(f"Created database: {args.db_name}")
        return 0
    else:
        print(f"Error: Failed to create database '{args.db_name}'")
        return 1


def cmd_list_dbs(args):
    """List all databases."""
    databases = list_databases()

    if not databases:
        print("No databases found.")
        return 0

    print("Available databases:")
    for db in databases:
        store = VectorStore(db)
        stats = store.get_stats()
        print(f"  - {db} ({stats['total_chunks']} chunks, {len(stats['documents'])} documents)")
    return 0


def cmd_delete_db(args):
    """Delete a database."""
    if not database_exists(args.db_name):
        print(f"Error: Database '{args.db_name}' not found.")
        return 1

    if delete_database(args.db_name):
        print(f"Deleted database: {args.db_name}")
        return 0
    else:
        print(f"Error: Failed to delete database '{args.db_name}'")
        return 1


def cmd_ingest(args):
    """Ingest a PDF or text/markdown file into a database."""
    if not database_exists(args.db_name):
        print(f"Error: Database '{args.db_name}' not found. Create it first with 'create-db'.")
        return 1

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    suffix = file_path.suffix.lower()
    supported = [".pdf", ".md", ".txt"]
    if suffix not in supported:
        print(f"Error: Unsupported file type '{suffix}'. Supported: {supported}")
        return 1

    print(f"Processing: {file_path.name}")

    # Parse extra metadata from --meta arguments
    extra_metadata = {}
    if args.meta:
        for meta in args.meta:
            if "=" in meta:
                key, value = meta.split("=", 1)
                extra_metadata[key] = value

    # Process file based on type
    if suffix == ".pdf":
        processor = PDFProcessor()
        chunks = processor.process_pdf(file_path, extra_metadata)
    else:
        processor = TextProcessor()
        chunks = processor.process_file(file_path, extra_metadata)

    print(f"  Extracted {len(chunks)} chunks")

    # Add to vector store
    def progress(batch_num, total_batches):
        print(f"\r  Embedding batch {batch_num}/{total_batches}...", end="", flush=True)

    store = VectorStore(args.db_name)
    added = store.add_documents(chunks, progress_callback=progress)
    print(f"\n  Added {added} chunks to database '{args.db_name}'")

    return 0


def cmd_search(args):
    """Search a database."""
    if not database_exists(args.db_name):
        print(f"Error: Database '{args.db_name}' not found.")
        return 1

    # Parse filters
    where = None
    if args.filter:
        where = {}
        for f in args.filter:
            if "=" in f:
                key, value = f.split("=", 1)
                # Try to convert to int if possible (for page numbers)
                try:
                    value = int(value)
                except ValueError:
                    pass
                where[key] = value

    store = VectorStore(args.db_name)

    start_time = time.perf_counter()
    # Expand candidate pool when post-processing (rerank/keyword boost) is enabled
    do_rerank = args.rerank or args.rerank_local or args.rerank_bge
    fetch_k = args.top_k * 5 if (do_rerank or args.keyword_boost) else args.top_k
    results = store.search(args.query, n_results=fetch_k, where=where)

    # Apply reranking if requested
    if do_rerank and results:
        try:
            from reranker import get_reranker
            if args.rerank_bge:
                provider = "bge"
            elif args.rerank_local:
                provider = "local"
            else:
                provider = "cohere"
            reranker = get_reranker(provider)
            results = reranker.rerank(args.query, results, top_n=fetch_k)  # Keep all for keyword boost
        except ValueError as e:
            print(f"Warning: Reranking disabled - {e}")

    # Apply keyword boost after reranking (if requested)
    if args.keyword_boost and results:
        results = store._apply_keyword_boost(args.query, results)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not results:
        print("No results found.")
        return 0

    print(f"\nSearch results for: \"{args.query}\"\n")
    for i, result in enumerate(results[:args.top_k], 1):
        score = result["score"]
        source = result["metadata"].get("source", "unknown")
        page = result["metadata"].get("page", "?")
        section = result["metadata"].get("section", "")
        text = result["text"]

        # Truncate long text for display
        if len(text) > 200:
            text = text[:200] + "..."

        location = f"Page: {page}" if page != "?" else f"Section: {section[:30]}" if section else ""
        print(f"[{i}] Score: {score:.2f} | Source: {source} | {location}")
        print(f"    \"{text}\"\n")

    print(f"Search time: {elapsed_ms:.0f}ms")
    return 0


def cmd_list_docs(args):
    """List documents in a database."""
    if not database_exists(args.db_name):
        print(f"Error: Database '{args.db_name}' not found.")
        return 1

    store = VectorStore(args.db_name)
    docs = store.list_documents()

    if not docs:
        print(f"No documents in database '{args.db_name}'.")
        return 0

    print(f"Documents in '{args.db_name}':")
    for doc in docs:
        print(f"  - {doc}")
    return 0


def cmd_delete_doc(args):
    """Delete a document from a database."""
    if not database_exists(args.db_name):
        print(f"Error: Database '{args.db_name}' not found.")
        return 1

    store = VectorStore(args.db_name)
    deleted = store.delete_document(args.doc_name)

    if deleted > 0:
        print(f"Deleted {deleted} chunks from '{args.doc_name}'")
        return 0
    else:
        print(f"Document '{args.doc_name}' not found in database.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Vector DB with Semantic Search over PDF documents"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create-db
    p_create = subparsers.add_parser("create-db", help="Create a new database")
    p_create.add_argument("db_name", help="Name of the database to create")
    p_create.set_defaults(func=cmd_create_db)

    # list-dbs
    p_list = subparsers.add_parser("list-dbs", help="List all databases")
    p_list.set_defaults(func=cmd_list_dbs)

    # delete-db
    p_delete = subparsers.add_parser("delete-db", help="Delete a database")
    p_delete.add_argument("db_name", help="Name of the database to delete")
    p_delete.set_defaults(func=cmd_delete_db)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a file into a database")
    p_ingest.add_argument("db_name", help="Target database name")
    p_ingest.add_argument("file_path", help="Path to file (.pdf, .md, .txt)")
    p_ingest.add_argument(
        "--meta",
        action="append",
        help="Extra metadata (key=value), can be repeated",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # search
    p_search = subparsers.add_parser("search", help="Search a database")
    p_search.add_argument("db_name", help="Database to search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument(
        "--filter",
        action="append",
        help="Filter by metadata (key=value), can be repeated",
    )
    p_search.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank results using Cohere API (requires COHERE_API_KEY)",
    )
    p_search.add_argument(
        "--rerank-local",
        action="store_true",
        help="Rerank results locally using FlashRank (free, no API)",
    )
    p_search.add_argument(
        "--rerank-bge",
        action="store_true",
        help="Rerank using BGE reranker (requires torch, best local quality)",
    )
    p_search.add_argument(
        "--keyword-boost",
        action="store_true",
        help="Boost results containing exact query terms (helps MAPR vs MAPR2)",
    )
    p_search.set_defaults(func=cmd_search)

    # list-docs
    p_listdocs = subparsers.add_parser(
        "list-docs", help="List documents in a database"
    )
    p_listdocs.add_argument("db_name", help="Database name")
    p_listdocs.set_defaults(func=cmd_list_docs)

    # delete-doc
    p_deldoc = subparsers.add_parser(
        "delete-doc", help="Delete a document from a database"
    )
    p_deldoc.add_argument("db_name", help="Database name")
    p_deldoc.add_argument("doc_name", help="Document filename to delete")
    p_deldoc.set_defaults(func=cmd_delete_doc)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
