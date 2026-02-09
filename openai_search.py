#!/usr/bin/env python3
"""Search OpenAI vector stores for comparison with local ChromaDB."""

import argparse
import os
import sys
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def search_vector_store(
    client: OpenAI,
    vector_store_id: str,
    query: str,
    max_results: int = 5,
    rerank: bool = True,
    score_threshold: float = 0.0,
):
    """Search an OpenAI vector store."""
    ranking_options = {
        "score_threshold": score_threshold,
    }
    if rerank:
        ranking_options["ranker"] = "auto"

    results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=max_results,
        ranking_options=ranking_options,
    )
    return results


def format_results(results, query: str):
    """Format results for display."""
    print(f'\nSearch results for: "{query}"\n')

    if not results.data:
        print("No results found.")
        return

    for i, result in enumerate(results.data, 1):
        score = getattr(result, "score", None)
        filename = result.filename

        # Combine all content parts
        content_parts = []
        for part in result.content:
            content_parts.append(part.text)
        content = " ".join(content_parts)

        # Truncate long content
        if len(content) > 200:
            content = content[:200] + "..."

        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"[{i}] Score: {score_str} | File: {filename}")
        print(f'    "{content}"\n')


def main():
    parser = argparse.ArgumentParser(
        description="Search OpenAI vector stores"
    )
    parser.add_argument(
        "vector_store_id",
        help="OpenAI vector store ID (e.g., vs_abc123...)",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Score threshold (default: 0.0)",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    start_time = time.perf_counter()
    results = search_vector_store(
        client=client,
        vector_store_id=args.vector_store_id,
        query=args.query,
        max_results=args.top_k,
        rerank=not args.no_rerank,
        score_threshold=args.threshold,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    format_results(results, args.query)
    print(f"Search time: {elapsed_ms:.0f}ms")


if __name__ == "__main__":
    main()
