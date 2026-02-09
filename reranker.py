"""Reranking support - Cohere API and local FlashRank."""

import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        pass


class CohereReranker(Reranker):
    """Reranker using Cohere's rerank API."""

    def __init__(self, api_key: str = None):
        import cohere

        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not set in environment")
        self.client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """Rerank search results using Cohere."""
        if not results:
            return results

        documents = [r["text"] for r in results]

        response = self.client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=top_n or len(results),
        )

        reranked = []
        for item in response.results:
            result = results[item.index].copy()
            result["original_score"] = result["score"]
            result["score"] = item.relevance_score
            reranked.append(result)

        return reranked


class FlashRankReranker(Reranker):
    """Local reranker using FlashRank (no API calls)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker

        # Available models:
        # - ms-marco-TinyBERT-L-2-v2 (~3MB, fastest, but poor for technical docs)
        # - ms-marco-MiniLM-L-12-v2 (default, ~22MB, good balance of speed/quality)
        # - rank-T5-flan (~110MB, slowest, best quality)
        self.ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """Rerank search results locally using FlashRank."""
        if not results:
            return results

        from flashrank import RerankRequest

        # Prepare passages for FlashRank
        passages = [{"id": i, "text": r["text"]} for i, r in enumerate(results)]

        request = RerankRequest(query=query, passages=passages)
        ranked = self.ranker.rerank(request)

        # Reorder results based on reranking
        top_n = top_n or len(results)
        reranked = []
        for item in ranked[:top_n]:
            idx = item["id"]
            result = results[idx].copy()
            result["original_score"] = result["score"]
            result["score"] = item["score"]
            reranked.append(result)

        return reranked


class BGEReranker(Reranker):
    """Local reranker using BAAI/bge-reranker-v2-m3 (requires torch)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.torch = torch

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """Rerank search results using BGE reranker."""
        if not results:
            return results

        # Score all query-document pairs
        scores = []
        with self.torch.no_grad():
            for result in results:
                inputs = self.tokenizer(
                    [[query, result["text"]]],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                score = self.model(**inputs, return_dict=True).logits.view(-1,).float().item()
                scores.append(score)

        # Create reranked results
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        top_n = top_n or len(results)
        reranked = []
        for score, result in scored_results[:top_n]:
            new_result = result.copy()
            new_result["original_score"] = result["score"]
            new_result["score"] = score
            reranked.append(new_result)

        return reranked


def get_reranker(provider: str = "cohere") -> Reranker:
    """Get a reranker instance.

    Args:
        provider: "cohere" for API-based, "local" for FlashRank, "bge" for BGE reranker
    """
    if provider == "local":
        return FlashRankReranker()
    elif provider == "bge":
        return BGEReranker()
    else:
        return CohereReranker()
