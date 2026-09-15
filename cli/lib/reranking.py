import json
import os
from time import sleep
from typing import Literal, NotRequired

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from .search_utils import SearchResult


class RerankedSearchResult(SearchResult, total=False):
    individual_score: NotRequired[int]
    batch_rank: NotRequired[int]
    crossencoder_score: NotRequired[float]


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


def llm_rerank_individual(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    scored_docs: list[RerankedSearchResult] = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        score_text = (response.choices[0].message.content or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def llm_rerank_batch(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    if not documents:
        return []

    doc_map: dict[int, SearchResult] = {}
    doc_list: list[str] = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return the movie IDs in order of relevance, best match first.

    Your response must be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    ranking_text = (response.choices[0].message.content or "").strip()

    parsed_ids = json.loads(ranking_text)

    reranked: list[RerankedSearchResult] = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    return reranked[:limit]


def cross_encoder_rerank(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    pairs: list[list[str]] = []
    reranked_documents: list[RerankedSearchResult] = [
        RerankedSearchResult(**doc) for doc in documents
    ]
    for doc in reranked_documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(reranked_documents, scores):
        doc["crossencoder_score"] = float(score)

    reranked_documents.sort(key=lambda x: float(x["crossencoder_score"]), reverse=True)
    return reranked_documents[:limit]


def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual", "batch", "cross_encoder"] = "batch",
    limit: int = 5,
) -> list[RerankedSearchResult]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    if method == "batch":
        return llm_rerank_batch(query, documents, limit)
    if method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)
    else:
        return documents[:limit]
