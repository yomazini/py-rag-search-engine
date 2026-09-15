import os
from typing import Any

from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import Movie, SearchResult, format_search_result, load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(
        self, documents: list[Movie] | None = None, model_name: str = "clip-ViT-B-32"
    ) -> None:
        if documents is None:
            documents = []
        self.documents = documents
        self.texts: list[str] = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> NDArray[Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        image_embedding = self.model.encode([image])  # type: ignore[arg-type]
        return image_embedding[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[SearchResult]:
        image_embedding = self.embed_image(image_path)

        similarities: list[tuple[int, float]] = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in similarities[:limit]:
            doc = self.documents[idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:100],
                    score=score,
                )
            )

        return results


def verify_image_embedding(image_path: str) -> None:
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(
    image_path: str = "data/paddington.jpeg", limit: int = 5
) -> dict[str, str | list[SearchResult]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    movies = load_movies()
    searcher = MultimodalSearch(movies)
    results = searcher.search_with_image(image_path, limit)

    return {"image_path": image_path, "results": results}
