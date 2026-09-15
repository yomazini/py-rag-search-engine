# Python Hybrid & Multimodal RAG Search Engine

A high-performance retrieval and search engine pipeline built from first principles in Python. Implements BM25 lexical search, dense semantic search (`all-MiniLM-L6-v2`), Reciprocal Rank Fusion (RRF), Cross-Encoder & LLM re-ranking, and Multimodal CLIP retrieval (`clip-ViT-B-32`).

Powered by [Boot.dev](https://boot.dev).

---

## 🏗️ Core Architecture

```
                       [User Query / Image]
                                │
                 [Query Enhancement Pipeline]
               (Spell Correction / HyDE / Expansion)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
 [BM25 Lexical Engine]                       [Dense Semantic Engine]
  • Inverted Index & Stopwords                • all-MiniLM-L6-v2 (384-d)
  • Term Saturation (k1=1.5)                  • L2 Normalization (Unit vectors)
  • Length Penalty (b=0.75)                   • Cosine Similarity
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
                   [Reciprocal Rank Fusion (RRF)]
                     Score = ∑ 1 / (k + rank_i)
                                │ Top Candidates
                                ▼
                   [Two-Stage Re-Ranking Layer]
        • Cross-Encoder (ms-marco-TinyBERT-L2-v2)
        • LLM Batch / Individual Re-ranking (OpenRouter)
                                │ Top-K Context Passages
                                ▼
                   [Augmented Generation (RAG)]
                     Context-Grounded LLM Output
```

---

## 🚀 Key Features

- **Lexical Search (BM25):** Custom-built inverted index with term frequency saturation ($k_1 = 1.5$) and document length normalization ($b = 0.75$).
- **Dense Semantic Search:** SentenceTransformer vector embeddings with $L_2$ normalization and cosine similarity.
- **Semantic Document Chunking:** Sliding window segmentation with configurable chunk size and token overlap.
- **Hybrid Search (RRF):** Reciprocal Rank Fusion combining sparse keyword and dense embedding ranks without scale bias.
- **Query Enhancement:** Automated query rewriting, spell correction, and HyDE (Hypothetical Document Embeddings).
- **Two-Stage Re-Ranking:**
  - **Cross-Encoder:** Full-attention scoring with `ms-marco-TinyBERT-L2-v2`.
  - **LLM Re-ranking:** Individual document rating and batch permutation ranking via OpenRouter.
- **Multimodal Search:** Cross-modal image-to-text retrieval using OpenAI's CLIP (`clip-ViT-B-32`) vision transformer.
- **Evaluation Suite:** Benchmark framework measuring Precision@k, Recall@k, and Mean Reciprocal Rank (MRR) against golden test sets.

---

## 📦 Prerequisites

- **Python 3.12+**
- [`uv`](https://docs.astral.sh/uv/) (Astral Python package manager)
- OpenRouter API key (optional, for LLM enhancement and re-ranking)

---

## 🛠️ Setup

1. **Clone the repository:**

   ```bash
   git clone git@github.com:yomazini/py-rag-search-engine.git
   cd py-rag-search-engine
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Add your OPENROUTER_API_KEY to .env (if using LLM features)
   ```

3. **Install dependencies:**

   ```bash
   uv sync
   uv sync --extra ml --extra dev # Installs ML dependencies as well
   ```

---

## 💻 CLI Usage Guide

### 1. Lexical (BM25) Search

```bash
# Term Frequency & BM25 IDF inspection
uv run cli/keyword_search_cli.py tf 1 "matrix"
uv run cli/keyword_search_cli.py bm25idf "grizzly"

# Full BM25 search
uv run cli/keyword_search_cli.py bm25search "bear adventure" --limit 5
```

### 2. Dense Semantic Search

```bash
# Verify embedding model and precompute vectors
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py verify_embeddings

# Semantic vector search
uv run cli/semantic_search_cli.py search "inspirational sports story" --limit 5

# Sliding window chunking & chunk-level search
uv run cli/semantic_search_cli.py embed_chunks
```

### 3. Hybrid Search (RRF) & Re-Ranking

```bash
# Reciprocal Rank Fusion (BM25 + Semantic)
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --limit 5

# Hybrid Search with Cross-Encoder Re-Ranking
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method cross_encoder --limit 5

# Hybrid Search with LLM Batch Re-Ranking
uv run cli/hybrid_search_cli.py rrf-search "scary movie in space" --rerank-method batch --limit 5
```

### 4. Multimodal Search (CLIP)

```bash
# Verify image embedding dimensions (512-d)
uv run cli/multimodal_search_cli.py verify_image_embedding "data/paddington.jpeg"

# Image-to-Text Movie Search
uv run cli/multimodal_search_cli.py search_image "data/paddington.jpeg" --limit 5
```

### 5. Evaluation Suite

```bash
# Run benchmark evaluation across golden test queries
uv run cli/evaluation_cli.py evaluate --k 5
```

---

## 🎓 Profile & Verification

- **Boot.dev Profile:** [boot.dev/u/thejoceph](https://www.boot.dev/u/thejoceph)
