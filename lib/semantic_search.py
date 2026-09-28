from __future__ import annotations

from typing import Any, Protocol


REQUIRED_DOCUMENT_FIELDS = ("id", "title", "category", "summary", "source")


DEFAULT_DOCUMENTS = [
    {
        "id": "DOC-101",
        "title": "Resetting a Forgotten Password",
        "category": "account",
        "summary": (
            "Explains how users can change or reset their account password "
            "after identity verification."
        ),
        "source": "platform-docs/account/password-reset",
        "tags": ["password", "account", "verification"],
    },
    {
        "id": "DOC-102",
        "title": "Fixing Invalid API Authentication Tokens",
        "category": "api",
        "summary": (
            "Explains how to resolve API calls blocked by expired, missing, "
            "or malformed bearer credentials."
        ),
        "source": "platform-docs/api/authentication-tokens",
        "tags": ["api", "authentication", "credentials"],
    },
    {
        "id": "DOC-103",
        "title": "Understanding Monthly Billing Limits",
        "category": "billing",
        "summary": (
            "Explains how usage limits affect invoices, monthly plans, "
            "and account upgrades."
        ),
        "source": "platform-docs/billing/monthly-limits",
        "tags": ["billing", "invoice", "plan"],
    },
    {
        "id": "DOC-104",
        "title": "Troubleshooting Slow Dashboard Loading",
        "category": "performance",
        "summary": (
            "Explains common causes of slow page loads, dashboard latency, "
            "and client-side rendering delays."
        ),
        "source": "platform-docs/performance/dashboard-loading",
        "tags": ["dashboard", "performance", "latency"],
    },
    {
        "id": "DOC-105",
        "title": "Updating User Profile Settings",
        "category": "account",
        "summary": (
            "Explains how users can update email, display name, "
            "notification preferences, and profile details."
        ),
        "source": "platform-docs/account/profile-settings",
        "tags": ["profile", "settings", "email"],
    },
]


class EmbeddingModel(Protocol):
    """Protocol for any embedding model used by the retrieval workflow."""

    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for the provided text."""
        ...


class OllamaEmbeddingModel:
    """
    Optional local embedding model wrapper.

    This class is not required by the pytest suite. It is provided so you can
    try the same workflow locally with Ollama after your core functions work.

    Example:
        embedder = OllamaEmbeddingModel(model_name="embeddinggemma")
        results = semantic_search(
            "Why does the mobile app say my token is expired?",
            DEFAULT_DOCUMENTS,
            embedder,
            top_k=3,
        )
    """

    def __init__(self, model_name: str = "embeddinggemma"):
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "The ollama package is required to use OllamaEmbeddingModel. "
                "Install dependencies with pipenv install."
            ) from exc

        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]


def build_search_text(document: dict[str, Any]) -> str:
    """
    Build the text that should be sent to the embedding model for one document.

    Requirements:
    - Include the document title.
    - Include the document category.
    - Include the document summary.
    - Include tags when available.
    - Return one clean, non-empty string.

    Why:
    Embedding only the title may lose important meaning. Embedding title,
    category, summary, and tags gives the model more context.
    """
    raise NotImplementedError("TODO: Build searchable text from document fields.")


def prepare_documents(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate and prepare raw documents for retrieval.

    Requirements:
    - Accept a list of document dictionaries.
    - Validate that each document includes REQUIRED_DOCUMENT_FIELDS.
    - Return a new list of document dictionaries.
    - Do not mutate the original input documents.
    - Add a 'text' field created by build_search_text(document).
    - Preserve useful metadata: id, title, category, summary, source, and tags.

    Raises:
    - ValueError if a required field is missing.
    - ValueError if raw_documents is empty.
    """
    raise NotImplementedError("TODO: Validate and prepare documents.")


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Requirements:
    - Return a float.
    - Return 1.0 for identical non-zero vectors.
    - Return 0.0 for orthogonal vectors.
    - Return 0.0 if either vector has zero magnitude.
    - Raise ValueError if the vectors have different dimensions.

    Do not use numpy for this lab. Implement the math with basic Python.
    """
    raise NotImplementedError("TODO: Compute cosine similarity.")


def embed_documents(
    prepared_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
) -> list[dict[str, Any]]:
    """
    Embed each prepared document.

    Requirements:
    - Accept documents that already include a 'text' field.
    - Call embedding_model.embed(document["text"]) once per document.
    - Return a new list of document dictionaries.
    - Add an 'embedding' field to each returned document.
    - Preserve metadata needed for source traceability.

    Do not mutate the input documents.
    """
    raise NotImplementedError("TODO: Embed each prepared document.")


def rank_documents(
    query: str,
    embedded_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Rank embedded documents against a user query.

    Requirements:
    - Validate that query is a non-empty string.
    - Validate that top_k is a positive integer.
    - Embed the query with embedding_model.embed(query).
    - Compare the query embedding to every document embedding.
    - Add a 'score' field to each returned result.
    - Sort results by score from highest to lowest.
    - Return only the top_k results.
    - Preserve source metadata: id, title, category, summary, and source.

    The returned result format should look like:
        {
            "id": "DOC-102",
            "title": "Fixing Invalid API Authentication Tokens",
            "category": "api",
            "summary": "...",
            "source": "platform-docs/api/authentication-tokens",
            "score": 0.87
        }
    """
    raise NotImplementedError("TODO: Rank documents by query similarity.")


def semantic_search(
    query: str,
    raw_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Run the full semantic retrieval workflow.

    Required sequence:
    1. Prepare documents.
    2. Embed documents.
    3. Embed the query.
    4. Compute similarity scores.
    5. Rank results.
    6. Return top-k results with source metadata.

    This function should orchestrate the smaller helper functions.
    """
    raise NotImplementedError("TODO: Run the full semantic search workflow.")


def main() -> None:
    """
    Optional manual run.

    This is not used by pytest. It is here so you can test your workflow locally
    after implementing the required functions.
    """
    embedder = OllamaEmbeddingModel()
    query = "Why does the mobile app say my token is expired?"

    results = semantic_search(
        query=query,
        raw_documents=DEFAULT_DOCUMENTS,
        embedding_model=embedder,
        top_k=3,
    )

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']} | score={result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Summary: {result['summary']}")


if __name__ == "__main__":
    main()