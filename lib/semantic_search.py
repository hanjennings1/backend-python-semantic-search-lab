from __future__ import annotations
from typing import Any, Protocol
import copy

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
    # Text that should be sent to the embedding model for one document:
    # include title, category, summary, (and tags)
    lines = []

    for field in ("title", "category", "summary"):
        value = str(document.get(field, "")).strip()
        if value:
            lines.append(f"{field.capitalize()}: {value}")

    # only include tags when availabe:
    tags = document.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    clean_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    if clean_tags:
        lines.append(f"Tags: {', '.join(clean_tags)}")

    # return one clean, non-empty string (this separates each part onto its own line)
    return "\n".join(lines)


def prepare_documents(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # === Validate and prepare raw documents for retrieval ===
    # Reject an empty list (or None) up front; there's nothing to search
    if not raw_documents:
        raise ValueError("raw_documents must be a non-empty list.")

    prepared = []  # new list, so the caller's list is never modified

    for document in raw_documents:
        # Collect every required field that's absent or blank (e.g. "title": "")
        missing = [
            field for field in REQUIRED_DOCUMENT_FIELDS
            if field not in document or not str(document[field]).strip()
        ]
        if missing:
            # .get() avoids a KeyError if "id" itself is the missing field
            raise ValueError(
                f"Document {document.get('id', '<unknown>')} is missing required fields: {missing}"
            )

        # .copy() makes a new dictionary with the same keys and values
        new_document = document.copy()

        # The tags list is still shared after .copy(), so give the new dict its own list
        if "tags" in new_document:
            new_document["tags"] = list(new_document["tags"])

        # Add the searchable text the embedding model will read
        new_document["text"] = build_search_text(new_document)

        prepared.append(new_document)

    return prepared



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