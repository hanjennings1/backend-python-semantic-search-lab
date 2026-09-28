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
    # === Compute cosine similarity between two vectors ===
    # Vectors must be the same length to compare position by position
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions.")

    dot_product = 0.0
    sum_squares_a = 0.0
    sum_squares_b = 0.0

    # One pass through both lists, building all three totals at once
    for i in range(len(vector_a)):
        dot_product += vector_a[i] * vector_b[i]
        sum_squares_a += vector_a[i] ** 2
        sum_squares_b += vector_b[i] ** 2

    # A zero vector has no direction, so return 0.0 instead of dividing by zero
    if sum_squares_a == 0 or sum_squares_b == 0:
        return 0.0

    # ** 0.5 is a square root; one root of the product avoids rounding issues
    return dot_product / (sum_squares_a * sum_squares_b) ** 0.5


def embed_documents(
    prepared_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
) -> list[dict[str, Any]]:
    
    # Embed each prepared document; Do not mutate the input documents.
    embedded = []  # new list, so the input list is never modified

    for document in prepared_documents:
        # Can't embed a document that hasn't been prepared
        if "text" not in document:
            raise ValueError(f"Document {document.get('id', '<unknown>')} has no 'text' field.")

        # Copy so the original prepared document isn't changed
        new_document = document.copy()

        # One embed call per document; the vector is stored alongside the metadata
        new_document["embedding"] = embedding_model.embed(document["text"])

        embedded.append(new_document)

    return embedded


def rank_documents(
    query: str,
    embedded_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    # Rank embedded documents against a user query.

    # Query must be real text, not empty or just spaces
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # top_k must be a whole number of at least 1 (bool is excluded since True counts as int)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # Embed the query once, with the same model used for the documents
    query_embedding = embedding_model.embed(query)

    results = []
    for document in embedded_documents:
        # Compare the query's direction to this document's direction
        score = cosine_similarity(query_embedding, document["embedding"])

        # Keep only user-facing metadata plus the score; drop text and embedding
        results.append({
            "id": document["id"],
            "title": document["title"],
            "category": document["category"],
            "summary": document["summary"],
            "source": document["source"],
            "score": score,
        })

    # Highest score first
    results.sort(key=lambda result: result["score"], reverse=True)

    # Slicing never errors if top_k is larger than the list
    return results[:top_k]


def semantic_search(
    query: str,
    raw_documents: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    # Run the full semantic retrieval workflow.

    # 1. Validate documents and add searchable text
    prepared = prepare_documents(raw_documents)

    # 2. Turn each document's text into a vector
    embedded = embed_documents(prepared, embedding_model)

    # 3–6. Embed the query, score, sort, and keep the top_k matches
    return rank_documents(query, embedded, embedding_model, top_k)


def main() -> None:
    """
    Optional manual run.

    This is not used by pytest. It is here so you can test your workflow locally
    after implementing the required functions.
    """
    embedder = OllamaEmbeddingModel()
    query = "How much is my monthly charge?"    # changed in testing

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