from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app import config


@lru_cache(maxsize=1)
def _local_embeddings() -> Embeddings:
    """MiniLM via torch. Imported lazily so the API path never loads torch."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(api_key: str | None = None) -> Embeddings:
    """Return the embedding backend for this request.

    In "api" mode the key is resolved by app.llm.resolve_embedding_key, which
    prefers the caller's own key and falls back to the server key outside
    production. The cache applies only to the keyless local MiniLM model, so no
    visitor's key is ever retained.
    """
    if config.embeddings_mode() == "api":
        from app.embeddings_api import GeminiEmbeddings
        from app.llm import log_embedding_key, resolve_embedding_key

        key, source = resolve_embedding_key(api_key)
        log_embedding_key(source)
        return GeminiEmbeddings(api_key=key)
    return _local_embeddings()


def build_index(chunks: list[Document], api_key: str | None = None) -> InMemoryVectorStore:
    """Embed all chunks once and return a searchable in-memory vector store."""
    vector_store = InMemoryVectorStore(embedding=get_embeddings(api_key))
    vector_store.add_documents(chunks)
    return vector_store


def retrieve(
    vector_store: InMemoryVectorStore,
    query: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """Return the top-k chunks closest to the query, with similarity scores."""
    return vector_store.similarity_search_with_score(query, k=k)
