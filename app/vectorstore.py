from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.embedding import embeddings


def build_index(chunks: list[Document]) -> InMemoryVectorStore:
    """Embed all chunks once and return a searchable in-memory vector store."""
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def retrieve(
    vector_store: InMemoryVectorStore,
    query: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """Return the top-k chunks closest to the query, with similarity scores."""
    return vector_store.similarity_search_with_score(query, k=k)