from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.chunking import chunk_documents
from app.llm import get_chat_model
from app.loader import load_repo
from app.prompt import build_rag_prompt
from app.vectorstore import build_index, retrieve

NOT_FOUND = "Not found in repository."


def build_vectorstore(repo_path) -> tuple[InMemoryVectorStore, int, int]:
    """Load, chunk and embed a repository. Returns (store, files, chunks)."""
    documents = load_repo(repo_path)
    chunks = chunk_documents(documents)
    vector_store = build_index(chunks)
    return vector_store, len(documents), len(chunks)


def ask(vector_store: InMemoryVectorStore, question: str, k: int = 4) -> tuple[str, list[str]]:
    """Retrieve the top-k chunks and generate an answer grounded in them.

    Returns (answer_text, sources) where sources is the sorted list of file
    names the answer was built from.
    """
    hits = retrieve(vector_store, question, k=k)
    if not hits:
        return NOT_FOUND, []

    context = "\n\n".join(doc.page_content for doc, _ in hits)
    sources = sorted({doc.metadata.get("filename", "unknown") for doc, _ in hits})

    prompt_template = build_rag_prompt()
    messages = prompt_template.format_messages(context=context, question=question)
    response = get_chat_model().invoke(messages)

    return response.content.strip(), sources