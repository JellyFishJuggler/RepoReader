from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from app.chunking import chunk_documents
from app.loader import load_repo
from app.llm import get_chat_model
from app.vectorstore import build_index, retrieve

NOT_FOUND = "Not found in repository."

SYSTEM_PROMPT = """You are an assistant that answers questions about a GitHub repository.

You must answer using ONLY the retrieved context below. Follow these rules:

1. Base your answer strictly on the context. Do not use outside knowledge.
2. If the context does NOT contain the answer, reply with exactly:
   Not found in repository.
3. Never invent facts, file names, or results that are not in the context.
4. Be concise and directly address the question.

Retrieved context:
{context}"""


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )


def build_vectorstore(repo_path, api_key: str | None = None) -> tuple[InMemoryVectorStore, int, int]:
    """Load, chunk and embed a repository. Returns (store, files, chunks)."""
    documents = load_repo(repo_path)
    chunks = chunk_documents(documents)
    vector_store = build_index(chunks, api_key=api_key)
    return vector_store, len(documents), len(chunks)


def ask(
    vector_store: InMemoryVectorStore,
    question: str,
    k: int = 4,
    chat_model=None,
) -> tuple[str, list[str]]:
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
    model = chat_model or get_chat_model()
    response = model.invoke(messages)

    return response.content.strip(), sources
