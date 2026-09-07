from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks and tag each with its source file.

    The `[filename: ...]` prefix is embedded along with the text so that
    queries mentioning a file name (e.g. "what does the README say?") match
    the right chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        filename = chunk.metadata.get("filename", "unknown")
        chunk.page_content = f"[filename: {filename}]\n{chunk.page_content}"

    return chunks