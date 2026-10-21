import json
from pathlib import Path

from langchain_core.documents import Document

EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", "venv", ".env"}
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
    ".jsx", ".tsx", ".md", ".txt", ".json",
    ".yaml", ".yml", ".toml", ".ipynb",
}


def is_remote(source: str) -> bool:
    """True when the source looks like a Git URL rather than a local path."""
    return source.startswith(("http://", "https://", "git@"))


def load_repo(repo_path: Path) -> list[Document]:
    """Load every supported file in a repository as a LangChain Document."""
    documents = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        documents.append(load_file(path))
    return documents


def load_file(path: Path) -> Document:
    """Convert a single file into a Document. Notebooks are parsed cell-wise."""
    if path.suffix == ".ipynb":
        content = extract_notebook_text(path)
    else:
        content = path.read_text(encoding="utf-8", errors="ignore")

    return Document(
        page_content=content,
        metadata={
            "filename": path.name,
            "extension": path.suffix,
        },
    )


def extract_notebook_text(path: Path) -> str:
    """Pull only the markdown/code cell sources out of a .ipynb file."""
    with open(path, encoding="utf-8") as f:
        notebook = json.load(f)

    parts = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        cell_type = cell.get("cell_type", "code")
        parts.append(f"[{cell_type}]\n{source}")

    return "\n\n".join(parts)
