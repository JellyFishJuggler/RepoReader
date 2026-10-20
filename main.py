import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from git import Repo

from app.loader import is_remote
from app.rag import NOT_FOUND, ask, build_vectorstore

load_dotenv()


def open_repo(source: str) -> tuple[Path, object]:
    """Return (repo_path, cleanup). Clones remote URLs into a temp dir."""
    if is_remote(source):
        tmp_dir = TemporaryDirectory()
        repo = Repo.clone_from(source, tmp_dir.name)
        return Path(repo.working_tree_dir), tmp_dir
    return Path(source), None


def print_answer(answer: str, sources: list[str]) -> None:
    print(f"Answer: {answer}\n")
    if sources:
        print("Sources:")
        for source in sources:
            print(f"- {source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about any GitHub repository using RAG."
    )
    parser.add_argument(
        "-r", "--repo",
        help="GitHub URL or local path of the repository (prompted if omitted)",
    )
    parser.add_argument(
        "-q", "--query",
        help="Ask a single question and exit (defaults to an interactive session)",
    )
    parser.add_argument(
        "-k", "--k",
        type=int,
        default=4,
        help="Number of context chunks to retrieve (default: 4)",
    )
    args = parser.parse_args()

    repo_source = args.repo or input("Repository URL or path: ").strip()

    repo_path, cleanup = open_repo(repo_source)
    try:
        vector_store, file_count, chunk_count = build_vectorstore(repo_path)
        print(f"Indexed {file_count} files as {chunk_count} chunks.\n")

        if args.query:
            answer, sources = ask(vector_store, args.query, k=args.k)
            print_answer(answer if answer else NOT_FOUND, sources)
            return

        print("Ask questions about the repository. Type 'exit' to quit.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue

            answer, sources = ask(vector_store, question, k=args.k)
            print_answer(answer if answer else NOT_FOUND, sources)
    finally:
        if cleanup:
            cleanup.cleanup()


if __name__ == "__main__":
    main()