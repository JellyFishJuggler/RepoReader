"""RepoReader web server: a thin Flask API over the same RAG pipeline the CLI uses."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory

from app.loader import is_remote
from app.llm import key_status, resolve_model
from app.rag import NOT_FOUND, ask, build_vectorstore

app = Flask(__name__, static_folder="static", static_url_path="")

_store_lock = Lock()
_stores: dict[str, tuple] = {}


@app.after_request
def _no_store(response):
    """Stop the browser from serving stale HTML/JS/CSS after an update."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    """Which providers are configured and available."""
    return jsonify(key_status())


def _repo_label(source: str) -> str:
    if is_remote(source):
        return source.rstrip("/").split("/")[-1].removesuffix(".git")
    return Path(source).name


def _get_or_build_store(repo_source: str):
    """Return (store, files, chunks, label) for a repo, indexing it if needed."""
    with _store_lock:
        cached = _stores.get(repo_source)
        if cached is not None:
            return cached

        if is_remote(repo_source):
            with TemporaryDirectory() as tmp:
                import git
                repo = git.Repo.clone_from(repo_source, tmp)
                built = build_vectorstore(Path(repo.working_tree_dir))
        else:
            path = Path(repo_source)
            if not path.exists():
                raise FileNotFoundError(f"Local path does not exist: {repo_source}")
            built = build_vectorstore(path)

        store, file_count, chunk_count = built
        entry = (store, file_count, chunk_count, _repo_label(repo_source))
        _stores[repo_source] = entry
        return entry


@app.post("/api/index")
def api_index():
    """Index a repository. Body: {"repo": "<url or local path>"}."""
    payload = request.get_json(silent=True) or {}
    repo_source = (payload.get("repo") or "").strip()
    if not repo_source:
        return jsonify(error="Missing 'repo' field."), 400

    try:
        store, file_count, chunk_count, label = _get_or_build_store(repo_source)
    except Exception as exc:
        return jsonify(error=f"Could not load repository: {exc}"), 400

    return jsonify(
        repo=repo_source,
        label=label,
        files=file_count,
        chunks=chunk_count,
        indexed=True,
    )


@app.post("/api/ask")
def api_ask():
    """Answer a question against an indexed repo."""
    payload = request.get_json(silent=True) or {}
    repo_source = (payload.get("repo") or "").strip()
    question = (payload.get("question") or "").strip()
    source = payload.get("source") or "Gemini"

    if not repo_source:
        return jsonify(error="Missing 'repo' field."), 400
    if not question:
        return jsonify(error="Missing 'question' field."), 400

    try:
        store = _stores.get(repo_source)
        if store is None:
            store, _, _, _ = _get_or_build_store(repo_source)
        vector_store = store[0]
    except Exception as exc:
        return jsonify(error=f"Repository is not indexed: {exc}"), 400

    try:
        chat_model = resolve_model(
            source,
            model=payload.get("model"),
            api_key=payload.get("api_key"),
            base_url=payload.get("base_url"),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400

    try:
        k = int(payload.get("k") or 4)
        answer, sources = ask(vector_store, question, k=k, chat_model=chat_model)
    except Exception as exc:
        return jsonify(error=f"Something went wrong while answering: {exc}"), 500

    return jsonify(
        answer=answer or NOT_FOUND,
        sources=sources,
        found=bool(answer and answer != NOT_FOUND),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
