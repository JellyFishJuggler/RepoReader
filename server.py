"""RepoReader web server: a thin Flask API over the same RAG pipeline the CLI uses."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.parse import urlparse

from flask import Flask, jsonify, request, send_from_directory

from app import config
from app.llm import key_status, log_key_status, resolve_model
from app.loader import is_remote
from app.rag import NOT_FOUND, ask, build_vectorstore

_STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__)

_store_lock = Lock()
_stores: dict[str, tuple] = {}
# Temp checkouts must stay on disk while their vector store is cached.
_checkouts: dict[str, TemporaryDirectory] = {}


@app.after_request
def _headers(response):
    """Stop stale asset caching; keep CORS open only outside production."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    if not config.is_production():
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename: str):
    """Serve anything in static/. send_from_directory rejects `..` escapes."""
    return send_from_directory(_STATIC_DIR, filename)


@app.get("/api/status")
def status():
    """Which providers are configured and available."""
    return jsonify(key_status())


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def api_preflight(_any):
    """Answer CORS preflight so the UI can be served from another origin."""
    return ("", 204)


def _validate_repo_source(repo_source: str) -> None:
    """In production only public GitHub https URLs are accepted."""
    if not config.is_production():
        return
    if not repo_source.startswith("https://"):
        raise ValueError("Only https:// repository URLs are allowed.")
    host = (urlparse(repo_source).hostname or "").lower()
    if host not in config.ALLOWED_REPO_HOSTS:
        allowed = ", ".join(config.ALLOWED_REPO_HOSTS)
        raise ValueError(f"Only repositories hosted on {allowed} are allowed.")


def _clone_remote(repo_source: str) -> tuple[Path, TemporaryDirectory]:
    """Clone a remote repo into a temp dir, enforcing size and file limits.

    Returns the path together with the TemporaryDirectory handle, which the
    caller must keep alive: dropping it deletes the checkout mid-request.
    """
    import git

    tmp = TemporaryDirectory()
    # GitPython only supports kill_after_timeout on POSIX; passing it on
    # Windows raises outright, so it is applied where it is supported.
    kwargs = {}
    if os.name != "nt":
        kwargs["kill_after_timeout"] = config.CLONE_TIMEOUT_SECONDS

    try:
        git.Repo.clone_from(
            repo_source,
            tmp.name,
            depth=1,
            multi_options=["--single-branch"],
            **kwargs,
        )
    except Exception:
        tmp.cleanup()
        raise

    root = Path(tmp.name)
    file_count = sum(1 for p in root.rglob("*") if p.is_file())
    if file_count > config.MAX_REPO_FILES:
        tmp.cleanup()
        raise ValueError(
            f"Repository has {file_count} files, over the {config.MAX_REPO_FILES} limit."
        )

    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if total_bytes > config.MAX_REPO_BYTES:
        tmp.cleanup()
        limit_mb = config.MAX_REPO_BYTES // (1024 * 1024)
        raise ValueError(f"Repository is larger than the {limit_mb} MB limit.")

    return root, tmp


def _repo_label(source: str) -> str:
    if is_remote(source):
        return source.rstrip("/").split("/")[-1].removesuffix(".git")
    return Path(source).name


def _get_or_build_store(repo_source: str, api_key: str | None = None):
    """Return (store, files, chunks, label) for a repo, indexing it if needed.

    The caller's key is used for embedding and is deliberately not part of the
    cache key or the cached tuple.
    """
    with _store_lock:
        cached = _stores.get(repo_source)
        if cached is not None:
            return cached

        tmp = None
        if is_remote(repo_source):
            root, tmp = _clone_remote(repo_source)
            built = build_vectorstore(root, api_key=api_key)
        else:
            path = Path(repo_source)
            if not path.exists():
                raise FileNotFoundError(f"Local path does not exist: {repo_source}")
            built = build_vectorstore(path, api_key=api_key)

        store, file_count, chunk_count = built
        entry = (store, file_count, chunk_count, _repo_label(repo_source))
        _stores[repo_source] = entry
        # Hold the checkout open for as long as the cached store references it.
        if tmp is not None:
            _checkouts[repo_source] = tmp
        return entry


def _payload_key(payload: dict) -> str | None:
    """Extract the caller's API key without letting it leak elsewhere.

    Mirrors /api/ask: the key is only taken when the request says the caller is
    using their own key, so a stray field cannot silently switch sources.
    """
    using_own = bool(payload.get("api_key"))
    if not using_own:
        return None
    return (payload.get("api_key") or "").strip() or None


def _embed_key_or_error(payload: dict) -> tuple[str | None, str | None]:
    """Resolve the embedding key for a request.

    Returns (key, error_message). The key is None when the caller's own key is
    needed but missing, which the endpoints report as a 400.
    """
    from app.llm import resolve_embedding_key

    try:
        key, _ = resolve_embedding_key(_payload_key(payload))
        return key, None
    except RuntimeError as exc:
        return None, str(exc)


@app.post("/api/index")
def api_index():
    """Index a repository. Body: {"repo": ..., "api_key": ..., "source": ...}."""
    payload = request.get_json(silent=True) or {}
    repo_source = (payload.get("repo") or "").strip()

    if not repo_source:
        return jsonify(error="Missing 'repo' field."), 400

    api_key, key_error = _embed_key_or_error(payload)
    if key_error and config.embeddings_mode() == "api":
        return jsonify(error=key_error), 400

    try:
        _validate_repo_source(repo_source)
        store, file_count, chunk_count, label = _get_or_build_store(repo_source, api_key)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
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
    api_key = _payload_key(payload)

    if not repo_source:
        return jsonify(error="Missing 'repo' field."), 400
    if not question:
        return jsonify(error="Missing 'question' field."), 400

    try:
        store = _stores.get(repo_source)
        if store is None:
            # Re-indexing embeds with the same resolved key as /api/index.
            embed_key, key_error = _embed_key_or_error(payload)
            if key_error and config.embeddings_mode() == "api":
                return jsonify(error=key_error), 400
            store, _, _, _ = _get_or_build_store(repo_source, embed_key)
        vector_store = store[0]
    except Exception as exc:
        return jsonify(error=f"Repository is not indexed: {exc}"), 400

    try:
        chat_model = resolve_model(
            source,
            model=payload.get("model"),
            api_key=api_key,
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
    log_key_status()
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 5000)),
        debug=False,
        threaded=True,
    )
