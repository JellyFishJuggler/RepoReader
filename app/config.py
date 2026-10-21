"""Environment-driven settings shared across the app."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def is_production() -> bool:
    """Production mode tightens paths, keys and request limits."""
    return os.getenv("PRODUCTION", "").strip().lower() in {"1", "true", "yes"}


def embeddings_mode() -> str:
    """Embedding backend: "local" (torch MiniLM) or "api" (Gemini)."""
    return os.getenv("EMBEDDINGS", "local").strip().lower() or "local"


GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIMS = int(os.getenv("GEMINI_EMBED_DIMS", "768"))
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))
EMBED_TIMEOUT_SECONDS = int(os.getenv("EMBED_TIMEOUT_SECONDS", "60"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "4"))

CLONE_TIMEOUT_SECONDS = int(os.getenv("CLONE_TIMEOUT_SECONDS", "60"))
MAX_REPO_FILES = int(os.getenv("MAX_REPO_FILES", "400"))
MAX_REPO_BYTES = int(os.getenv("MAX_REPO_BYTES", str(80 * 1024 * 1024)))

ALLOWED_REPO_HOSTS = ("github.com",)
