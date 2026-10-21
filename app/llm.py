import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app import config

# Resolve .env against the project root so the file is found regardless of the
# working directory. Real environment variables win over the file, which is what
# a hosted deployment needs.
ENV_PATH = config.ENV_PATH
load_dotenv(ENV_PATH, override=False)

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PROVIDER = "together"

TOKEN_ENV_NAMES = (
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_TOKEN",
    "TOGETHER_API_KEY",
)
GEMINI_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# Ollama support is disabled; kept for re-enable.
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

MODEL_SOURCES = ("Gemini", "Hugging Face", "My API key")


def _first_env(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty environment variable from `names`."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip("'\"")
    return None


def get_hf_token() -> str | None:
    """Resolve the Hugging Face token from the environment or .env."""
    return _first_env(TOKEN_ENV_NAMES)


def get_gemini_key() -> str | None:
    """Resolve the Gemini API key from the environment or .env."""
    return _first_env(GEMINI_ENV_NAMES)


# Ollama support is disabled; kept for re-enable.
# def list_ollama_models() -> list[str]:
#     """Query the running Ollama server for available models."""
#     import json
#     import urllib.request
#
#     root = OLLAMA_BASE_URL.rsplit("/", 1)[0]
#     try:
#         with urllib.request.urlopen(f"{root}/api/tags", timeout=5) as resp:
#             data = json.loads(resp.read())
#         return sorted(m["name"] for m in data.get("models", []))
#     except Exception:
#         return []


def build_chat_model(
    model: str,
    api_key: str,
    base_url: str | None = None,
    max_tokens: int | None = None,
):
    """Build a chat model over any OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        max_tokens=max_tokens,
        max_retries=2,
    )


def build_gemini_model(model: str = GEMINI_DEFAULT_MODEL):
    """Gemini via the .env key."""
    key = get_gemini_key()
    if not key:
        raise RuntimeError("No Gemini API key found. Add GEMINI_API_KEY=... to .env.")
    # Gemini 2.5 spends tokens thinking, so allow headroom.
    return build_chat_model(model, api_key=key, base_url=GEMINI_BASE_URL, max_tokens=4096)


# Ollama support is disabled; kept for re-enable.
# def build_ollama_model(model: str = OLLAMA_DEFAULT_MODEL):
#     """Local Ollama model."""
#     return build_chat_model(model, api_key="ollama", base_url=OLLAMA_BASE_URL, max_tokens=2048)


def build_custom_model(model: str, api_key: str, base_url: str | None = None):
    """User-supplied key against any OpenAI-compatible endpoint."""
    if not model or not api_key:
        raise RuntimeError("Provide both a model name and an API key.")
    return build_chat_model(model, api_key=api_key, base_url=base_url, max_tokens=4096)


def resolve_embedding_key(api_key: str | None = None) -> tuple[str, str]:
    """Resolve which key to embed with, and report where it came from.

    Order: the caller's own key, then the server key outside production, then
    an error asking for a key when PRODUCTION is on. Returns (key, source).
    """
    if api_key:
        return api_key, "user"

    server_key = get_gemini_key()
    if server_key and not config.is_production():
        return server_key, "server"

    raise RuntimeError(
        "A Gemini API key is required for embeddings. Turn on “Use my own API key” "
        "and enter your key."
    )


def log_embedding_key(source: str) -> None:
    """Record which key source an embedding request used. Never logs the key.

    Flushed explicitly: under a non-interactive stdout (Render's log capture)
    buffered output can sit unflushed and the line would never appear.
    """
    print(f"[embeddings] key source: {source}", flush=True)


def resolve_model(source: str, model: str | None = None, api_key: str | None = None,
                  base_url: str | None = None):
    """Build a chat model for the chosen answer source.

    When PRODUCTION is on, answers must run on the caller's own key: the server
    never spends its own quota on a visitor.
    """
    if config.is_production():
        if source != "My API key":
            raise RuntimeError(
                "This deployment only answers with your own API key. "
                "Turn on “Use my own API key” and enter your key."
            )
        return build_custom_model(model or "", api_key or "", base_url)

    if source == "Gemini":
        return build_gemini_model(model or GEMINI_DEFAULT_MODEL)
    # Ollama support is disabled; kept for re-enable.
    # if source == "Ollama":
    #     return build_ollama_model(model or OLLAMA_DEFAULT_MODEL)
    if source == "Hugging Face":
        return get_chat_model()
    if source == "My API key":
        return build_custom_model(model or "", api_key or "", base_url)
    raise RuntimeError(f"Unknown answer source: {source!r}")


def key_status() -> dict:
    """Report which providers are configured."""
    return {
        "gemini": bool(get_gemini_key()),
        "huggingface": bool(get_hf_token()),
        "production": config.is_production(),
        "embeddings": config.embeddings_mode(),
        # Ollama support is disabled; kept for re-enable.
        # "ollama_models": list_ollama_models(),
    }


def log_key_status() -> None:
    """Print which keys were found, without ever printing their values."""
    print(f"[config] .env path        : {ENV_PATH}", flush=True)
    print(f"[config] .env exists      : {ENV_PATH.exists()}", flush=True)
    print(f"[config] production       : {config.is_production()}", flush=True)
    print(f"[config] embeddings mode  : {config.embeddings_mode()}", flush=True)
    for label, names, getter in (
        ("GEMINI_API_KEY", GEMINI_ENV_NAMES, get_gemini_key),
        ("HUGGINGFACEHUB_API_TOKEN", TOKEN_ENV_NAMES, get_hf_token),
    ):
        value = getter()
        source = next((n for n in names if os.getenv(n)), "-")
        print(
            f"[config] {label:<24} loaded: {bool(value)}  (via {source}, len {len(value or '')})",
            flush=True,
        )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatHuggingFace:
    """Build (once) the default Hugging Face chat model.

    Note: the Hugging Face Inference Providers route needs an account with
    inference credits enabled — a token without them returns HTTP 402.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Add one of these to .env:\n"
            "    HUGGINGFACEHUB_API_TOKEN=hf_...\n"
            "    HF_TOKEN=hf_..."
        )

    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        provider=PROVIDER,
        huggingfacehub_api_token=token,
    )

    return ChatHuggingFace(llm=llm)
