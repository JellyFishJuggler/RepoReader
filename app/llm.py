import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PROVIDER = "together"

# Any of these env/secret names is accepted.
TOKEN_ENV_NAMES = (
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_TOKEN",
    "TOGETHER_API_KEY",
)

# Local Ollama server (OpenAI-compatible endpoint).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Cheapest local model — smallest footprint, fastest tokens/sec.
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"

# Gemini via its OpenAI-compatible endpoint, so no extra dependency is needed.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

# The sources offered in the web UI.
MODEL_SOURCES = ("Gemini", "Ollama", "Hugging Face", "My API key")


def get_hf_token() -> str | None:
    """Resolve the Hugging Face provider token.

    Priority order:
    1. Environment variables        (Cloud injects secrets here; HF_TOKEN too)
    2. st.secrets                   (Streamlit's explicit secrets API)
    3. .env via load_dotenv         (local CLI development)

    Returns None if no token is configured anywhere.
    """
    for name in TOKEN_ENV_NAMES:
        token = os.getenv(name)
        if token:
            return token

    secrets = _read_streamlit_secrets()
    if secrets is not None:
        for name in TOKEN_ENV_NAMES:
            try:
                token = secrets.get(name) or secrets[name]
            except Exception:
                token = None
            if token:
                return token

    load_dotenv()
    for name in TOKEN_ENV_NAMES:
        token = os.getenv(name)
        if token:
            return token

    return None


def get_gemini_key() -> str | None:
    """Resolve the Gemini API key from env, st.secrets or .env."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        token = os.getenv(name)
        if token:
            return token

    secrets = _read_streamlit_secrets()
    if secrets is not None:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                token = secrets.get(name) or secrets[name]
            except Exception:
                token = None
            if token:
                return token

    load_dotenv()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        token = os.getenv(name)
        if token:
            return token

    return None


def _read_streamlit_secrets():
    """Return the st.secrets object when running inside Streamlit, else None."""
    try:
        import streamlit as st

        return st.secrets
    except Exception:
        return None


def list_ollama_models() -> list[str]:
    """Query the running Ollama server for available models.

    Returns [] when the server isn't reachable so callers can show a helpful
    message instead of crashing.
    """
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL.rsplit('/', 1)[0]}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


def build_chat_model(model: str, api_key: str, base_url: str | None = None, max_tokens: int | None = None):
    """Build a chat model over any OpenAI-compatible endpoint.

    Covers Gemini, Ollama and arbitrary providers — they all speak the
    /chat/completions protocol, so one client is enough.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        max_tokens=max_tokens,
        max_retries=2,
    )


def build_gemini_model(model: str = GEMINI_DEFAULT_MODEL):
    """Gemini via the .env / secrets key, over Google's OpenAI-compatible endpoint."""
    key = get_gemini_key()
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Add GEMINI_API_KEY=... to .env "
            "(or to Streamlit secrets)."
        )
    # Gemini's 2.5 models are reasoning models: they spend tokens thinking before
    # answering, so leave headroom rather than capping low.
    return build_chat_model(model, api_key=key, base_url=GEMINI_BASE_URL, max_tokens=4096)


def build_ollama_model(model: str = OLLAMA_DEFAULT_MODEL):
    """Local Ollama model. No API key required; a dummy key satisfies the client."""
    return build_chat_model(model, api_key="ollama", base_url=OLLAMA_BASE_URL, max_tokens=2048)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatHuggingFace:
    """Build (once) the default Hugging Face chat model. Token is resolved at
    first use so it works on local CLI (.env) and on Streamlit Cloud (secrets)
    alike.

    Raises a clear, actionable error when no token can be found instead of
    building a broken endpoint.

    Note: the Hugging Face Inference Providers route needs an account with
    inference credits enabled — a token without them returns HTTP 402.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. "
            "On Streamlit Cloud add it under Settings -> Secrets as:\n"
            '    HUGGINGFACEHUB_API_TOKEN = "hf_..."\n'
            "Locally, put it in .env or .streamlit/secrets.toml.\n"
            "Alternatively, pick Gemini, Ollama, or your own API key in the sidebar."
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
