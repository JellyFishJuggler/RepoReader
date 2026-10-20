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

# Presets for user-supplied (bring-your-own-key) models in the web UI.
# Everything here is OpenAI-compatible: ChatOpenAI + base_url covers all of them.
PROVIDER_PRESETS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "Together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "Hugging Face": {
        "base_url": "https://api-inference.huggingface.co/models",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "default_model": "",
    },
}


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


def _read_streamlit_secrets():
    """Return the st.secrets object when running inside Streamlit, else None."""
    try:
        import streamlit as st

        return st.secrets
    except Exception:
        return None


def build_user_chat_model(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
):
    """Build a chat model from a user-supplied key (bring-your-own-key).

    Uses ChatOpenAI, so any OpenAI-compatible endpoint works — the presets in
    PROVIDER_PRESETS just pre-fill the right base_url and model name.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatHuggingFace:
    """Build (once) the default chat model. Token is resolved at first use so it
    works on local CLI (.env) and on Streamlit Cloud (secrets) alike.

    Raises a clear, actionable error when no token can be found instead of
    building a broken endpoint.
    """
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. "
            "On Streamlit Cloud add it under Settings -> Secrets as:\n"
            '    HUGGINGFACEHUB_API_TOKEN = "hf_..."\n'
            "Locally, put it in .env or .streamlit/secrets.toml.\n"
            "Alternatively, add your own API key in the web UI sidebar."
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
