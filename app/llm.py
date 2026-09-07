import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PROVIDER = "together"


def get_hf_token() -> str | None:
    """Resolve the Hugging Face token.

    Priority order:
    1. os.environ                 (Streamlit Cloud injects secrets as env vars)
    2. st.secrets                 (Streamlit's explicit secrets API)
    3. .env via load_dotenv       (local CLI development)
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token

    try:
        import streamlit as st

        token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        if token:
            return token
    except Exception:
        pass

    load_dotenv()
    return os.getenv("HUGGINGFACEHUB_API_TOKEN")


@lru_cache(maxsize=1)
def get_chat_model() -> ChatHuggingFace:
    """Build (once) the chat model. Token is resolved at first use so it works
    on local CLI (.env) and on Streamlit Cloud (secrets) alike."""
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        provider=PROVIDER,
        huggingfacehub_api_token=get_hf_token(),
    )

    return ChatHuggingFace(llm=llm)