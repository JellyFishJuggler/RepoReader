import streamlit as st
from dotenv import load_dotenv
from git import Repo
from pathlib import Path
from tempfile import TemporaryDirectory

from app.loader import is_remote
from app.llm import (
    GEMINI_DEFAULT_MODEL,
    MODEL_SOURCES,
    OLLAMA_DEFAULT_MODEL,
    build_gemini_model,
    build_ollama_model,
    build_chat_model,
    get_gemini_key,
    get_hf_token,
    list_ollama_models,
)
from app.rag import NOT_FOUND, ask, build_vectorstore

load_dotenv()

st.set_page_config(page_title="RepoReader", page_icon="📚")


@st.cache_resource(show_spinner="Cloning and indexing repository...")
def get_reader(repo_source: str):
    """Build the vector store once per repository and cache it."""
    if is_remote(repo_source):
        with TemporaryDirectory() as tmp:
            repo = Repo.clone_from(repo_source, tmp)
            repo_path = Path(repo.working_tree_dir)
            store, file_count, chunk_count = build_vectorstore(repo_path)
    else:
        store, file_count, chunk_count = build_vectorstore(Path(repo_source))
    return store, file_count, chunk_count


@st.cache_resource(show_spinner=False)
def cached_model(kind: str, model: str, api_key: str = "", base_url: str = ""):
    """Build each chat model once and reuse it across reruns.

    Keyed on every input that affects the client so editing the key or base
    URL rebuilds the model instead of returning a stale one.
    """
    if kind == "Gemini":
        return build_gemini_model(model)
    if kind == "Ollama":
        return build_ollama_model(model)
    return build_chat_model(model, api_key=api_key, base_url=base_url or None)


def repo_label(source: str) -> str:
    if is_remote(source):
        return source.rstrip("/").split("/")[-1].removesuffix(".git")
    return Path(source).name


st.title("📚 RepoReader")
st.caption(
    "Ask questions about any GitHub repository. Answers are grounded strictly in "
    "the repository — it says **Not found in repository.** when the answer isn't there."
)

with st.sidebar:
    st.header("Repository")
    repo_source = st.text_input(
        "GitHub URL or local path",
        value="https://github.com/JellyFishJuggler/diabetes-slr.git",
        help="A `.git` URL or a path to a folder with source code.",
    )
    load_repo_btn = st.button("Load repository", type="primary", width="stretch")
    k = st.slider("Context chunks (k)", 1, 10, 4,
                  help="How many retrieved chunks are fed to the model.")
    st.divider()

    st.header("Model")
    source = st.radio("Answer with", MODEL_SOURCES, index=0, horizontal=True,
                      help="Pick where answers come from.")

    user_chat_model = None
    model_error = None

    if source == "Gemini":
        if get_gemini_key():
            model_name = st.selectbox("Model", [GEMINI_DEFAULT_MODEL, "gemini-2.5-pro"])
            user_chat_model = cached_model("Gemini", model_name)
            st.success(f"Using **{model_name}** (key from `.env`)")
        else:
            model_error = "Add `GEMINI_API_KEY=...` to `.env` to use Gemini."

    elif source == "Ollama":
        available = list_ollama_models()
        if available:
            default_ix = available.index(OLLAMA_DEFAULT_MODEL) if OLLAMA_DEFAULT_MODEL in available else 0
            model_name = st.selectbox("Model", available, index=default_ix,
                                      help="qwen2.5:3b is the cheapest — smallest and fastest.")
            user_chat_model = cached_model("Ollama", model_name)
            st.success(f"Running **{model_name}** locally (no key needed)")
        else:
            model_error = ("Couldn't reach Ollama at `localhost:11434`. "
                           "Start it with `ollama serve` and reload.")

    elif source == "Hugging Face":
        if get_hf_token():
            st.info("Using the token from `.env`. Note: this needs an HF account "
                    "with Inference Providers credits — otherwise it returns 402.")
        else:
            model_error = "Add `HUGGINGFACEHUB_API_TOKEN=...` to `.env` to use Hugging Face."

    else:  # My API key
        st.session_state["custom_base"] = st.text_input(
            "Base URL", placeholder="https://api.openai.com/v1",
            help="Any OpenAI-compatible endpoint. Leave blank for OpenAI itself.")
        st.session_state["custom_model"] = st.text_input(
            "Model name", placeholder="gpt-4o-mini")
        st.session_state["custom_key"] = st.text_input(
            "API key", type="password")
        if st.session_state["custom_key"] and st.session_state["custom_model"]:
            user_chat_model = cached_model(
                "custom",
                st.session_state["custom_model"],
                api_key=st.session_state["custom_key"],
                base_url=st.session_state.get("custom_base") or "",
            )
            st.success("Custom model ready")
        else:
            model_error = "Enter your model name and API key to activate it."

    if model_error:
        st.warning(model_error)
    st.caption("Answers use `app/rag.py` — the same pipeline as the CLI.")

    if st.session_state.get("reader") is not None:
        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []

if load_repo_btn or st.session_state.get("reader") is None:
    try:
        store, file_count, chunk_count = get_reader(repo_source)
        st.session_state.reader = store
        st.session_state.file_count = file_count
        st.session_state.chunk_count = chunk_count
        st.session_state.repo_label = repo_label(repo_source)
        st.session_state.messages = []
    except Exception as exc:
        st.sidebar.error(f"Could not load repository: {exc}")

reader = st.session_state.get("reader")
if reader is None:
    st.info("Enter a repository in the sidebar and click **Load repository** to begin.")
    st.stop()

st.sidebar.success(
    f"Indexed **{st.session_state.repo_label}**\n\n"
    f"`{st.session_state.file_count}` files · `{st.session_state.chunk_count}` chunks"
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                st.markdown("\n".join(f"- `{src}`" for src in message["sources"]))

question = st.chat_input("Ask something about the repository...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            try:
                # HF falls back to the default token-based model; the other
                # sources pass an explicit user_chat_model.
                answer, sources = ask(
                    st.session_state.reader, question, k=k,
                    chat_model=user_chat_model if source != "Hugging Face" else None,
                )
            except Exception as exc:
                st.error(f"Something went wrong while answering:\n\n{exc}")
                st.stop()
        st.markdown(answer if answer else NOT_FOUND)
        with st.expander("Sources", expanded=False):
            st.markdown("\n".join(f"- `{src}`" for src in sources) if sources else "_none_")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
