import streamlit as st
from dotenv import load_dotenv
from git import Repo
from pathlib import Path
from tempfile import TemporaryDirectory

from app.loader import load_repo
from app.llm import get_hf_token
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


def is_remote(source: str) -> bool:
    return source.startswith(("http://", "https://", "git@"))


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
    load_repo_btn = st.button("Load repository", type="primary", use_container_width=True)
    k = st.slider("Context chunks (k)", 1, 10, 4, help="How many retrieved chunks are fed to the model.")
    st.divider()

    if get_hf_token():
        st.sidebar.success("HF token configured")
    else:
        st.sidebar.error(
            "**HF token not found.**\n\n"
            "On Streamlit Cloud add it under **Settings → Secrets** as "
            "`HUGGINGFACEHUB_API_TOKEN = \"hf_...\"`. "
            "Locally, set it in `.env` or `.streamlit/secrets.toml`."
        )
    st.caption("Answers use `app/rag.py` — the same pipeline as the CLI.")

    if st.session_state.get("reader") is not None:
        if st.button("Clear chat", use_container_width=True):
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
                answer, sources = ask(st.session_state.reader, question, k=k)
            except Exception as exc:
                st.error(f"Something went wrong while answering:\n\n{exc}")
                st.stop()
        st.markdown(answer if answer else NOT_FOUND)
        if sources:
            with st.expander("Sources"):
                st.markdown("\n".join(f"- `{src}`" for src in sources))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )