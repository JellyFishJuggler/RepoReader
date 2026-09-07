# 📚 RepoReader — Ask Questions About Any GitHub Repository

A Retrieval-Augmented Generation (RAG) CLI that clones any GitHub repository,
indexes its code and docs, and lets you **ask questions in plain English**.
Answers are grounded strictly in the repository (no hallucination — it says
`Not found in repository.` when the answer isn't there) and every answer lists
the source files it was built from.

Built with LangChain, Hugging Face embeddings, and the Llama-3.3-70B chat model.

---

## Features

- **Any repository** — pass a GitHub URL or a local path.
- **Notebook-aware** — `.ipynb` files are parsed cell-by-cell instead of embedding
  raw JSON garbage (base64 images, cell ids, rendered HTML).
- **Hallucination guardrail** — the model is instructed to answer *only* from the
  retrieved context and must reply `Not found in repository.` otherwise.
- **Source attribution** — every answer is followed by a `Sources:` list of the
  file names it came from.
- **Interactive or one-shot** — REPL for follow-up questions, or `-q` for a
  single answer (great for scripting).
- **Web UI** — a Streamlit chat interface (`streamlit_app.py`) for the same
  pipeline with source attribution and a context `k` slider.
- **Tiny & fast index** — overlapping 1000-char chunks tagged with their source
  file name; embedded locally with `all-MiniLM-L6-v2` (384-dim).

---

## Architecture

```
                    ┌────────────────────────────────────────────┐
 GitHub URL ───────▶│  main.py (CLI / interactive loop)          │
                    └─────────────────────┬──────────────────────┘
                                          │
                    ┌─────────────────────▼──────────────────────┐
                    │  app/loader.py      clone & load files     │
                    │  app/chunking.py    split into chunks      │
│  app/vectorstore.py embed + index (LangChain       │
│                     in-memory vector store)         │
                    └─────────────────────┬──────────────────────┘
                                          │ top-k similar chunks
                    ┌─────────────────────▼──────────────────────┐
                    │  app/rag.py         build prompt, call LLM │
                    │  app/llm.py         Llama-3.3-70B via HF   │
                    │  app/prompt.py      guardrail system prompt│
                    └─────────────────────┬──────────────────────┘
                                          │
                                          ▼
                                   Answer + Sources
```

### Module breakdown

| File | Responsibility |
|------|----------------|
| `main.py` | CLI entry point — repo handling, interactive loop, output formatting |
| `app/loader.py` | walks the repo, filters files, extracts notebook cells |
| `app/chunking.py` | splits documents into overlapping, filename-tagged chunks |
| `app/embedding.py` | local sentence embedding model (`all-MiniLM-L6-v2`) |
| `app/vectorstore.py` | embeds chunks once, returns the in-memory vector store |
| `app/rag.py` | retrieval + generation glue, source attribution |
| `app/prompt.py` | the "answer only from context" system prompt |
| `app/llm.py` | `HuggingFaceEndpoint` + `ChatHuggingFace` config |

---

## How it works

1. **Load** — clone the repository (or open a local folder) and read every
   supported file (`*.py`, `*.js`, `*.ts`, `*.md`, `*.json`, `*.yaml`,
   `*.ipynb`, …). Jupyter notebooks are parsed cell-by-cell so only the real
   markdown/code text is kept.
2. **Chunk** — split each file into 1000-character chunks with a 200-character
   overlap so nothing meaningful is lost at boundaries. Each chunk is prefixed
   with `[filename: <name>]` so queries that mention a file's name match it.
3. **Embed** — each chunk is embedded locally (no API cost) into a 384-dim
   vector, once per run, in a fast in-memory vector store.
4. **Retrieve** — the user's question is embedded and the top-k most similar
   chunks are pulled back.
5. **Generate** — the chunks are packed into a system prompt that *forbids* the
   model from using outside knowledge. The model answers, and the app attaches
   the source file list deterministically.

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Hugging Face token
cp .env.example .env
#   then edit .env and set your token:
#   HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxx
```

* Get a token at https://huggingface.co/settings/tokens
* The chat model runs through [HF Inference Providers](https://huggingface.co/docs/inference-providers)
  using the **Together** provider, so a free/pro subscription-backed token with
  Together enabled works out of the box.

---

## Usage

**Interactive session** (ask as many questions as you want, `exit` to quit):

```bash
python main.py -r https://github.com/JellyFishJuggler/diabetes-slr.git
```

**One-shot question** (also scripts well):

```bash
python main.py -r https://github.com/JellyFishJuggler/diabetes-slr.git \
               -q "what is the model equation?"
```

**Local folder** instead of a URL:

```bash
python main.py -r ../my-codebase -q "how does the auth middleware work?"
```

**Web UI (Streamlit)**:

```bash
streamlit run streamlit_app.py
```

Enter a repository in the sidebar, click **Load repository**, then chat with the
codebase. Every answer shows its `Sources:` in an expandable panel, and the
`k` slider controls how much context reaches the model.

| Flag | Description | Default |
|------|-------------|---------|
| `-r, --repo` | GitHub URL or local path (prompts interactively if omitted) | — |
| `-q, --query` | Ask a single question, then exit | interactive mode |
| `-k, --k` | Number of context chunks to retrieve | `4` |

---

## Example questions

Run against `https://github.com/JellyFishJuggler/diabetes-slr.git`:

```
> what is the model equation?

Answer: y = 998.578 × BMI + 152.003

Sources:
- README.md
- diabetes.ipynb
```

```
> what libraries are used?

Answer: The libraries used are:
1. pandas
2. numpy
3. matplotlib
4. scikit-learn
5. jupyter

Sources:
- diabetes.ipynb
- requiremenrs.txt
```

**Failure case (guardrail)** — the question is *not* in the repo:

```
> what is the capital of France?

Answer: Not found in repository.

Sources:
- README.md
- diabetes.ipynb
- requiremenrs.txt
```

---

## Tech stack

| Layer | Tool |
|-------|------|
| Orchestration | LangChain (`langchain-huggingface`, `langchain-text-splitters`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, purely local) |
| Vector store | LangChain `InMemoryVectorStore` |
| LLM | `meta-llama/Llama-3.3-70B-Instruct` via HF Inference Providers (Together) |
| Repo access | GitPython |
| Web UI | Streamlit |

---

## Project structure

```
.
├── main.py              # CLI entry point
├── streamlit_app.py     # Streamlit chat UI
├── requirements.txt
├── .env.example         # copy to .env and add your HF token
├── .gitignore
└── app/
    ├── __init__.py
    ├── loader.py        # file discovery + notebook parsing
    ├── chunking.py      # chunking + filename tags
    ├── embedding.py     # local embedding model
    ├── vectorstore.py   # in-memory index + retrieval
    ├── rag.py           # retrieval + generation glue
    ├── prompt.py        # guardrail system prompt
    └── llm.py           # chat model config
```

---

## Limitations

- **Embeddings are re-computed every run** — the index is in-memory and rebuilt
  on each launch. For large repos, starting up takes a bit and a persistent
  store (e.g. Chroma) would be the next step.
- **Repository size** — very large repos generate many chunks, which slows the
  initial index and can dilute retrieval. `-k` tunes how much context is used.
- **LLM cost** — answers go through a hosted 70B model on Together; light usage
  is fine, heavy use may incur provider costs.
- **Notebook outputs are skipped by design** — charts, printed results and
  images aren't indexed, only cell code/markdown. Good for retrieval signal,
  but you won't be able to "ask" about a plot.
- **Top-k file bias** — with small repos and `k=4`, results can include all
  files even when only one is relevant (as seen in the examples above).
- **Dense-only retrieval misses table-heavy content** — retrieval uses pure
  semantic (cosine) similarity with no keyword layer or reranking, and
  markdown tables embed poorly. Queries that name exact metric words
  ("accuracy", "f1") find the metrics table easily, but abstract phrasings
  ("how is the model performing?") can rank the right chunk 10th+ and miss
  the top-`k`. The guardrail then correctly reports `Not found in
  repository.` — the model only answers from whatever chunks were retrieved.
  Observed on `JellyFishJuggler/upi-finance-detector`: the metrics chunk
  ranked #11 for "how is the model performing?" but #1 once reranked with a
  cross-encoder. A reranking step (or hybrid keyword+dense search) is the
  planned fix.

---

## Roadmap

- [ ] Persistent vector store (Chroma) — no more re-embedding per run
- [x] Streamlit UI for a chat web interface
- [ ] Cross-encoder reranker (retrieve 20 → rerank → top-k) to fix table /
      phrase-sensitive retrieval recall
- [ ] `--notebook-outputs` flag to optionally index cell outputs
- [ ] Follow-up / conversational memory across questions