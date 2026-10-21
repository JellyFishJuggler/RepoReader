// ============ RepoReader frontend ============
// Talks to the Flask API in server.py. Keeps the selected source + fields in
// memory and renders the chat.

const $ = (id) => document.getElementById(id);

const state = {
  source: "Gemini",       // current answer source
  ready: false,           // is a repo indexed?
  repo: "",               // indexed repo source
  status: {},             // /api/status payload
};

// Which preset models to offer per source (overridden by live data when we have it).
const PRESETS = {
  Gemini: ["gemini-2.5-flash", "gemini-2.5-pro"],
};

// ============ boot ============
document.addEventListener("DOMContentLoaded", async () => {
  renderSources();
  bindEvents();
  await loadStatus();
});

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    state.status = await res.json();
  } catch (e) {
    state.status = { error: "Could not reach server." };
  }
  // Auto-select the first source that's actually ready.
  const s = state.status;
  if (s.gemini) state.source = "Gemini";
  else if (s.ollama_models && s.ollama_models.length) state.source = "Ollama";
  else state.source = "My API key";
  renderSources();
  renderModelPicker();
  updateHint();
}

// ============ sidebar rendering ============
function renderSources() {
  const s = state.status;
  const wrap = $("sources");
  const defs = [
    { name: "Gemini", tag: "cloud", ready: !!s.gemini },
    { name: "Ollama", tag: "local · free", ready: !!(s.ollama_models && s.ollama_models.length) },
    { name: "Hugging Face", tag: "cloud", ready: !!s.huggingface },
    { name: "My API key", tag: "any provider", ready: true },
  ];
  wrap.innerHTML = "";
  for (const d of defs) {
    const el = document.createElement("div");
    el.className = "source" + (d.ready ? " ready" : " disabled") +
                   (d.name === state.source ? " active" : "");
    el.innerHTML = `<span class="dot"></span>
                    <span class="name">${d.name}</span>
                    <span class="tag">${d.tag}</span>`;
    if (d.ready) el.onclick = () => selectSource(d.name);
    wrap.appendChild(el);
  }
}

function selectSource(name) {
  state.source = name;
  renderSources();
  renderModelPicker();
  updateHint();
}

function renderModelPicker() {
  const isCustom = state.source === "My API key";
  $("customFields").classList.toggle("hidden", !isCustom);
  $("modelRow").classList.toggle("hidden", isCustom);
  if (isCustom) return;

  const sel = $("model");
  let options = [];
  if (state.source === "Ollama") options = state.status.ollama_models || [];
  else if (state.source === "Gemini") options = PRESETS.Gemini;
  else options = ["meta-llama/Llama-3.3-70B-Instruct"]; // HF default

  sel.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o; opt.textContent = o;
    sel.appendChild(opt);
  }
  // Prefer the cheapest Ollama model if present.
  if (state.source === "Ollama" && options.includes("qwen2.5:3b")) sel.value = "qwen2.5:3b";
}

function updateHint() {
  const hint = $("sourceHint");
  const s = state.status;
  hint.className = "hint";
  if (state.source === "Gemini") {
    hint.textContent = s.gemini
      ? "Using GEMINI_API_KEY from your .env."
      : "Add GEMINI_API_KEY=... to .env to use Gemini.";
    if (!s.gemini) hint.className = "hint warn";
  } else if (state.source === "Ollama") {
    hint.textContent = (s.ollama_models && s.ollama_models.length)
      ? "Runs locally — no key, no cost."
      : "Ollama isn't running. Start it with `ollama serve`.";
    if (!(s.ollama_models && s.ollama_models.length)) hint.className = "hint warn";
  } else if (state.source === "Hugging Face") {
    hint.textContent = s.huggingface
      ? "Needs an HF account with Inference Providers credits."
      : "Add HUGGINGFACEHUB_API_TOKEN=... to .env.";
    if (!s.huggingface) hint.className = "hint warn";
  } else {
    hint.textContent = "Enter a base URL, model name and key. Works with any OpenAI-compatible API.";
  }
}

// ============ events ============
function bindEvents() {
  $("k").oninput = (e) => { $("kVal").textContent = e.target.value; };
  $("loadBtn").onclick = loadRepo;
  $("clearBtn").onclick = clearChat;
  $("askForm").onsubmit = (e) => { e.preventDefault(); ask(); };
}

async function loadRepo() {
  const repo = $("repo").value.trim();
  if (!repo) return showRepoStatus("Enter a repository URL or path first.", "err");

  const btn = $("loadBtn");
  btn.disabled = true;
  showRepoStatus("Cloning and indexing… this can take a moment.", "");

  try {
    const res = await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to index.");

    state.repo = repo;
    state.ready = true;
    showRepoStatus(`Indexed ${data.label} — ${data.files} files · ${data.chunks} chunks`, "ok");
    $("question").disabled = false;
    $("askBtn").disabled = false;
    $("question").focus();
    $("emptyState")?.remove();
  } catch (e) {
    showRepoStatus(e.message, "err");
  } finally {
    btn.disabled = false;
  }
}

function showRepoStatus(msg, cls) {
  const el = $("repoStatus");
  el.textContent = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

// ============ chat ============
function addMessage(role, text, sources) {
  $("emptyState")?.remove();
  const wrap = $("messages");
  const div = document.createElement("div");
  div.className = "msg " + role;

  const avatar = role === "user" ? "🧑" : "🤖";
  const who = role === "user" ? "You" : "RepoReader";

  const notFound = role === "assistant" && text.startsWith("Not found in repository");
  let sourcesHtml = "";
  if (sources && sources.length) {
    sourcesHtml = `<details class="sources-list"><summary>Sources (${sources.length})</summary>
      <ul>${sources.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul></details>`;
  }

  div.innerHTML = `<div class="avatar">${avatar}</div>
    <div class="body">
      <div class="who">${who}</div>
      <div class="text${notFound ? " notfound" : ""}">${escapeHtml(text)}</div>
      ${sourcesHtml}
    </div>`;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return div;
}

function addThinking() {
  const wrap = $("messages");
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.innerHTML = `<div class="avatar">🤖</div>
    <div class="body"><div class="who">RepoReader</div>
    <div class="thinking">Retrieving and answering<span class="bounce"></span><span class="bounce"></span><span class="bounce"></span></div></div>`;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return div;
}

async function ask() {
  const question = $("question").value.trim();
  if (!question || !state.ready) return;

  addMessage("user", question);
  $("question").value = "";
  $("askBtn").disabled = true;

  const thinking = addThinking();

  const body = {
    repo: state.repo,
    question,
    k: parseInt($("k").value, 10),
    source: state.source,
  };
  if (state.source === "My API key") {
    body.base_url = $("baseUrl").value.trim();
    body.model = $("customModel").value.trim();
    body.api_key = $("apiKey").value.trim();
  } else {
    body.model = $("model").value;
  }

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    thinking.remove();
    if (!res.ok) throw new Error(data.error || "Failed to answer.");
    addMessage("assistant", data.answer, data.sources);
  } catch (e) {
    thinking.remove();
    addError(e.message);
  } finally {
    $("askBtn").disabled = false;
    $("question").focus();
  }
}

function addError(msg) {
  const wrap = $("messages");
  const div = document.createElement("div");
  div.className = "err-banner";
  div.textContent = msg;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
}

function clearChat() {
  $("messages").innerHTML = `<div class="empty" id="emptyState">
      <div class="empty-icon">💬</div>
      <h3>Load a repository to start</h3>
      <p>Answers are grounded strictly in the repo — if it isn't in there,
         you'll get <strong>“Not found in repository.”</strong></p>
    </div>`;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
