const $ = (id) => document.getElementById(id);

const state = {
  source: "Gemini",
  custom: false,
  ready: false,
  repo: "",
  status: {},
  backendDown: false,
};

const PRESETS = {
  Gemini: ["gemini-2.5-flash", "gemini-2.5-pro"],
  "Hugging Face": ["meta-llama/Llama-3.3-70B-Instruct"],
};

document.addEventListener("DOMContentLoaded", async () => {
  renderSources();
  bindEvents();
  await loadStatus();
});

async function loadStatus() {
  try {
    state.status = await apiFetch(API.status);
    state.backendDown = false;
  } catch {
    // The API itself is unreachable — that is not the same as "no key".
    state.status = {};
    state.backendDown = true;
  }
  state.source = defaultSource();
  renderSources();
  renderModelPicker();
  updateHint();
}

/** Keep the chosen source if it's still valid, otherwise default to Gemini. */
function defaultSource() {
  const s = state.status;
  if (state.source === "Gemini" || s.gemini || !s.huggingface) return "Gemini";
  return state.source;
}

function renderSources() {
  const s = state.status;
  const down = state.backendDown;
  const ownKeyOnly = !!s.production;
  const defs = [
    { name: "Gemini", ready: !!s.gemini },
    { name: "Hugging Face", ready: !!s.huggingface },
  ];
  const wrap = $("sources");
  wrap.innerHTML = "";
  for (const d of defs) {
    const el = document.createElement("div");
    let cls = "source";
    let tag;
    if (down) {
      cls += " unreachable";
      tag = "backend unreachable";
    } else if (ownKeyOnly) {
      // The server's own key is never used here, so a configured provider is
      // still not usable without the visitor's key.
      cls += " unconfigured";
      tag = "needs your key";
    } else if (d.ready) {
      cls += " ready";
      tag = "cloud";
    } else {
      cls += " unconfigured";
      tag = "no key";
    }
    if (!down && d.name === state.source && !state.custom) cls += " active";

    el.className = cls;
    el.innerHTML = `<span class="dot"></span>
                    <span class="name">${d.name}</span>
                    <span class="tag">${tag}</span>`;
    if (!down) el.onclick = () => selectSource(d.name);
    wrap.appendChild(el);
  }
}

function selectSource(name) {
  state.source = name;
  state.custom = false;
  $("customToggle").checked = false;
  renderSources();
  renderModelPicker();
  updateHint();
}

function renderModelPicker() {
  $("customFields").classList.toggle("hidden", !state.custom);
  $("modelRow").classList.toggle("hidden", state.custom);
  if (state.custom) return;

  const sel = $("model");
  const options = PRESETS[state.source] || [];
  sel.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o;
    sel.appendChild(opt);
  }
}

function updateHint() {
  const hint = $("sourceHint");
  const s = state.status;
  hint.className = "hint";

  if (state.backendDown) {
    hint.className = "hint warn";
    hint.textContent = "Backend unreachable — the API is not responding. Is the server running?";
    return;
  }

  if (state.custom) {
    hint.textContent = "Enter a base URL, model name and key. Works with any OpenAI-compatible API.";
    return;
  }

  if (s.production) {
    hint.className = "hint warn";
    hint.textContent = "This deployment answers only with your own API key. Turn on the switch above.";
    return;
  }

  if (state.source === "Gemini") {
    hint.textContent = s.gemini
      ? "Using GEMINI_API_KEY from your .env."
      : "Add GEMINI_API_KEY=... to .env to use Gemini.";
    if (!s.gemini) hint.className = "hint warn";
  } else {
    hint.textContent = s.huggingface
      ? "Needs an HF account with Inference Providers credits."
      : "Add HUGGINGFACEHUB_API_TOKEN=... to .env.";
    if (!s.huggingface) hint.className = "hint warn";
  }
}

function bindEvents() {
  $("k").oninput = (e) => { $("kVal").textContent = e.target.value; };
  $("loadBtn").onclick = loadRepo;
  $("clearBtn").onclick = clearChat;
  $("askForm").onsubmit = (e) => { e.preventDefault(); ask(); };
  $("customToggle").onchange = (e) => {
    state.custom = e.target.checked;
    renderSources();
    renderModelPicker();
    updateHint();
    if (state.custom) $("customModel").focus();
  };
}

const API_BASE = location.port === "5500" ? "http://127.0.0.1:5000/" : "";

const API = {
  status: API_BASE + "api/status",
  index: API_BASE + "api/index",
  ask: API_BASE + "api/ask",
};

/** POST/GET JSON, tolerating empty or non-JSON bodies instead of throwing a
 *  cryptic "Unexpected end of JSON input". Transport failures are tagged so
 *  the UI can tell "backend unreachable" apart from a server-side rejection. */
async function apiFetch(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    const err = new Error("Backend unreachable — could not reach the API.");
    err.unreachable = true;
    throw err;
  }

  const text = await res.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      const err = new Error(`Server returned a non-JSON response (HTTP ${res.status}).`);
      err.unreachable = true;
      throw err;
    }
  }
  if (!res.ok) throw new Error(data.error || `Request failed (HTTP ${res.status}).`);
  return data;
}

/** The caller's own key, sent with both indexing and answering. */
function userKeyPayload() {
  if (!state.custom) return {};
  return {
    api_key: $("apiKey").value.trim(),
    base_url: $("baseUrl").value.trim(),
    model: $("customModel").value.trim(),
  };
}

async function loadRepo() {
  const repo = $("repo").value.trim();
  if (!repo) return showRepoStatus("Enter a repository URL or path first.", "err");

  const btn = $("loadBtn");
  btn.disabled = true;
  showRepoStatus("Cloning and indexing… this can take a moment.", "");

  try {
    const data = await apiFetch(API.index, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo, ...userKeyPayload() }),
    });

    state.repo = repo;
    state.ready = true;
    showRepoStatus(`Indexed ${data.label} — ${data.files} files · ${data.chunks} chunks`, "ok");
    $("question").disabled = false;
    $("askBtn").disabled = false;
    $("question").focus();
    $("emptyState")?.remove();
  } catch (e) {
    // A 400 means the server is alive and refused the request; anything else
    // at the transport level means it could not be reached.
    if (e.unreachable) markBackendDown();
    showRepoStatus(e.message, "err");
  } finally {
    btn.disabled = false;
  }
}

/** Flip the UI into the "backend unreachable" state after a transport failure. */
function markBackendDown() {
  state.backendDown = true;
  renderSources();
  updateHint();
}

function showRepoStatus(msg, cls) {
  const el = $("repoStatus");
  el.textContent = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

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
    source: state.custom ? "My API key" : state.source,
    ...userKeyPayload(),
  };
  if (!state.custom) body.model = $("model").value;

  try {
    const data = await apiFetch(API.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    thinking.remove();
    addMessage("assistant", data.answer, data.sources);
  } catch (e) {
    thinking.remove();
    if (e.unreachable) markBackendDown();
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
