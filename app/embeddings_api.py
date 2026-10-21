"""Gemini embeddings over plain REST, with no torch dependency.

Implements the langchain Embeddings interface so it drops into
InMemoryVectorStore unchanged.
"""

import json
import time
import urllib.error
import urllib.request

from langchain_core.embeddings import Embeddings

from app import config

_RETRY_STATUS = {429, 500, 502, 503, 504}


class GeminiEmbeddings(Embeddings):
    """Embed text with the Gemini API using the caller's own key.

    The key is sent in the x-goog-api-key header so it never appears in a URL
    or an access log. Documents and queries use different task types, which
    measurably improves retrieval.
    """

    def __init__(self, api_key: str, model: str | None = None, dims: int | None = None):
        if not api_key:
            raise RuntimeError("A Gemini API key is required for embeddings.")
        self._api_key = api_key
        self._model = model or config.GEMINI_EMBED_MODEL
        self._dims = dims or config.GEMINI_EMBED_DIMS

    def __repr__(self) -> str:
        # Never let the key reach a log or a traceback.
        return f"GeminiEmbeddings(model={self._model!r}, dims={self._dims})"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of chunks, batched to stay within request limits."""
        out: list[list[float]] = []
        size = config.EMBED_BATCH_SIZE
        for start in range(0, len(texts), size):
            batch = texts[start:start + size]
            out.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Embed a single question using the query task type."""
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        requests = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                "outputDimensionality": self._dims,
            }
            for text in texts
        ]
        data = self._post(f"models/{self._model}:batchEmbedContents", {"requests": requests})
        return [item["values"] for item in data["embeddings"]]

    def _post(self, path: str, payload: dict) -> dict:
        """POST with retry and exponential backoff on rate limits and 5xx."""
        url = f"{config.GEMINI_API_BASE}/{path}"
        body = json.dumps(payload).encode()
        delay = 1.0

        for attempt in range(config.EMBED_MAX_RETRIES):
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=config.EMBED_TIMEOUT_SECONDS) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRY_STATUS and attempt < config.EMBED_MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if exc.code == 401 or exc.code == 403:
                    raise RuntimeError("Gemini rejected the API key for embeddings.") from None
                if exc.code == 429:
                    raise RuntimeError(
                        "Gemini rate limit hit while embedding. Try a smaller repository."
                    ) from None
                detail = exc.read()[:300].decode(errors="ignore")
                raise RuntimeError(f"Gemini embeddings failed (HTTP {exc.code}): {detail}") from None
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < config.EMBED_MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Could not reach Gemini embeddings: {exc}") from None

        raise RuntimeError("Gemini embeddings failed after retries.")
