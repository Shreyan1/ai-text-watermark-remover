"""The real semantic guard, cosine similarity over local sentence embeddings.

This is the piece that makes the gated pipeline usable on real text. The Jaccard
placeholder rewards *word overlap*, so it punishes exactly what a good rewrite
does (change the words, keep the meaning). Embeddings measure meaning, so a
faithful reword scores HIGH and a topic drift scores LOW, which is the signal
the gate actually needs.

Runs locally via Ollama (`all-minilm`, the all-MiniLM-L6-v2 sentence encoder,
384-dim). No new Python dependencies: stdlib HTTP + math.

Measured behaviour of the underlying encoder:
    paraphrase, different words  -> cos 0.650
    unrelated topic              -> cos -0.077

KNOWN BLIND SPOT, negation and factual inversion (measured, tests/harness/guard_eval.py):
    "X are computers"      vs "X are NOT computers"      -> 0.959
    "deployment succeeded" vs "deployment failed"        -> 0.898
    "revenue increased 40%" vs "revenue decreased 40%"   -> 0.776

Sentence embeddings encode topic, not truth value, so a rewrite that INVERTS a
claim still scores far above any usable floor. This guard therefore prevents
TOPIC DRIFT, not FACT CORRUPTION. Do not read a passing similarity as "the facts
survived". Closing this needs a different instrument: an entailment/NLI check, or
verification that `Point.constraints` (the must-keep numbers and names captured at
extraction) still appear in the output. Tracked as future work in RESULTS.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request

from .._ollama import DEFAULT_HOST, OllamaError
from ..core.interfaces import SemanticGuard
from ..core.registry import KIND_GUARD, register
from ..core.types import Document

_SENT = re.compile(r"(?<=[.!?])\s+")


def _chunk(text: str, max_chars: int = 1200) -> list[str]:
    """Split into embedding-sized chunks on sentence boundaries.

    all-MiniLM truncates at 512 tokens, so a long document embedded whole would
    silently lose its tail, and the guard would be comparing prefixes. Chunking
    and mean-pooling keeps the whole document in the comparison.
    """
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    chunks: list[str] = []
    cur = ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text[:max_chars]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@register(KIND_GUARD, "ollama-embed")
class OllamaEmbeddingGuard(SemanticGuard):
    """Meaning-preservation floor via local sentence embeddings."""

    name = "ollama-embed"

    def __init__(self, model: str = "all-minilm", host: str = DEFAULT_HOST,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}  # the loop re-embeds the source each retry

    def _embed_one(self, text: str) -> list[float]:
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps({"model": self.model, "input": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["embeddings"][0]
        except Exception as e:  # noqa: BLE001 - surface any transport/shape failure uniformly
            raise OllamaError(
                f"embedding failed ({self.model}); is it pulled? "
                f"`ollama pull {self.model}`, {e}"
            ) from e

    def embed(self, text: str) -> list[float]:
        """Mean-pooled embedding over the document's chunks, L2-normalised."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        vecs = [self._embed_one(c) for c in _chunk(text)]
        dim = len(vecs[0])
        pooled = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in pooled))
        if norm > 0:
            pooled = [x / norm for x in pooled]
        self._cache[key] = pooled
        return pooled

    def similarity(self, a: Document, b: Document) -> float:
        """Cosine similarity in 0..1. Negative cosine means unrelated, so it
        clamps to 0 rather than being rescaled, a rewrite about a different
        subject should read as 0 similarity, not 0.5."""
        if not a.text.strip() or not b.text.strip():
            return 0.0
        cos = _cosine(self.embed(a.text), self.embed(b.text))
        return max(0.0, min(1.0, cos))
