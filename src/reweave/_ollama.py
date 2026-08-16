"""Minimal Ollama HTTP client, stdlib only, no `ollama` package needed.

Kept private (`_ollama`) because it is an edge detail: a local model runtime the
adapters happen to use. The core never imports it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


class WatermarkRiskError(ValueError):
    """A chosen model could re-stamp a watermark or leak the source text.

    Subclasses ValueError so callers that only knew about the old `"cloud"`
    ValueError guard keep catching it.
    """


#: Local open-weight model families with NO proactive token watermark in their
#: published weights. This is the load-bearing safety claim of the whole project,
#: so it is grounded rather than assumed:
#:
#:   * Statistical "green-list" watermarking (Kirchenbauer et al., 2023) and
#:     SynthID-Text (Dathathri et al., Nature 634, 2024; google-deepmind/
#:     synthid-text) are DECODE-TIME logits processors. They are applied by the
#:     serving stack, not baked into the weights.
#:   * The open weights you pull with Ollama contain no such processor. Verified
#:     by independent reverse-engineering of Gemma running in-browser: the
#:     watermark lives in Google's service layer; local generation is unmarked.
#:
#: So the watermark trap bites only HOSTED models. Any family below, run locally,
#: emits text carrying no proactive watermark. Deliberately conservative: a model
#: is refused unless it is on this list (or the caller opts in), rather than
#: allowed unless known-bad. Reasoning models (qwen3, deepseek-r1) are omitted on
#: purpose, not for watermarking but because they leak a `thinking` channel.
_WATERMARK_SAFE = (
    "gemma3", "gemma2", "gemma",
    "llama3.2", "llama3.1", "llama3", "llama2", "llama",
    "qwen2.5", "qwen2",
    "mistral", "mixtral",
    "phi4", "phi3",
)


def assert_local(model: str) -> None:
    """Refuse a hosted model: it sees the source text and may carry SynthID."""
    if "cloud" in model.lower():
        raise WatermarkRiskError(
            f"refusing model {model!r}: a hosted/cloud model receives your source "
            "text and may apply a decode-time watermark (SynthID-Text). Use a "
            "local open-weight model."
        )


def assert_watermark_safe(model: str, *, allow_unlisted: bool = False) -> None:
    """Gate the model that EMITS the final text (Stage ③ regeneration).

    A hosted model is always refused. A local model off the verified allowlist is
    refused unless the caller explicitly accepts the risk, because a rewrite is
    only watermark-free if the model that produced it is.
    """
    assert_local(model)
    if allow_unlisted:
        return
    family = model.lower().split(":", 1)[0]
    if not any(family.startswith(f) for f in _WATERMARK_SAFE):
        raise WatermarkRiskError(
            f"refusing model {model!r}: not on the verified-unwatermarked "
            f"allowlist ({', '.join(_WATERMARK_SAFE)}). If you have confirmed this "
            "local model applies no decode-time watermark, pass allow_unlisted=True "
            "(CLI: --allow-model)."
        )


def generate(prompt: str, model: str = "llama3.2:1b", *, host: str = DEFAULT_HOST,
             temperature: float = 0.8, num_predict: int = 512,
             system: str | None = None, timeout: float = 180.0,
             think: bool | None = None) -> str:
    """Generate text. Returns the response body, with a caveat worth knowing:

    reasoning models (qwen3, deepseek-r1, …) put their chain of thought in a
    separate `thinking` field and leave `response` EMPTY until reasoning ends.
    Cap `num_predict` too low and you get an empty string rather than an error —
    a silent failure that reads like a model with nothing to say. We therefore
    surface `thinking` in the error when `response` comes back empty, and accept
    `think=False` to switch reasoning off where the runtime supports it.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if system:
        payload["system"] = system
    if think is not None:
        payload["think"] = think
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        raise OllamaError(f"Ollama generate failed ({model}): {e}") from e

    text = (body.get("response") or "").strip()
    if not text and (body.get("thinking") or "").strip():
        raise OllamaError(
            f"{model} returned only reasoning and no answer (done_reason="
            f"{body.get('done_reason')!r}). It is a reasoning model: raise "
            "num_predict, or pass think=False."
        )
    return text


def is_up(host: str = DEFAULT_HOST, timeout: float = 5.0) -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False
