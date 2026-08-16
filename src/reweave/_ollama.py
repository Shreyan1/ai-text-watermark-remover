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
