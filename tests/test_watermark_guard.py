"""The mandated model must be watermark-safe, enforced at construction.

The project's whole promise is that regeneration emits UNMARKED text. That only
holds if the model producing it carries no proactive watermark. So the choice of
model is a safety boundary, not a preference, and it is guarded where the object
is built rather than trusted to the caller. These tests pin the boundary:

  * a hosted ("cloud") model is always refused (it also sees the source text);
  * a local model off the verified-unwatermarked allowlist is refused by default;
  * the refusal is opt-out-able for a model the caller has vetted themselves;
  * a judge (NLI) enforces only the hosted refusal, since a label cannot carry a
    watermark into the user's prose.

No Ollama is needed: every check happens in __init__, before any network call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reweave._ollama import WatermarkRiskError  # noqa: E402
from reweave.regenerate.ollama_regenerator import OllamaRegenerator  # noqa: E402
from reweave.verify.ollama_nli import OllamaNLIBackend  # noqa: E402


def test_regenerator_defaults_to_an_allowlisted_local_model():
    assert OllamaRegenerator().model == "gemma3:4b"


def test_regenerator_accepts_other_allowlisted_families():
    for m in ("llama3.2:1b", "llama3.1:8b", "mistral:7b", "gemma2:2b", "phi4"):
        assert OllamaRegenerator(model=m).model == m


def test_regenerator_refuses_hosted_model():
    for m in ("gemma3-cloud", "gpt-oss:120b-cloud", "some-cloud-model"):
        with pytest.raises(WatermarkRiskError):
            OllamaRegenerator(model=m)


def test_regenerator_refuses_unlisted_model_by_default():
    # qwen3 is a reasoning model we deliberately keep off the allowlist.
    with pytest.raises(WatermarkRiskError):
        OllamaRegenerator(model="qwen3:4b")


def test_regenerator_allows_unlisted_model_on_explicit_opt_in():
    assert OllamaRegenerator(model="qwen3:4b", allow_unlisted=True).model == "qwen3:4b"


def test_watermark_risk_is_a_valueerror():
    # Callers that only knew the old `raise ValueError("cloud")` guard still catch.
    with pytest.raises(ValueError):
        OllamaRegenerator(model="anything-cloud")


def test_nli_refuses_hosted_but_not_unlisted():
    # A judge outputs a label, so the allowlist does not gate it; hosted is still
    # refused because the judge would receive the source text.
    with pytest.raises(WatermarkRiskError):
        OllamaNLIBackend(model="x-cloud")
    assert OllamaNLIBackend(model="qwen3:4b").model == "qwen3:4b"
