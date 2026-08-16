"""Stage ③ — Regenerator adapters.

Rebuild prose from Meaning, steered by VoiceProfile toward high burstiness, high
perplexity, the author's vocabulary. This is the stage that produces the fresh,
un-watermarked token sequence.

HARD INVARIANT (the Self-Watermark Trap): the backing model MUST be local,
open-weight, and unwatermarked. Every adapter here declares `is_unwatermarked`
truthfully; the Pipeline refuses to run one that is False (unless explicitly
overridden). Routing this through a watermarked/hosted frontier model re-stamps a
new watermark and defeats the whole system.

STATUS: contract + an echo stub for wiring. Real adapters wrap a local runtime
(llama.cpp / vLLM / transformers) and land with the `[regenerate]` extra.
"""

from __future__ import annotations

from ..core.interfaces import Regenerator
from ..core.registry import KIND_REGENERATOR, register
from ..core.types import Document, Meaning, VoiceProfile


@register(KIND_REGENERATOR, "echo-stub")
class EchoStubRegenerator(Regenerator):
    """Wiring-only stub. Concatenates point intents; does NOT remove watermarks.

    Declared unwatermarked purely so the pipeline runs in tests. Never use for
    real removal — it performs no regeneration.
    """

    name = "echo-stub"
    is_unwatermarked = True

    def regenerate(
        self, meaning: Meaning, voice: VoiceProfile, aggressiveness: float = 0.5
    ) -> Document:
        text = " ".join(p.intent for p in meaning.points)
        return Document(text=text, meta={"regenerator": self.name, "stub": True})


@register(KIND_REGENERATOR, "local-llm")
class LocalLLMRegenerator(Regenerator):
    """Real regenerator over a LOCAL open-weight model.

    Contract for the eventual implementation:
      - load an unwatermarked open-weight model (Llama/Mistral/Qwen/…)
      - build a prompt from `meaning` + `voice` that demands varied sentence
        length, contractions, the author's vocabulary, and bans AI tells
      - map `aggressiveness` to temperature / restructuring latitude
      - set is_unwatermarked=True ONLY because the backing model truly is
    """

    name = "local-llm"
    is_unwatermarked = True  # true ONLY for an open-weight, unwatermarked backend

    def __init__(self, model: str = "mistral-7b-instruct", runtime: str = "llama.cpp") -> None:
        self.model = model
        self.runtime = runtime

    def regenerate(  # pragma: no cover - stub
        self, meaning: Meaning, voice: VoiceProfile, aggressiveness: float = 0.5
    ) -> Document:
        raise NotImplementedError(
            "LocalLLMRegenerator needs a local model runtime (see [regenerate] extra). "
            "This is where the fresh, un-watermarked token sequence is produced."
        )
