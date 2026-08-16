"""Stage ③, the real regenerator, over a LOCAL open-weight model via Ollama.

This is the production shape of Stage ③: rebuild prose from Meaning, steered by
VoiceProfile toward burstiness, contractions, the author's vocabulary, away from
AI tells. The fresh token sequence it emits is what severs any watermark.

WHY THIS IS SAFE (the Self-Watermark Trap, ARCHITECTURE.md §1):
`llama3.2:1b` run locally is open-weight and unwatermarked, so `is_unwatermarked`
is truthfully True. We deliberately do NOT default to a hosted Google model
(e.g. `gemma:*-cloud`): a hosted frontier model may carry SynthID, which would
re-stamp a fresh watermark and defeat the pipeline. Point this at a local
open-weight model only.
"""

from __future__ import annotations

from .._ollama import generate
from ..core.interfaces import Regenerator
from ..core.registry import KIND_REGENERATOR, register
from ..core.types import Document, Meaning, VoiceProfile

_SYSTEM = (
    "You rewrite text so it reads like a specific human wrote it, not an AI. "
    "Rules: vary sentence length hard, mix very short sentences with long ones. "
    "Use contractions. Prefer plain words. Never use: delve, tapestry, seamless, "
    "leverage, robust, ecosystem, testament, landscape, navigate, underscore. "
    "No rule-of-three lists, no tidy summary sentence at the end. "
    "Keep every fact, name, and number, and never reverse a statement, if the "
    "source says something IS or DID, your rewrite must not say it IS NOT or "
    "DID NOT. Output only the rewritten text."
)


@register(KIND_REGENERATOR, "ollama")
class OllamaRegenerator(Regenerator):
    """Local, open-weight regeneration. The real Stage ③."""

    name = "ollama"
    is_unwatermarked = True  # TRUE only because llama3.2:1b is local + open-weight

    def __init__(self, model: str = "llama3.2:1b", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host
        if "cloud" in model.lower():
            # Guard the invariant at construction: a hosted model may re-watermark.
            raise ValueError(
                f"refusing model {model!r}: a hosted/cloud model may carry its own "
                "watermark (Self-Watermark Trap). Use a local open-weight model."
            )

    def _prompt(self, meaning: Meaning, voice: VoiceProfile, aggressiveness: float) -> str:
        points = "\n".join(f"- {p.intent}" for p in meaning.points)
        extra = []
        if voice.vocabulary:
            extra.append("Favour these words where natural: "
                         + ", ".join(sorted(voice.vocabulary)[:20]) + ".")
        if voice.banned_terms:
            extra.append("Never use: " + ", ".join(sorted(voice.banned_terms)[:20]) + ".")
        if aggressiveness > 0.6:
            extra.append("Restructure freely, do not mirror the original sentence order.")

        # Pin the hard facts explicitly. Prompting is not a guarantee, the fact
        # gate still verifies, but telling the model beats only catching it.
        keep_nums = meaning.meta.get("must_keep_numerals") or []
        keep_ents = meaning.meta.get("must_keep_entities") or []
        if keep_nums:
            extra.append("These numbers must appear unchanged: " + ", ".join(keep_nums) + ".")
        if keep_ents:
            extra.append("These names must appear: " + ", ".join(keep_ents[:20]) + ".")
        return (
            "Rewrite the following points into natural human prose.\n"
            f"{chr(10).join(extra)}\n\nPoints:\n{points}\n\nRewritten:"
        )

    def regenerate(
        self, meaning: Meaning, voice: VoiceProfile, aggressiveness: float = 0.5
    ) -> Document:
        text = generate(
            self._prompt(meaning, voice, aggressiveness),
            model=self.model, host=self.host,
            temperature=min(1.2, 0.6 + 0.6 * aggressiveness),
            num_predict=640,
            system=_SYSTEM,
        )
        return Document(text=text, meta={"regenerator": self.name, "model": self.model,
                                         "aggressiveness": aggressiveness})
