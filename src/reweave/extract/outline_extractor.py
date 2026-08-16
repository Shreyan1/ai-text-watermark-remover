"""Stage ② — Extractor adapters.

Reduce a document to its Meaning (ordered Points), discarding surface form. This
severing of the token sequence is what breaks the watermark substrate
(Invariant I4). Extractors are model-backed but model-agnostic behind the ABC.

STATUS: contract + a trivial fallback. The real extractor prompts a local model
to outline the text into atomic points with must-keep constraints (numbers,
names). That backend lands with the `[extract]`/`[regenerate]` extras.
"""

from __future__ import annotations

import re

from ..core.interfaces import Extractor
from ..core.registry import KIND_EXTRACTOR, register
from ..core.types import Document, Meaning, Point

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@register(KIND_EXTRACTOR, "sentence-fallback")
class SentenceFallbackExtractor(Extractor):
    """Dependency-free placeholder: one Point per sentence.

    Good enough to exercise the pipeline end to end in tests, but it does NOT
    truly abstract meaning from form — a proper LLM extractor is required for
    real watermark removal, because keeping sentence-level surface intact keeps
    too much of the original token sequence. Use only for wiring/tests.
    """

    name = "sentence-fallback"

    def extract(self, doc: Document) -> Meaning:
        sents = [s.strip() for s in _SENT_SPLIT.split(doc.text) if s.strip()]
        points = tuple(Point(intent=s) for s in sents)
        return Meaning(points=points, order="as-given", meta={"extractor": self.name})


@register(KIND_EXTRACTOR, "local-llm")
class LocalLLMExtractor(Extractor):
    """Real extractor: prompt a LOCAL model to outline the document into atomic
    points, capturing must-keep facts as constraints. Not yet implemented."""

    name = "local-llm"

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def extract(self, doc: Document) -> Meaning:  # pragma: no cover - stub
        raise NotImplementedError(
            "LocalLLMExtractor needs a local model backend (see [regenerate] extra). "
            "Use 'sentence-fallback' for wiring/tests."
        )
