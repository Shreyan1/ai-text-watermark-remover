"""Stage ② — the real extractor, over a LOCAL model via Ollama.

Reduce a document to its Meaning (atomic points), discarding surface form. This
severing of the token sequence is what breaks the watermark substrate (I4). We
keep the extraction faithful — every fact, name, number preserved — because the
guard stage will reject regenerations that drift too far.
"""

from __future__ import annotations

import re

from .._ollama import generate
from ..core.interfaces import Extractor
from ..core.registry import KIND_EXTRACTOR, register
from ..core.types import Document, Meaning, Point
from ..verify.constraints import entities, numerals

_SYSTEM = (
    "You extract the meaning of a text as a flat list of atomic points. "
    "One point per line, starting with '- '. Preserve every fact, name, and number. "
    "Do not add, interpret, or editorialise. Output only the list."
)
_LINE = re.compile(r"^\s*[-*]\s+(.*)$")


@register(KIND_EXTRACTOR, "ollama")
class OllamaExtractor(Extractor):
    name = "ollama"

    def __init__(self, model: str = "llama3.2:1b", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host

    def extract(self, doc: Document) -> Meaning:
        raw = generate(
            f"Extract the meaning as atomic points:\n\n{doc.text}\n\nPoints:",
            model=self.model, host=self.host, temperature=0.2, num_predict=512,
            system=_SYSTEM,
        )
        points = []
        for line in raw.splitlines():
            m = _LINE.match(line)
            if m and m.group(1).strip():
                points.append(_with_constraints(m.group(1).strip()))
        if not points:  # fallback: treat whole output as one point
            points = [_with_constraints(raw.strip())]

        # Source-level constraints, not just per-point: the model may drop a
        # fact during extraction, and the checker verifies against the SOURCE.
        # Carrying them on Meaning lets the regenerator be told them explicitly.
        return Meaning(
            points=tuple(points),
            order="free",
            meta={
                "extractor": self.name,
                "must_keep_numerals": sorted(numerals(doc.text)),
                "must_keep_entities": sorted(entities(doc.text)),
            },
        )


def _with_constraints(intent: str) -> Point:
    """Attach the point's own hard facts. These are what the fact gate verifies
    and what the regenerator prompt pins, so extraction and verification read
    the same list."""
    nums, ents = numerals(intent), entities(intent)
    cons: dict[str, list[str]] = {}
    if nums:
        cons["numerals"] = sorted(nums)
    if ents:
        cons["entities"] = sorted(ents)
    return Point(intent=intent, constraints=cons)
