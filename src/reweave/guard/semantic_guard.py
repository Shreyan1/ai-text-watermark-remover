"""Semantic guard — the meaning-preservation floor.

Removal that garbles content is a failure, not a success (Invariant I5). The gate
rejects any regeneration whose similarity to the source falls below the floor.

The real guard is `embedding_guard.OllamaEmbeddingGuard` (registered as
"ollama-embed"), which measures meaning rather than word overlap. What remains
here is the dependency-free lexical fallback, for environments with no local
embedding model.
"""

from __future__ import annotations

import re

from ..core.interfaces import SemanticGuard
from ..core.registry import KIND_GUARD, register
from ..core.types import Document

_WORD = re.compile(r"[a-z0-9']+")


def _content_tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@register(KIND_GUARD, "jaccard-fallback")
class JaccardGuard(SemanticGuard):
    """Lexical Jaccard overlap. A crude proxy — it rewards surface overlap, which
    is exactly the wrong bias for a system whose job is to change the surface: a
    faithful reword scores LOW and gets rejected. Kept only as a zero-dependency
    fallback. Prefer OllamaEmbeddingGuard on real text."""

    name = "jaccard-fallback"

    def similarity(self, a: Document, b: Document) -> float:
        ta, tb = _content_tokens(a.text), _content_tokens(b.text)
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
