"""Stage ① — deterministic character/encoding hygiene.

Pure standard-library code, zero dependencies, no model. Defeats the entire
character-artifact watermark family (threat-model Family A) and normalises the
punctuation tells that make text read as machine-formatted. This is the first
working piece of the pipeline and the fast path for callers who only want
artifact hygiene.
"""

from __future__ import annotations

import unicodedata

from ..core.interfaces import Scrubber
from ..core.registry import KIND_SCRUBBER, register
from ..core.types import Document

# Invisible / zero-width / formatting characters with no legitimate place in prose.
_ZERO_WIDTH = {
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "⁠",  # word joiner
    "﻿",  # zero-width no-break space / BOM
    "­",  # soft hyphen
    "᠎",  # Mongolian vowel separator
}
# Bidirectional / directional controls sometimes used to hide payloads.
_BIDI_CONTROLS = {chr(c) for c in range(0x202A, 0x202F)} | {chr(c) for c in range(0x2066, 0x206A)}

# Homoglyph map: common non-Latin look-alikes → their ASCII equivalent.
_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "ј": "j", "һ": "h",
    "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Χ": "X", "Ѕ": "S", "І": "I", "А": "A", "В": "B",
}
# Quote / dash normalisation → straight ASCII.
_PUNCT = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "―": "-",  # en/em/horizontal dash → hyphen
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ",
    "−": "-",  # minus sign
}


@register(KIND_SCRUBBER, "unicode")
class UnicodeScrubber(Scrubber):
    """Strip invisible characters, fold homoglyphs, normalise punctuation.

    Options let callers keep typographic quotes/dashes if they only care about
    the invisible-payload family, not the style tells.
    """

    name = "unicode"

    def __init__(self, normalise_punctuation: bool = True, fold_homoglyphs: bool = True) -> None:
        self.normalise_punctuation = normalise_punctuation
        self.fold_homoglyphs = fold_homoglyphs

    def scrub(self, doc: Document) -> Document:
        out_chars: list[str] = []
        for ch in doc.text:
            if ch in _ZERO_WIDTH or ch in _BIDI_CONTROLS:
                continue  # drop entirely
            if unicodedata.category(ch) == "Cf":  # any other format char
                continue
            if self.fold_homoglyphs and ch in _HOMOGLYPHS:
                out_chars.append(_HOMOGLYPHS[ch])
                continue
            if self.normalise_punctuation and ch in _PUNCT:
                out_chars.append(_PUNCT[ch])
                continue
            out_chars.append(ch)

        text = "".join(out_chars)
        # NFKC folds compatibility variants (e.g. ﬁ ligature, full-width forms).
        text = unicodedata.normalize("NFKC", text)
        # Collapse runs of spaces introduced by the substitutions, keep newlines.
        text = "\n".join(" ".join(line.split()) for line in text.split("\n"))

        meta = dict(doc.meta)
        meta["scrubbed"] = True
        return Document(text=text, meta=meta)
