"""Feature extraction for the human-signature scorer.

The zero-dependency features (burstiness, TTR, punctuation, density) are
implemented here and work today. Perplexity requires a reference language model
and is optional, it stays None until a backend is wired in the `[score]` extra.

AI's failure mode is uniformity; every feature here measures a dimension of
variance that human writing has and unedited AI lacks. The set is OPEN
(Invariant I1): add a feature, give it a slot on FeatureVector, weight it in
human_signature.py. No trained classifier, nothing to retrain as models drift.
"""

from __future__ import annotations

import re
from statistics import mean, pstdev

from ..core.types import FeatureVector

_SENT_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
_WORD = re.compile(r"[A-Za-z0-9']+")
_PARA_SPLIT = re.compile(r"\n\s*\n")
# Capitalised multi-word runs, a cheap proxy for named entities, no NLP dep.
_ENTITY = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_NUMERAL = re.compile(r"\b\d[\d,.]*\b")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _cv(values: list[float]) -> float | None:
    """Coefficient of variation, scale-free dispersion. Higher = more human."""
    if len(values) < 2:
        return None
    m = mean(values)
    if m == 0:
        return None
    return pstdev(values) / m


def extract_features(text: str) -> FeatureVector:
    sents = _sentences(text)
    words = _words(text)
    paras = [p for p in _PARA_SPLIT.split(text) if p.strip()]

    sent_lengths = [float(len(_words(s))) for s in sents]
    para_lengths = [float(len(_words(p))) for p in paras]
    lower_words = [w.lower() for w in words]

    burstiness = pstdev(sent_lengths) if len(sent_lengths) >= 2 else None
    ttr = (len(set(lower_words)) / len(lower_words)) if lower_words else None
    paragraph_cv = _cv(para_lengths)

    n_words = len(words) or 1
    em_dash_rate = text.count("—") / n_words
    entity_density = len(_ENTITY.findall(text)) / n_words
    numeral_density = len(_NUMERAL.findall(text)) / n_words

    # Rule-of-three: comma-separated triads like "fast, cheap, and reliable".
    triads = len(re.findall(r"\w+,\s+\w+,\s+and\s+\w+", text))
    rule_of_three_rate = triads / (len(sents) or 1)

    return FeatureVector(
        burstiness=burstiness,
        type_token_ratio=ttr,
        paragraph_cv=paragraph_cv,
        em_dash_rate=em_dash_rate,
        rule_of_three_rate=rule_of_three_rate,
        entity_density=entity_density,
        numeral_density=numeral_density,
        # perplexity_* left None; filled by an optional LM backend.
    )
