"""The NLI backend contract — one call, three probabilities.

Kept deliberately small so the *method* (SummaC-style aggregation, in
nli_checker.py) is independent of what computes the labels. Two backends ship:
a local instruct model via Ollama, and a trained cross-encoder behind the
`[verify]` extra. Both satisfy this contract; the method does not change.

Why NLI at all: the rule-based checker (constraint_checker.py) catches negation
and known antonym flips, but it is lexical. "Sales climbed" rewritten as "sales
were disappointing" shares no negator and no listed antonym, so no rule sees it.
Entailment does: that pair is a textbook CONTRADICTION.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class NLIScores:
    """Probabilities over the three NLI classes, given (premise, hypothesis).

    A trained cross-encoder supplies a calibrated distribution. An instruct model
    prompted for a label supplies a one-hot. Both are valid here; the difference
    is confidence resolution, and it is why thresholds are backend-specific.
    """

    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        best = max(
            ("entailment", self.entailment),
            ("neutral", self.neutral),
            ("contradiction", self.contradiction),
            key=lambda kv: kv[1],
        )
        return best[0]

    @classmethod
    def one_hot(cls, label: str) -> "NLIScores":
        lab = label.strip().lower()
        return cls(
            entailment=1.0 if lab == "entailment" else 0.0,
            neutral=1.0 if lab == "neutral" else 0.0,
            contradiction=1.0 if lab == "contradiction" else 0.0,
        )


class NLIBackend(ABC):
    """Classify (premise, hypothesis) as entailment / neutral / contradiction."""

    name: str = "nli"

    #: Backends that return one-hot labels need a low threshold; calibrated
    #: cross-encoders can afford a high one. Declared here so the checker does
    #: not have to know which kind it was handed.
    default_contradiction_threshold: float = 0.5

    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> NLIScores:
        ...

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """Override where the backend can batch; the default is honest and slow."""
        return [self.predict(p, h) for p, h in pairs]
