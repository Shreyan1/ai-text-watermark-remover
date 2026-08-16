"""Stage ④, the human-signature scorer.

Blends the feature vector into a transparent 0..1 score (higher = more
human-like) with a Verdict. This is the achievable, honestly-labelled half of
'detection': it estimates how AI-uniform a text reads. It is NOT watermark
detection and must never be presented as such (Invariant I5).

The blend is a transparent weighted sum with documented normalisers, no trained
model, so nothing to retrain as LLMs drift. Tune weights, don't fit them.
"""

from __future__ import annotations

from ..core.interfaces import Scorer
from ..core.registry import KIND_SCORER, register
from ..core.types import Document, FeatureVector, HumanSignature, Verdict, VoiceProfile
from .features import extract_features


def _sat(value: float | None, human_ref: float) -> float | None:
    """Map a 'higher = more human' feature to 0..1, saturating at human_ref."""
    if value is None:
        return None
    return max(0.0, min(1.0, value / human_ref)) if human_ref > 0 else None


def _inv(value: float | None, bad_ref: float) -> float | None:
    """Map a 'higher = more AI' feature to 0..1 (1 = human, 0 = very AI)."""
    if value is None:
        return None
    return max(0.0, min(1.0, 1.0 - value / bad_ref)) if bad_ref > 0 else None


# Reference points sit near the HUMAN target so that AI text (which falls below on
# the discriminative axes) maps lower. A ref set *below* both classes saturates the
# gap to nothing, an early bug the harness caught.
#
# Only features that actually discriminate go in the SCORE. The harness measured
# single-feature AUROC(human>ai) on real Darwin-vs-Gemma text:
#   burstiness 0.906 | TTR 0.593 | em_dash 0.583 | rule_of_three 0.405
#   paragraph_cv 0.008 | entity_density 0.315 | numeral_density 0.308  <- ANTI-correlated
# So paragraph_cv / entity_density / numeral_density are DROPPED from the score:
# modern instruction-tuned models are specific and well-structured, so "specificity
# = human" is a false prior. Those features are still computed and reported as
# diagnostics on FeatureVector; they just don't vote. Perplexity is the intended
# PRIMARY signal but needs an LM backend (the `[score]` extra); until then it is
# None and the weights renormalise over what's available.
_REFS = {
    "burstiness": ("sat", 13.0),         # human sentence-length std ~12-13 words
    "type_token_ratio": ("sat", 0.66),   # human TTR target on medium passages
    "perplexity_var": ("sat", 25.0),     # spiky surprise reads human (needs backend)
    "em_dash_rate": ("inv", 0.02),       # heavy em-dash use reads AI
    "rule_of_three_rate": ("inv", 0.06), # frequent triads read AI
}
_WEIGHTS = {
    "burstiness": 0.45,        # the one strong signal without an LM
    "perplexity_var": 0.22,    # primary when a backend is present; renormalised out otherwise
    "type_token_ratio": 0.18,
    "rule_of_three_rate": 0.10,
    "em_dash_rate": 0.05,
}


@register(KIND_SCORER, "statistical")
class StatisticalScorer(Scorer):
    """Zero-training, transparent human-signature heuristic."""

    name = "statistical"

    #: Below this fraction of total feature weight, we abstain rather than guess.
    #: Short text loses burstiness (the dominant signal, needs >=2 sentences), and
    #: the remaining features renormalise to an inflated, meaningless score. This
    #: mirrors the selective-prediction/abstention mechanism in the SynthID-Text
    #: paper: report UNCERTAIN instead of a confident wrong answer.
    MIN_COVERAGE = 0.60

    def __init__(self, human_threshold: float = 0.70, ai_threshold: float = 0.45) -> None:
        self.human_threshold = human_threshold
        self.ai_threshold = ai_threshold

    def score(self, doc: Document, voice: VoiceProfile | None = None) -> HumanSignature:
        feats = extract_features(doc.text)
        contributions: dict[str, float] = {}
        total_w = 0.0
        acc = 0.0

        for key, weight in _WEIGHTS.items():
            raw = getattr(feats, key, None)
            if raw is None:
                raw = feats.extra.get(key)
            if raw is None:
                continue  # feature unavailable (e.g. perplexity without a backend)
            kind, ref = _REFS[key]
            norm = _sat(raw, ref) if kind == "sat" else _inv(raw, ref)
            if norm is None:
                continue
            contributions[key] = norm
            acc += weight * norm
            total_w += weight

        score = acc / total_w if total_w > 0 else 0.0
        score = self._apply_banned_terms(score, doc.text, voice, contributions)

        coverage = total_w / sum(_WEIGHTS.values())
        if coverage < self.MIN_COVERAGE:
            # Not enough signal to judge, abstain rather than emit a confident
            # wrong answer (e.g. a one-sentence AI-tell-laden string scoring high
            # because burstiness was unavailable).
            verdict = Verdict.UNCERTAIN
        elif score >= self.human_threshold:
            verdict = Verdict.HUMAN_LIKE
        elif score <= self.ai_threshold:
            verdict = Verdict.AI_LIKE
        else:
            verdict = Verdict.UNCERTAIN

        return HumanSignature(
            score=score,
            verdict=verdict,
            features=feats,
            rationale={
                "contributions": contributions,
                "weight_covered": total_w,
                "coverage": coverage,
                "abstained": coverage < self.MIN_COVERAGE,
            },
        )

    def _apply_banned_terms(
        self,
        score: float,
        text: str,
        voice: VoiceProfile | None,
        contributions: dict[str, float],
    ) -> float:
        """Each AI-tell word present pushes the score down. Cheap, high-signal."""
        banned = (voice.banned_terms if voice else None)
        if not banned:
            from ..core.types import DEFAULT_BANNED_TERMS
            banned = DEFAULT_BANNED_TERMS
        low = text.lower()
        hits = sum(1 for term in banned if term in low)
        if hits:
            penalty = min(0.3, 0.05 * hits)
            contributions["banned_terms_penalty"] = -penalty
            return max(0.0, score - penalty)
        return score
