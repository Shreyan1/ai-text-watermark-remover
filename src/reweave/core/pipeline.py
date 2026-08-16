"""The orchestrator + convergence loop.

STABLE CORE. This is the only place the five stages are wired together. It holds
no watermark-scheme knowledge and no model knowledge, only the sequencing and
the loop rule that makes the system 'ever-evolving':

    optimise toward a FIXED target (statistical human-ness), never against a
    named detector (Invariant I2).

The loop turns up `aggressiveness` until the human-signature clears threshold AND
the semantic guard is satisfied, or `max_iterations` is hit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .interfaces import (
    Extractor,
    FactChecker,
    Regenerator,
    Scorer,
    Scrubber,
    SemanticGuard,
)
from .types import (
    Document,
    FactReport,
    HumanSignature,
    StageTrace,
    TransformResult,
    Verdict,
    VoiceProfile,
)


@dataclass(frozen=True)
class PipelineConfig:
    human_threshold: float = 0.70  # emit when human-signature >= this
    similarity_floor: float = 0.82  # reject if meaning drifts below this
    max_iterations: int = 4
    start_aggressiveness: float = 0.45
    aggressiveness_step: float = 0.18
    enforce_unwatermarked: bool = True  # refuse a watermarked regenerator (Self-Watermark Trap)
    #: Reject candidates that lose or invert facts. Only takes effect when a
    #: FactChecker is wired in; the guard alone cannot see truth value.
    enforce_facts: bool = True


class Pipeline:
    """Wires the five stages. Stages are injected (dependency-inward, Invariant
    I3): the pipeline depends on the *interfaces*, never on concrete adapters."""

    def __init__(
        self,
        scrubber: Scrubber,
        extractor: Extractor,
        regenerator: Regenerator,
        scorer: Scorer,
        guard: SemanticGuard,
        config: PipelineConfig | None = None,
        fact_checker: FactChecker | None = None,
    ) -> None:
        self.scrubber = scrubber
        self.extractor = extractor
        self.regenerator = regenerator
        self.scorer = scorer
        self.guard = guard
        self.fact_checker = fact_checker
        self.config = config or PipelineConfig()

        if self.config.enforce_unwatermarked and not getattr(
            regenerator, "is_unwatermarked", False
        ):
            raise ValueError(
                f"regenerator {regenerator.name!r} is not declared unwatermarked. "
                "Regenerating with a watermarked model re-stamps a fresh mark "
                "(the Self-Watermark Trap). Use a local open-weight model, or set "
                "enforce_unwatermarked=False only if you truly know what you are doing."
            )

    def run(self, doc: Document, voice: VoiceProfile | None = None) -> TransformResult:
        voice = voice or VoiceProfile()
        trace: list[StageTrace] = []

        # ① Scrub, deterministic artifact hygiene.
        scrubbed = self.scrubber.scrub(doc)
        trace.append(StageTrace("scrub", self.scrubber.name, "character/encoding hygiene"))

        before = self.scorer.score(scrubbed, voice)
        trace.append(StageTrace("score", self.scorer.name, "baseline",
                                {"score": before.score, "verdict": before.verdict.value}))

        # If already human-like, don't touch the prose, humans stop when done.
        if before.score >= self.config.human_threshold:
            return TransformResult(
                original=doc, output=scrubbed, before=before, after=before,
                semantic_similarity=1.0, iterations=0, converged=True,
                trace=tuple(trace), facts=FactReport(ok=True),  # unchanged text
            )

        # ② Extract, collapse to meaning once; reused across retries.
        meaning = self.extractor.extract(scrubbed)
        trace.append(StageTrace("extract", self.extractor.name,
                                f"{len(meaning.points)} points"))

        best: Document = scrubbed
        best_sig: HumanSignature = before
        best_sim: float = 1.0
        best_facts: FactReport | None = FactReport(ok=True) if self.fact_checker else None
        converged = False
        aggressiveness = self.config.start_aggressiveness
        i = 0

        # ③–⑤ Regenerate -> Score -> Gate, looping toward the fixed target.
        for i in range(1, self.config.max_iterations + 1):
            candidate = self.regenerator.regenerate(meaning, voice, aggressiveness)
            sig = self.scorer.score(candidate, voice)
            sim = self.guard.similarity(scrubbed, candidate)

            # Two orthogonal meaning tests. The guard sees topic drift; it is
            # blind to truth value (negation pairs measure ~0.9 similar), so the
            # fact checker has to answer that half separately.
            facts = self.fact_checker.check(scrubbed, candidate) if self.fact_checker else None
            facts_ok = (
                facts.ok if (facts is not None and self.config.enforce_facts) else True
            )

            trace.append(StageTrace(
                "iterate", self.regenerator.name,
                f"aggr={aggressiveness:.2f}",
                {"score": sig.score, "similarity": sim, "verdict": sig.verdict.value,
                 "facts": facts.summary() if facts else "not checked"},
            ))

            meaning_ok = sim >= self.config.similarity_floor and facts_ok
            human_ok = sig.score >= self.config.human_threshold

            # Track the best meaning-preserving candidate seen so far.
            if meaning_ok and sig.score > best_sig.score:
                best, best_sig, best_sim, best_facts = candidate, sig, sim, facts

            if human_ok and meaning_ok:
                best, best_sig, best_sim, best_facts = candidate, sig, sim, facts
                converged = True
                break

            aggressiveness = min(1.0, aggressiveness + self.config.aggressiveness_step)

        return TransformResult(
            original=doc, output=best, before=before, after=best_sig,
            semantic_similarity=best_sim, iterations=i, converged=converged,
            trace=tuple(trace), facts=best_facts,
        )
