"""Fact preservation by entailment, closing the rule checker's lexical gap.

`constraint_checker.py` catches negation and *listed* antonym flips. It cannot
catch a reversal written in different words:

    source    : "Sales climbed through the summer."
    rewrite   : "Sales were disappointing through the summer."

No negator, no listed antonym, high embedding similarity (same topic). Every
instrument we had says fine. An NLI model says CONTRADICTION.

METHOD, SummaC (Laban, Schnabel, Bennett & Hearst, TACL 2022), with one
deliberate deviation.

  Faithful to the paper:
    * Sentence-level granularity. NLI models are trained on sentence pairs, and
      the paper's central finding is that feeding whole documents to a
      sentence-trained model is the reason earlier NLI approaches underperformed.
      Both texts are split into sentences and compared pairwise.
    * Max-then-mean aggregation over the pair matrix for the coverage score
      (SummaC-ZS): per candidate sentence, the best supporting source sentence;
      then the mean across candidate sentences.

  Deviation, we do NOT compute the full M×N matrix:
    * SummaC scores *consistency* via max ENTAILMENT, where taking a max over all
      pairs is safe: the best supporter is the right one to keep. We need
      CONTRADICTION, where a max over all pairs is actively wrong, two unrelated
      sentences in the same document routinely look contradictory ("the API is
      synchronous" vs "the webhook is asynchronous"), and one spurious pair would
      veto a good rewrite.
    * So contradiction is judged only on *aligned* pairs: each source claim is
      matched to its nearest candidate claim first, then judged. This is more
      precise for our question and reduces an M×N call count to O(M), which is
      what makes an LLM backend affordable at all.
    * The cost of the deviation, stated plainly: a claim reversed AND moved to a
      lexically distant sentence may not align, and would then be reported as
      dropped rather than inverted. Alignment quality bounds recall.

Alignment uses embeddings when an embedder is supplied, and falls back to lexical
set cosine. That split plays to each instrument's strength: embeddings are
excellent at "are these two sentences about the same thing?" and blind only to
polarity, which is precisely the question NLI then answers.
"""

from __future__ import annotations

import math

from ..core.interfaces import FactChecker
from ..core.registry import KIND_FACTCHECKER, register
from ..core.types import Constraints, Document, FactReport
from .constraint_checker import _align_score
from .constraints import MIN_CLAIM_WORDS, claim_key, extract_constraints, sentences
from .nli import NLIBackend


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


@register(KIND_FACTCHECKER, "nli")
class NLIChecker(FactChecker):
    """Entailment-based fact preservation.

    Honest limits:
      * Bounded by the NLI backend. On the six-dataset SummaC benchmark, the
        best trained configuration reaches 74.4% balanced accuracy, automated
        factual-consistency detection is an open problem, not a solved one. Do
        not read a pass as proof.
      * Bounded by alignment (see the deviation note above).
      * Costs a model call per aligned claim. `ConstraintChecker` is free and
        deterministic; prefer `CompositeChecker`, which runs the cheap one first.
    """

    name = "nli"

    def __init__(
        self,
        backend: NLIBackend,
        embedder=None,
        align_floor: float = 0.45,
        embed_align_floor: float = 0.45,
        contradiction_threshold: float | None = None,
        max_claims: int = 40,
    ) -> None:
        self.backend = backend
        #: Any callable str -> vector; `OllamaEmbeddingGuard.embed` fits.
        self.embedder = embedder
        self.align_floor = align_floor
        self.embed_align_floor = embed_align_floor
        self.contradiction_threshold = (
            backend.default_contradiction_threshold
            if contradiction_threshold is None
            else contradiction_threshold
        )
        #: Hard cap so a long document cannot fan out into unbounded model calls.
        #: Truncation is REPORTED in FactReport.detail, a silent cap would let a
        #: partial check read as a full one (I5).
        self.max_claims = max_claims

    def constraints(self, doc: Document) -> Constraints:
        return extract_constraints(doc.text)

    def _claims(self, text: str) -> list[str]:
        return [s for s in sentences(text) if len(claim_key(s)) >= MIN_CLAIM_WORDS]

    def _align(self, src: list[str], cand: list[str]) -> list[tuple[int, int, float]]:
        """Greedy one-to-one best matches, embeddings first, lexical otherwise."""
        if self.embedder is not None:
            sv = [self.embedder(s) for s in src]
            cv = [self.embedder(c) for c in cand]
            floor = self.embed_align_floor
            score = lambda i, j: _cosine(sv[i], cv[j])  # noqa: E731
        else:
            sk = [claim_key(s) for s in src]
            ck = [claim_key(c) for c in cand]
            floor = self.align_floor
            score = lambda i, j: _align_score(sk[i], ck[j])  # noqa: E731

        pairs = sorted(
            ((score(i, j), i, j) for i in range(len(src)) for j in range(len(cand))),
            key=lambda p: -p[0],
        )
        used_s: set[int] = set()
        used_c: set[int] = set()
        out: list[tuple[int, int, float]] = []
        for sc, i, j in pairs:
            if sc < floor:
                break
            if i in used_s or j in used_c:
                continue
            used_s.add(i)
            used_c.add(j)
            out.append((i, j, sc))
        return out

    def check(self, source: Document, candidate: Document) -> FactReport:
        src = self._claims(source.text)
        cand = self._claims(candidate.text)
        truncated = len(src) > self.max_claims or len(cand) > self.max_claims
        src, cand = src[: self.max_claims], cand[: self.max_claims]

        if not src or not cand:
            return FactReport(ok=True, detail={"checker": self.name, "reason": "no claims"})

        aligned = self._align(src, cand)
        scored = self.backend.predict_batch([(src[i], cand[j]) for i, j, _ in aligned])

        inversions: list[tuple[str, str]] = []
        entail_hits = 0
        for (i, j, _), nli in zip(aligned, scored):
            if nli.contradiction >= self.contradiction_threshold:
                inversions.append((src[i], cand[j]))
            if nli.entailment >= 0.5:
                entail_hits += 1

        # SummaC-ZS coverage: per source claim, its best support; mean across all.
        # An unaligned claim contributes 0, it found no support anywhere.
        coverage = entail_hits / len(src) if src else 1.0

        return FactReport(
            ok=not inversions,
            inversions=tuple(inversions),
            detail={
                "checker": self.name,
                "backend": self.backend.name,
                "entailment_coverage": coverage,
                "claims_source": len(src),
                "claims_aligned": len(aligned),
                "truncated": truncated,
            },
        )
