"""The fact gate, did the rewrite keep what it was supposed to keep?

This exists because the semantic guard is measurably blind to truth value. The
harness scored these pairs with local sentence embeddings:

    "X are computers"        vs "X are NOT computers"        -> 0.959
    "deployment succeeded"   vs "deployment failed"          -> 0.898
    "revenue increased 40%"  vs "revenue decreased 40%"      -> 0.776

All three sail past any usable similarity floor, because embeddings encode topic
and these pairs share their topic exactly. That is not a tuning problem, it is
what the representation is for. So the fix is not a better floor; it is a second
check that looks at the thing embeddings discard.

Three tests, cheapest first:
  1. numerals  , every number in the source must still be present
  2. entities  , proper nouns must survive (coverage floor, since a reword may
                  legitimately pronominalise some)
  3. polarity  , align each source claim to its nearest candidate claim and
                  compare negation + antonym direction

Alignment uses set cosine, |A∩B|/sqrt(|A|·|B|), and is assigned greedily
one-to-one. Both choices were forced by measurement, not taste. The overlap
coefficient |A∩B|/min(|A|,|B|) looks like the obvious pick, it tolerates the
length changes a rewrite makes, but it scores a one-word fragment at a perfect
1.0 against any sentence containing that word. On real Gemma output, headings
like "Metaphor:" hijacked whole paragraphs and produced phantom inversions.
Cosine penalises that size mismatch. Greedy one-to-one then stops several source
claims from all collapsing onto the same candidate.
"""

from __future__ import annotations

from ..core.interfaces import FactChecker
from ..core.registry import KIND_FACTCHECKER, register
from ..core.types import Constraints, Document, FactReport
from .constraints import (
    MIN_CLAIM_WORDS,
    claim_key,
    entities,
    extract_constraints,
    numerals,
    polarity_flipped,
    sentences,
)


def _align_score(a: frozenset[str], b: frozenset[str]) -> float:
    """Set cosine. Unlike the overlap coefficient it cannot be gamed by a
    one-word fragment, which is what broke alignment on real markdown output."""
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


@register(KIND_FACTCHECKER, "constraint")
class ConstraintChecker(FactChecker):
    """Deterministic, zero-dependency fact preservation check.

    Honest limits, stated up front:
      * It catches *lexical* inversion, a negator or a known antonym flipping.
        A claim rewritten into an opposite meaning with entirely different words
        ("sales climbed" -> "sales were disappointing") is not caught.
      * `_ANTONYMS` is a finite list. It covers the common quantitative and
        outcome flips; it is not a lexicon.
      * A dropped claim reads as unaligned, not inverted. That is reported, but
        only counted against the candidate as coverage, not as a flip.

    So: passing means no *detectable* fact corruption, not proof of fidelity.
    It converts the blind spot into a narrower blind spot, which is the honest
    claim (Invariant I5).
    """

    name = "constraint"

    def __init__(
        self,
        entity_floor: float = 0.75,
        numeral_floor: float = 1.0,
        align_floor: float = 0.45,
        allow_inversions: int = 0,
    ) -> None:
        #: Names may be legitimately pronominalised, so allow some slack.
        self.entity_floor = entity_floor
        #: Numbers may not. Dropping "40%" from a claim about 40% is fact loss.
        self.numeral_floor = numeral_floor
        #: Minimum content-word overlap for two claims to count as the same claim.
        self.align_floor = align_floor
        self.allow_inversions = allow_inversions

    def constraints(self, doc: Document) -> Constraints:
        return extract_constraints(doc.text)

    def check(self, source: Document, candidate: Document) -> FactReport:
        src_nums, cand_nums = numerals(source.text), numerals(candidate.text)
        src_ents, cand_ents = entities(source.text), entities(candidate.text)

        missing_nums = sorted(src_nums - cand_nums)
        # Entity match is substring-tolerant: "OpenAI's" and possessives, and
        # multi-word names whose parts survive separately.
        cand_blob = candidate.text.lower()
        missing_ents = sorted(e for e in src_ents if e not in cand_blob)

        num_cov = 1.0 if not src_nums else 1.0 - len(missing_nums) / len(src_nums)
        ent_cov = 1.0 if not src_ents else 1.0 - len(missing_ents) / len(src_ents)

        inversions = self._find_inversions(source.text, candidate.text)

        ok = (
            num_cov >= self.numeral_floor
            and ent_cov >= self.entity_floor
            and len(inversions) <= self.allow_inversions
        )
        return FactReport(
            ok=ok,
            missing_numerals=tuple(missing_nums),
            missing_entities=tuple(missing_ents),
            inversions=tuple(inversions),
            numeral_coverage=num_cov,
            entity_coverage=ent_cov,
            detail={
                "source_numerals": len(src_nums),
                "source_entities": len(src_ents),
                "checker": self.name,
            },
        )

    def _find_inversions(self, src_text: str, cand_text: str) -> list[tuple[str, str]]:
        # Only real claims on BOTH sides. Filtering the source alone was a bug:
        # candidate fragments ("Hardware:", "species.") aligned to full source
        # sentences and produced phantom inversions on real model output.
        src = [(s, claim_key(s)) for s in sentences(src_text)]
        src = [(s, k) for s, k in src if len(k) >= MIN_CLAIM_WORDS]
        cand = [(s, claim_key(s)) for s in sentences(cand_text)]
        cand = [(s, k) for s, k in cand if len(k) >= MIN_CLAIM_WORDS]
        if not src or not cand:
            return []

        # Greedy one-to-one: score every pair, take them best-first, and let each
        # claim be used once. Prevents many source claims collapsing onto one
        # candidate sentence and being judged against it repeatedly.
        pairs = [
            (_align_score(sk, ck), si, ci)
            for si, (_, sk) in enumerate(src)
            for ci, (_, ck) in enumerate(cand)
        ]
        pairs.sort(key=lambda p: -p[0])

        used_s: set[int] = set()
        used_c: set[int] = set()
        found: list[tuple[str, str]] = []
        for score, si, ci in pairs:
            if score < self.align_floor:
                break  # sorted, so nothing further can align
            if si in used_s or ci in used_c:
                continue
            used_s.add(si)
            used_c.add(ci)
            if polarity_flipped(src[si][0], cand[ci][0]):
                found.append((src[si][0], cand[ci][0]))
        return found
