"""Rules first, then entailment, the checker you actually want to run.

The two checkers fail in opposite directions, which is the whole reason to run
both rather than pick one:

  ConstraintChecker  deterministic, free, exact on numbers and names.
                     Blind to reversal written in unlisted words.
  NLIChecker         catches reworded reversal. Costs a model call per claim,
                     is only as good as its backend (SummaC's best trained
                     configuration is 74.4% balanced accuracy), and cannot tell
                     you that "40%" went missing, a dropped number is not a
                     contradiction, it is an omission, and NLI scores omission
                     as NEUTRAL.

Order matters for cost, not just tidiness: the deterministic checker runs first
and its findings stand on their own. NLI runs second and can only ADD findings.
A model that is unavailable or wrong can therefore never *weaken* the guarantee
the rules already give you, the union is monotonic in safety.
"""

from __future__ import annotations

from ..core.interfaces import FactChecker
from ..core.registry import KIND_FACTCHECKER, register
from ..core.types import Constraints, Document, FactReport


@register(KIND_FACTCHECKER, "composite")
class CompositeChecker(FactChecker):
    """Union of a deterministic checker and an optional semantic one."""

    name = "composite"

    def __init__(self, rules: FactChecker, nli: FactChecker | None = None,
                 fail_open: bool = False) -> None:
        self.rules = rules
        self.nli = nli
        #: If the NLI backend errors, treat it as "no extra findings" (True) or
        #: re-raise (False). Defaults to raising: a silent downgrade from
        #: semantic+lexical to lexical-only would let the caller believe they
        #: still had a check they no longer have.
        self.fail_open = fail_open

    def constraints(self, doc: Document) -> Constraints:
        return self.rules.constraints(doc)

    def check(self, source: Document, candidate: Document) -> FactReport:
        base = self.rules.check(source, candidate)
        if self.nli is None:
            return base

        try:
            extra = self.nli.check(source, candidate)
        except Exception as e:  # noqa: BLE001 - backend transport/availability
            if not self.fail_open:
                raise
            detail = dict(base.detail)
            detail["nli_error"] = str(e)
            detail["nli_ran"] = False
            return FactReport(
                ok=base.ok, missing_numerals=base.missing_numerals,
                missing_entities=base.missing_entities, inversions=base.inversions,
                numeral_coverage=base.numeral_coverage,
                entity_coverage=base.entity_coverage, detail=detail,
            )

        # Union the inversions, de-duplicated on the source claim: both checkers
        # finding the same flip is confirmation, not two problems.
        seen = {s for s, _ in base.inversions}
        merged = list(base.inversions) + [p for p in extra.inversions if p[0] not in seen]

        detail = dict(base.detail)
        detail.update({
            "checker": self.name,
            "nli_ran": True,
            "rules_inversions": len(base.inversions),
            "nli_only_inversions": len(merged) - len(base.inversions),
            **{f"nli_{k}": v for k, v in extra.detail.items()},
        })
        return FactReport(
            ok=base.ok and extra.ok,
            missing_numerals=base.missing_numerals,
            missing_entities=base.missing_entities,
            inversions=tuple(merged),
            numeral_coverage=base.numeral_coverage,
            entity_coverage=base.entity_coverage,
            detail=detail,
        )
