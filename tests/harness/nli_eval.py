"""Does NLI close the lexical gap — and what does it cost in false alarms?

The rule checker catches negation and listed antonyms. This measures the case it
provably cannot see: a reversal written in entirely different words, with no
negator and no listed antonym pair. If NLI does not beat the rules here, it is
not worth a model call per claim.

Three checkers, three question sets, so the trade is visible rather than asserted:

  LEXICAL      inversions the rules already catch  — NLI must not regress
  REWORDED     the gap: same topic, opposite claim, no shared cue
  FAITHFUL     honest rewrites — the false-alarm test that decides usability

Run:
    PYTHONPATH=src python3 tests/harness/nli_eval.py            # rules only
    PYTHONPATH=src python3 tests/harness/nli_eval.py --nli      # + local NLI
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reweave.core.types import Document  # noqa: E402
from reweave.guard import OllamaEmbeddingGuard  # noqa: E402
from reweave.verify import (  # noqa: E402
    CompositeChecker,
    ConstraintChecker,
    NLIChecker,
    OllamaNLIBackend,
)

# ── Inversions the RULES catch: an explicit negator or a listed antonym. ──────
LEXICAL = [
    ("Xbox and PlayStation are computers that run an operating system.",
     "Xbox and PlayStation aren't computers that run an operating system."),
    ("The deployment succeeded and traffic was restored.",
     "The deployment failed and traffic was restored."),
    ("Revenue increased 40% in the third quarter on enterprise renewals.",
     "Revenue decreased 40% in the third quarter on enterprise renewals."),
]

# ── THE GAP: reversed meaning, no negator, no listed antonym. ────────────────
REWORDED = [
    ("Sales climbed steadily through the summer.",
     "Sales were disappointing throughout the summer."),
    ("The migration finished ahead of schedule.",
     "The migration overran its deadline by several weeks."),
    ("The API is stable and safe to build on.",
     "The API is still experimental and may break without warning."),
    ("Every region stayed online during the incident.",
     "Several regions went dark during the incident."),
    ("The method requires the model owner's secret key.",
     "The method works for anyone, using only public information."),
    ("Adoption was strongest among enterprise customers.",
     "Enterprise customers were the slowest to adopt it."),
    ("The watermark survives light copy-editing.",
     "Light copy-editing is enough to destroy the watermark."),
    ("Reviewers praised the paper's clarity.",
     "Reviewers found the paper hard to follow."),
]

# ── Honest rewrites. Any rejection here is a false alarm. ────────────────────
FAITHFUL = [
    ("Sales climbed steadily through the summer.",
     "Revenue rose month over month across the summer."),
    ("The migration finished ahead of schedule.",
     "We wrapped the migration up early."),
    ("The API is stable and safe to build on.",
     "You can build on the API — it's solid."),
    ("Every region stayed online during the incident.",
     "All regions remained available while the incident was ongoing."),
    ("The method requires the model owner's secret key.",
     "You need the model owner's secret key to run the method."),
    ("Adoption was strongest among enterprise customers.",
     "Enterprise buyers took it up faster than anyone else."),
    ("The watermark survives light copy-editing.",
     "Small copy-edits leave the watermark intact."),
    ("Reviewers praised the paper's clarity.",
     "The reviewers thought the paper was very clearly written."),
]


def _run(checker, cases, expect_reject: bool) -> tuple[int, float, list[str]]:
    hits, notes = 0, []
    t0 = time.time()
    for src, cand in cases:
        r = checker.check(Document(text=src), Document(text=cand))
        rejected = not r.ok
        if rejected == expect_reject:
            hits += 1
        else:
            notes.append(cand[:64])
    return hits, time.time() - t0, notes


def main() -> int:
    use_nli = "--nli" in sys.argv
    model = os.environ.get("REWEAVE_NLI_MODEL", "gemma3:4b")

    rules = ConstraintChecker()
    checkers: list[tuple[str, object]] = [("rules", rules)]

    if use_nli:
        backend = OllamaNLIBackend(model=model)
        # Lexical alignment CANNOT be used here and the gap set proves why: for
        # "migration finished ahead of schedule" vs "migration overran its
        # deadline", content-word cosine is 0.22 — below any sane floor — so the
        # pair never reaches the NLI model at all. Embeddings align on topic,
        # which is their strength and exactly what they are blind-to-polarity
        # good for. Each instrument used where it is strong.
        nli = NLIChecker(backend, embedder=OllamaEmbeddingGuard().embed)
        checkers.append((f"NLI ({model})", nli))
        checkers.append(("composite", CompositeChecker(rules, nli)))
        checkers.append(("NLI lexical-align", NLIChecker(backend)))

    print("\nCLOSING THE LEXICAL GAP WITH NLI")
    print("=" * 78)
    print(f"{'checker':<18} {'LEXICAL':>10} {'REWORDED':>10} {'FAITHFUL':>10} {'time':>8}")
    print(f"{'':18} {'(recall)':>10} {'(recall)':>10} {'(no alarm)':>10}")
    print("-" * 78)

    misses: dict[str, list[str]] = {}
    for label, c in checkers:
        lex, t1, _ = _run(c, LEXICAL, True)
        rew, t2, m_rew = _run(c, REWORDED, True)
        fai, t3, m_fai = _run(c, FAITHFUL, False)
        print(f"{label:<18} {lex:>6}/{len(LEXICAL):<3} {rew:>6}/{len(REWORDED):<3} "
              f"{fai:>6}/{len(FAITHFUL):<3} {t1 + t2 + t3:>7.1f}s")
        misses[label] = [f"MISSED reversal: {x}" for x in m_rew] + \
                        [f"FALSE ALARM   : {x}" for x in m_fai]

    print("-" * 78)
    if use_nli:
        print("  NOTE: rows after the first NLI row reuse its (premise,hypothesis)")
        print("        cache, so their times are cache hits — not their real cost.")
        print(f"        Real cost: {backend.calls} model calls, "
              f"{backend.parse_failures} unparseable.")
    for label, ms in misses.items():
        if ms:
            print(f"\n  {label}:")
            for m in ms:
                print(f"    · {m}")

    if not use_nli:
        print("\n  Run with --nli to measure the NLI and composite rows "
              "(needs Ollama + an instruct model).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
