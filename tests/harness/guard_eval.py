"""Guard evaluation: does the embedding guard fix Jaccard's failure mode?

The gate needs a similarity that says YES to "same meaning, different words" and
NO to "different meaning". Jaccard cannot: it measures word overlap, so a
faithful reword looks like a topic change. This quantifies the difference.

The decisive metric is the MARGIN: similarity(faithful reword) minus
similarity(different topic). A usable guard has a large positive margin, because
the floor has to sit between them.

Run:  PYTHONPATH=src python3 tests/harness/guard_eval.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from reweave.core.types import Document  # noqa: E402
from reweave.guard import JaccardGuard, OllamaEmbeddingGuard  # noqa: E402

# (source, faithful reword, different topic)
CASES = [
    (
        "The deadline moved to Friday because the vendor shipped the parts late, "
        "so we lost three days of assembly time.",
        "We're now finishing on Friday. The supplier's delivery ran behind, which "
        "cost us about three days on the build.",
        "The quarterly marketing budget was reallocated toward paid search after "
        "the conference underperformed.",
    ),
    (
        "Watermarks embedded in token choices survive copy-paste but degrade when "
        "the text is rewritten substantially.",
        "If you copy and paste, the signal hidden in word selection travels with it. "
        "Rewrite the thing properly and it falls apart.",
        "Sourdough needs a mature starter, a long cold proof, and an oven hot "
        "enough to blister the crust.",
    ),
    (
        "Our error rate dropped from 4.2% to 0.8% after we added retry logic to the "
        "payment webhook.",
        "Adding retries on the payment webhook took us from a 4.2% error rate down "
        "to 0.8%.",
        "Migratory birds navigate using a combination of magnetic sensing and "
        "learned landmarks along the flyway.",
    ),
]


# Logical inversions: same topic, opposite truth value. A guard that scores these
# HIGH cannot protect facts, only topic.
INVERSIONS = [
    ("Xbox and PlayStation are computers.", "Xbox and PlayStation are not computers."),
    ("The deployment succeeded and no data was lost.",
     "The deployment failed and all data was lost."),
    ("Revenue increased by 40% this quarter.", "Revenue decreased by 40% this quarter."),
]


def blind_spot(guard) -> None:
    print("\nNegation / inversion blind spot (same topic, opposite meaning):")
    scores = []
    for a, b in INVERSIONS:
        s = guard.similarity(Document(text=a), Document(text=b))
        scores.append(s)
        print(f"  {s:.3f}   \"{a[:46]}\" vs inverted")
    print(f"  ── mean={sum(scores)/len(scores):.3f}, well above any usable floor.")
    print("     -> the guard prevents TOPIC DRIFT, not FACT CORRUPTION. Protecting")
    print("       facts needs entailment/NLI or Point.constraints verification.")


def main() -> int:
    guards = [("Jaccard (word overlap)", JaccardGuard()),
              ("Embedding (meaning)", OllamaEmbeddingGuard())]

    print("=" * 74)
    print("GUARD EVAL, can the guard tell 'same meaning' from 'different meaning'?")
    print("=" * 74)

    for label, guard in guards:
        rewords, others = [], []
        print(f"\n{label}")
        for i, (src, reword, other) in enumerate(CASES, 1):
            s = Document(text=src)
            r = guard.similarity(s, Document(text=reword))
            o = guard.similarity(s, Document(text=other))
            rewords.append(r)
            others.append(o)
            print(f"  case {i}:  faithful-reword={r:.3f}   different-topic={o:.3f}")
        mr = sum(rewords) / len(rewords)
        mo = sum(others) / len(others)
        margin = mr - mo
        print(f"  ── mean: reword={mr:.3f}  other={mo:.3f}  MARGIN={margin:+.3f}")
        if mr < 0.5:
            print("     ✗ rejects faithful rewords, unusable as a gate")
        elif margin < 0.25:
            print("     ✗ margin too small to place a floor between them")
        else:
            floor = (mr + mo) / 2
            print(f"     ✓ usable, a floor near {floor:.2f} separates them cleanly")

    blind_spot(guards[-1][1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
