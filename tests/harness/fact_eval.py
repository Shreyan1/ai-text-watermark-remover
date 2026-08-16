"""Does the fact checker close the embedding guard's blind spot?

Two things have to be true for this to be worth having:

  1. It CATCHES the inversions the embedding guard waves through (recall).
  2. It does NOT reject faithful rewordings (precision) — a fact gate that
     fires on every legitimate rewrite would shut the pipeline down exactly
     the way JaccardGuard did, just for a different reason.

Both are measured below. Run:

    PYTHONPATH=src python3 tests/harness/fact_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reweave.core.types import Document  # noqa: E402
from reweave.verify import ConstraintChecker  # noqa: E402


# ── The measured blind spot: same topic, opposite truth ────────────────────
# Embedding similarities from guard_eval.py are shown for comparison.
INVERSIONS = [
    ("Xbox and PlayStation are computers. They run an operating system and "
     "execute arbitrary code.",
     "Xbox and PlayStation aren't computers. They run an operating system and "
     "execute arbitrary code.",
     0.959),
    ("The deployment succeeded and the service came back online at full capacity.",
     "The deployment failed and the service came back online at full capacity.",
     0.898),
    ("Revenue increased 40% in the third quarter, driven by enterprise renewals.",
     "Revenue decreased 40% in the third quarter, driven by enterprise renewals.",
     0.776),
    ("The model was never trained on user conversations, and Anthropic has said so.",
     "The model was trained on user conversations, and Anthropic has said so.",
     None),
    ("All 12 regions reported the outage within the first hour.",
     "None of the 12 regions reported the outage within the first hour.",
     None),
]

# ── Faithful rewordings: different words, same claims. Must PASS. ──────────
FAITHFUL = [
    ("Xbox and PlayStation are computers. They run an operating system and "
     "execute arbitrary code.",
     "A PlayStation is a computer, and so is an Xbox. Each one boots an "
     "operating system and will execute whatever code you hand it."),
    ("Revenue increased 40% in the third quarter, driven by enterprise renewals.",
     "Enterprise renewals drove the quarter. Revenue climbed 40% in Q3."),
    ("The deployment succeeded and the service came back online at full capacity.",
     "That deploy worked. The service was back at full capacity afterwards."),
    ("SynthID-Text embeds a watermark using tournament sampling with 30 layers.",
     "Using tournament sampling across 30 layers, SynthID-Text plants its "
     "watermark right in the sampling step."),
]

# ── Fact loss (not inversion): a dropped number must also fail. ────────────
DROPPED = [
    ("Revenue increased 40% in the third quarter, driven by enterprise renewals.",
     "Revenue went up quite a bit last quarter, thanks to enterprise renewals."),
    ("Anthropic published the watermark explainer in August 2026.",
     "The company published its explainer at some point."),
]


def live_rejection_rate(n: int = 8) -> None:
    """How often does the gate fire on REAL regenerations?

    There are no ground-truth fidelity labels here, so this is deliberately NOT
    reported as precision. It is the rejection rate plus the reason for each
    rejection, so the reasons can be read and judged. A gate that rejects
    everything is as useless as one that rejects nothing — this is the number
    that tells you which one you built.
    """
    import json
    import os

    from reweave._ollama import is_up
    from reweave.extract import OllamaExtractor
    from reweave.regenerate import OllamaRegenerator
    from reweave.scrub import UnicodeScrubber
    from reweave.core.types import VoiceProfile

    if not is_up():
        print("\n[4] LIVE REGENERATIONS — skipped (Ollama not reachable)")
        return

    model = os.environ.get("REWEAVE_MODEL", "llama3.2:1b")
    data = Path(__file__).resolve().parents[2] / "dataset" / "synthid" / "human_eval.jsonl"
    docs = []
    with open(data, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            docs.append(json.loads(line)["watermarked_model_response"])

    scrub, ext, reg = UnicodeScrubber(), OllamaExtractor(model=model), OllamaRegenerator(model=model)
    checker = ConstraintChecker()

    print(f"\n[4] LIVE REGENERATIONS — {model} rewriting {n} real Gemma responses")
    rejected = 0
    for i, text in enumerate(docs, 1):
        src = scrub.scrub(Document(text=text))
        out = reg.regenerate(ext.extract(src), VoiceProfile(), aggressiveness=0.55)
        r = checker.check(src, out)
        if not r.ok:
            rejected += 1
        mark = "reject" if not r.ok else "accept"
        print(f"  {mark}  doc {i}: {r.summary()[:96]}")
        for s, c in r.inversions[:1]:
            print(f"          src : {s[:88]}")
            print(f"          cand: {c[:88]}")
    print(f"\n  rejection rate: {rejected}/{n} — read the reasons above, not just the ratio")


def _row(label: str, ok: bool, expected_ok: bool, detail: str) -> bool:
    passed = ok == expected_ok
    mark = "✓" if passed else "✗ MISS"
    print(f"  {mark:<7} {label:<9} {detail}")
    return passed


def main() -> int:
    checker = ConstraintChecker()
    hits = misses = 0

    print("\nFACT CHECKER — closing the embedding blind spot")
    print("=" * 74)

    print("\n[1] INVERSIONS — must be REJECTED (embeddings score these ~0.78-0.96)")
    for src, cand, emb in INVERSIONS:
        r = checker.check(Document(text=src), Document(text=cand))
        emb_s = f"embed={emb:.3f}" if emb is not None else "embed=n/a  "
        ok = _row(emb_s, r.ok, False, r.summary())
        hits, misses = (hits + 1, misses) if ok else (hits, misses + 1)

    print("\n[2] FAITHFUL REWORDS — must be ACCEPTED (no false alarms)")
    for src, cand in FAITHFUL:
        r = checker.check(Document(text=src), Document(text=cand))
        ok = _row("reword", r.ok, True, r.summary())
        hits, misses = (hits + 1, misses) if ok else (hits, misses + 1)

    print("\n[3] DROPPED FACTS — must be REJECTED (loss, not inversion)")
    for src, cand in DROPPED:
        r = checker.check(Document(text=src), Document(text=cand))
        ok = _row("dropped", r.ok, False, r.summary())
        hits, misses = (hits + 1, misses) if ok else (hits, misses + 1)

    total = hits + misses
    print("\n" + "=" * 74)
    print(f"  {hits}/{total} correct")

    n_inv = len(INVERSIONS)
    caught = sum(
        1 for s, c, _ in INVERSIONS
        if not checker.check(Document(text=s), Document(text=c)).ok
    )
    clean = sum(
        1 for s, c in FAITHFUL
        if checker.check(Document(text=s), Document(text=c)).ok
    )
    print(f"  inversion recall     {caught}/{n_inv}   (embedding guard: 0/{n_inv})")
    print(f"  reword precision     {clean}/{len(FAITHFUL)}   (no false rejections)")

    if "--live" in sys.argv:
        live_rejection_rate()
    else:
        print("\n  (pass --live to also measure the rejection rate on real "
              "model regenerations)")
    print()
    return 0 if misses == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
