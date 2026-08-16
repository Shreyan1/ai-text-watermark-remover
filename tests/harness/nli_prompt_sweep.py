"""Why is contradiction recall only 0.444, and which prompt fixes it?

The SNLI benchmark's confusion matrix is unambiguous about the failure mode:

    gold=contradiction -> predicted neutral : 27
    gold=contradiction -> predicted entail  :  3
    gold=contradiction -> predicted contra  : 24

The model is not confused between contradiction and entailment. It is SHY: it
retreats to NEUTRAL. So the fix is not a better model, it is a prompt that stops
neutral from being the comfortable default.

Four variants, measured on the same cached SNLI rows so the comparison is exact:

  A baseline      current 3-class prompt
  B strict-neutral  defines NEUTRAL narrowly, forcing a commitment
  C few-shot      6 labelled examples, weighted toward hard contradictions
  D binary        decomposes into one yes/no question ("can both be true?"),
                  which is a strictly easier decision than a 3-way one

Run:  PYTHONPATH=src python3 tests/harness/nli_prompt_sweep.py [--n 100]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reweave._ollama import generate  # noqa: E402
from reweave.verify.ollama_nli import _ANSWER, _SYSTEM as BASELINE_SYSTEM  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / "dataset" / "nli" / "snli_validation.jsonl"

STRICT_NEUTRAL = (
    "You are a natural language inference classifier. Given a PREMISE and a "
    "HYPOTHESIS, decide:\n"
    "ENTAILMENT - if the premise is true, the hypothesis must also be true\n"
    "CONTRADICTION - the premise and the hypothesis cannot both be true of the "
    "same situation\n"
    "NEUTRAL - both could be true at once; the hypothesis just adds unrelated "
    "detail the premise does not settle\n\n"
    "NEUTRAL is the narrowest label, not the safe one. Before choosing it, ask: "
    "could both sentences describe the same scene at the same moment? If they "
    "could not, the answer is CONTRADICTION, even when no negation word appears "
    "and even when only one detail conflicts.\n\n"
    "End your reply with exactly:\nANSWER: <ENTAILMENT|CONTRADICTION|NEUTRAL>"
)

FEW_SHOT = STRICT_NEUTRAL + (
    "\n\nExamples:\n"
    "PREMISE: A man is playing a guitar on stage.\n"
    "HYPOTHESIS: A man is performing music.\nANSWER: ENTAILMENT\n\n"
    "PREMISE: A man is playing a guitar on stage.\n"
    "HYPOTHESIS: A man is asleep in bed.\nANSWER: CONTRADICTION\n\n"
    "PREMISE: A man is playing a guitar on stage.\n"
    "HYPOTHESIS: The man is playing his own songs.\nANSWER: NEUTRAL\n\n"
    "PREMISE: Two children run across a grassy field.\n"
    "HYPOTHESIS: The children are sitting still indoors.\nANSWER: CONTRADICTION\n\n"
    "PREMISE: Sales climbed steadily through the summer.\n"
    "HYPOTHESIS: Sales were disappointing all summer.\nANSWER: CONTRADICTION\n\n"
    "PREMISE: A woman in a red coat waits at a bus stop.\n"
    "HYPOTHESIS: A woman waits for the bus in the rain.\nANSWER: NEUTRAL\n"
)

BINARY = (
    "You compare two statements about the same situation.\n\n"
    "Answer ONE question: could both statements be true of the same situation "
    "at the same time?\n\n"
    "Answer NO if they conflict in any way - opposite outcome, opposite "
    "direction, incompatible detail, or one ruling out the other. A conflict "
    "counts even with no negation word present.\n"
    "Answer YES if they can coexist, including when the second merely adds "
    "detail the first does not settle.\n\n"
    "End your reply with exactly:\nANSWER: <YES|NO>"
)


def _label_3class(system: str, premise: str, hypothesis: str) -> str:
    raw = generate(f"PREMISE: {premise}\nHYPOTHESIS: {hypothesis}",
                   model=MODEL, temperature=0.0, num_predict=400,
                   system=system, think=False)
    m = None
    for m in _ANSWER.finditer(raw):
        pass
    return m.group(1).lower() if m else "neutral"


def _label_binary(premise: str, hypothesis: str) -> str:
    raw = generate(f"STATEMENT A: {premise}\nSTATEMENT B: {hypothesis}",
                   model=MODEL, temperature=0.0, num_predict=400,
                   system=BINARY, think=False)
    m = None
    for m in __import__("re").finditer(r"answer\s*:\s*\**\s*(yes|no)", raw, __import__("re").I):
        pass
    if not m:
        return "neutral"
    # NO = cannot coexist = contradiction. YES collapses entailment+neutral, which
    # is fine: the gate only ever acts on contradiction.
    return "contradiction" if m.group(1).lower() == "no" else "neutral"


MODEL = "gemma3:4b"


def evaluate(name: str, fn) -> None:
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines()][:N]
    tp = fp = fn_ = 0
    for r in rows:
        pred = fn(r["premise"], r["hypothesis"])
        gold = r["label"]
        if gold == "contradiction" and pred == "contradiction":
            tp += 1
        elif gold != "contradiction" and pred == "contradiction":
            fp += 1
        elif gold == "contradiction":
            fn_ += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"  {name:<16} precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  "
          f"(tp={tp} fp={fp} fn={fn_})", flush=True)


N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100

if __name__ == "__main__":
    print(f"\nNLI PROMPT SWEEP, contradiction class, SNLI n={N}, {MODEL}")
    print("=" * 74, flush=True)
    evaluate("A baseline", lambda p, h: _label_3class(BASELINE_SYSTEM, p, h))
    evaluate("B strict-neutral", lambda p, h: _label_3class(STRICT_NEUTRAL, p, h))
    evaluate("C few-shot", lambda p, h: _label_3class(FEW_SHOT, p, h))
    evaluate("D binary", _label_binary)
    print("\n  Recall is the target; precision is the budget. A variant that wins")
    print("  recall by collapsing precision has just moved the failure, not fixed it.\n")
