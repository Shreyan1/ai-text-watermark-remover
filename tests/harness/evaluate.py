"""The harness scorecard. Two tracks:

  Track A — watermark ground truth (keys we hold):
    A1  detection: do watermarked sequences separate from unwatermarked?
    A2  removal:   does regenerating the token substrate collapse the score?
    A3  I4 proof:  full regeneration vs light in-place editing.

  Track B — human vs AI, for the human-signature scorer:
    real AI text (Gemma, from the dataset) vs real human text (public-domain).

Run:  PYTHONPATH=src python3 tests/harness/evaluate.py
"""

from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # harness modules
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))  # reweave

from attacks import paraphrase_remove  # noqa: E402
from generate import DATA_ROOT, build  # noqa: E402
from metrics import summary  # noqa: E402
from synthid_ref import mean_score  # noqa: E402

from reweave.core.types import Document  # noqa: E402
from reweave.score import StatisticalScorer  # noqa: E402


def _fmt(d: dict) -> str:
    return (f"mean_pos={d['mean_pos']:.3f} mean_neg={d['mean_neg']:.3f} "
            f"AUROC={d['auroc']:.3f} TPR@FPR=1%={d['tpr@fpr=1%']:.3f} "
            f"(n={d['n_pos']}/{d['n_neg']})")


def track_a(n: int = 120, length: int = 70) -> dict:
    print("\n" + "=" * 70)
    print("TRACK A — watermark ground truth (pure-Python SynthID, keys we hold)")
    print("=" * 70)
    built = build(n=n, length=length)
    lm, cfg, corpora = built["lm"], built["cfg"], built["corpora"]
    print(f"LM vocab={lm.vocab_size}  m_layers={cfg.m_layers}  H={cfg.H}  "
          f"len={length}  n={n}/class")

    wm = [r["words"] for r in corpora["watermarked"]]
    un = [r["words"] for r in corpora["unwatermarked"]]
    wm_scores = [mean_score(x, cfg) for x in wm]
    un_scores = [mean_score(x, cfg) for x in un]

    a1 = summary(wm_scores, un_scores)
    print("\n[A1] Detection (watermarked vs unwatermarked):")
    print("     " + _fmt(a1))

    print("\n[A2] Removal — regenerate a fraction p of the token substrate:")
    rng = random.Random(101)
    curve = []
    for p in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        removed = [paraphrase_remove(x, p, lm, rng) for x in wm]
        s = [mean_score(x, cfg) for x in removed]
        mean_s = sum(s) / len(s)
        tpr = summary(s, un_scores)["tpr@fpr=1%"]
        curve.append({"p": p, "mean_score": mean_s, "tpr@fpr=1%": tpr})
        span = a1["mean_pos"] - 0.5 + 1e-9
        bar = "#" * max(0, int((mean_s - 0.5) / span * 40))
        print(f"     p={p:>3}  mean_wm_score={mean_s:.3f}  still-detected(TPR@1%)={tpr:.3f}  {bar}")

    print("\n[A3] Invariant I4 — removal tracks the FRACTION of token identities changed:")
    light = next(c for c in curve if c["p"] == 0.1)
    heavy = next(c for c in curve if c["p"] == 1.0)
    print(f"     light copy-edit  (~10% of words changed): TPR@1%={light['tpr@fpr=1%']:.3f}  → watermark SURVIVES")
    print(f"     full regeneration (~100% changed):        TPR@1%={heavy['tpr@fpr=1%']:.3f}  → watermark GONE")
    print("     → In-place edits change few tokens, so they under-remove. Regenerating")
    print("       the whole surface changes every token. That is why I4 says regenerate,")
    print("       never edit in place.")

    return {"a1": a1, "removal_curve": curve}


def track_b(n_ai: int = 60) -> dict:
    print("\n" + "=" * 70)
    print("TRACK B — human-signature scorer on real human vs real AI text")
    print("=" * 70)
    scorer = StatisticalScorer()

    human_path = os.path.join(DATA_ROOT, "seed_human.jsonl")
    humans = [json.loads(l)["text"] for l in open(human_path, encoding="utf-8")]

    ai: list[str] = []
    with open(os.path.join(DATA_ROOT, "synthid", "human_eval.jsonl"), encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n_ai:
                break
            ai.append(json.loads(line)["unwatermarked_model_response"])

    h_scores = [scorer.score(Document(text=t)).score for t in humans]
    a_scores = [scorer.score(Document(text=t)).score for t in ai]
    b = summary(h_scores, a_scores)
    print(f"human (public-domain, n={len(humans)}): mean human-signature = {b['mean_pos']:.3f}")
    print(f"AI    (Gemma 7B,        n={len(ai)}): mean human-signature = {b['mean_neg']:.3f}")
    print(f"separation: AUROC={b['auroc']:.3f}  TPR@FPR=1%={b['tpr@fpr=1%']:.3f}")
    print("NOTE: register differs (essay vs ELI5); this validates the scorer's")
    print("      direction, not a calibrated human/AI decision boundary.")
    return {"b": b}


def main() -> int:
    a = track_a()
    b = track_b()
    report = {"track_a": a, "track_b": b}
    out = os.path.join(DATA_ROOT, "generated", "report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
