"""Benchmark the NLI backend on REAL labelled data, not our own examples.

Why this file exists: `nli_eval.py` scores 8/8 on a gap set we wrote ourselves,
and that number is worth very little. A hand-written set measures whether the
mechanism fires, not whether the model is right — and it is trivially
overfittable, because the author of the examples is the author of the system.
The counterexample that forced this file: gemma3:4b correctly labels

    "The migration finished ahead of schedule."
    "The migration overran its deadline by several weeks."   → CONTRADICTION

but on a barely-rephrased version of the same claim

    "The migration finished ahead of its published schedule."
    "The migration overran the published schedule by several weeks."

it answers ENTAILMENT. Same meaning, different words, opposite verdict. One
curated set cannot see that; a real benchmark can.

DATA — SNLI validation split (Bowman et al., EMNLP 2015), fetched from the public
HuggingFace datasets-server. Human-annotated, 3-class, and NOT written by us.
Rows labelled -1 (no annotator consensus) are dropped, as the dataset intends.

REPORTED — overall 3-class accuracy, plus precision/recall/F1 for the
CONTRADICTION class specifically, because that is the only class the fact gate
acts on. A backend with good overall accuracy but poor contradiction recall would
be useless here, and an average would hide it.

Run:  PYTHONPATH=src python3 tests/harness/nli_benchmark.py [--n 150] [--model gemma3:4b]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reweave.verify import OllamaNLIBackend  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / "dataset" / "nli" / "snli_validation.jsonl"
API = ("https://datasets-server.huggingface.co/rows"
       "?dataset=stanfordnlp/snli&config=plain_text&split=validation")
LABELS = ("entailment", "neutral", "contradiction")


def fetch(n: int) -> list[dict]:
    if CACHE.exists():
        rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines()]
        if len(rows) >= n:
            return rows[:n]

    rows: list[dict] = []
    offset = 0
    while len(rows) < n:
        url = f"{API}&offset={offset}&length=100"
        with urllib.request.urlopen(url, timeout=60) as resp:
            batch = json.loads(resp.read())["rows"]
        if not batch:
            break
        for r in batch:
            row = r["row"]
            if row["label"] in (0, 1, 2):  # -1 = no annotator consensus
                rows.append({"premise": row["premise"],
                             "hypothesis": row["hypothesis"],
                             "label": LABELS[row["label"]]})
        offset += 100

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows[:n]


def main() -> int:
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 150
    model = (sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv
             else os.environ.get("REWEAVE_NLI_MODEL", "gemma3:4b"))

    rows = fetch(n)
    backend = OllamaNLIBackend(model=model)

    print(f"\nNLI BACKEND BENCHMARK — SNLI validation, n={len(rows)}, model={model}")
    print("=" * 74)

    tp = fp = fn = correct = 0
    confusion: dict[tuple[str, str], int] = {}
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        pred = backend.predict(r["premise"], r["hypothesis"]).label
        gold = r["label"]
        confusion[(gold, pred)] = confusion.get((gold, pred), 0) + 1
        correct += pred == gold
        if gold == "contradiction" and pred == "contradiction":
            tp += 1
        elif gold != "contradiction" and pred == "contradiction":
            fp += 1
        elif gold == "contradiction" and pred != "contradiction":
            fn += 1
        if i % 25 == 0:
            print(f"  … {i}/{len(rows)}  running acc={correct / i:.3f}")

    elapsed = time.time() - t0
    acc = correct / len(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("\n  3-class accuracy      : {:.3f}".format(acc))
    print("  CONTRADICTION precision: {:.3f}   (of what we reject, how much deserved it)".format(prec))
    print("  CONTRADICTION recall   : {:.3f}   (of real inversions, how many we catch)".format(rec))
    print("  CONTRADICTION F1       : {:.3f}".format(f1))
    print(f"  unparseable replies    : {backend.parse_failures}/{backend.calls}")
    print(f"  cost                   : {elapsed:.1f}s total, {elapsed / len(rows):.2f}s/pair")

    print("\n  confusion (gold → predicted):")
    print(f"    {'':>14} " + " ".join(f"{p[:5]:>7}" for p in LABELS))
    for g in LABELS:
        cells = " ".join(f"{confusion.get((g, p), 0):>7}" for p in LABELS)
        print(f"    {g:>14} {cells}")

    print("\n  Read precision as the false-alarm rate on honest rewrites, and recall")
    print("  as coverage of real inversions. For a gate that blocks a user's work,")
    print("  precision is the one that decides whether the tool is usable.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
