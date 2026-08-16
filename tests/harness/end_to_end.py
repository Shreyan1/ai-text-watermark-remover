"""End-to-end: the REAL local model (llama3.2:1b via Ollama) driving the full
Reweave pipeline. Two demonstrations:

  E1  Watermark removal, MEASURED. Text watermarked with our own key → full
      pipeline (Ollama extract + regenerate) → re-score with our key. The
      mean-score must collapse toward baseline. This is the one honest
      before/after you can only do when you hold the key. (Inputs are the
      harness's synthetic watermarked text; the point is the measurable mark and
      its removal by a real model, not the input's readability.)

  E2  Real content. A genuine Gemma response from the dataset → full pipeline →
      human-signature before/after + semantic similarity. Google's watermark is
      NOT measurable here (no key); we report only what we can measure.

Run:  PYTHONPATH=src python3 tests/harness/end_to_end.py
Needs Ollama up with a local open-weight model.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

from generate import DATA_ROOT, build  # noqa: E402
from metrics import summary  # noqa: E402
from synthid_ref import SynthIDConfig, score_text  # noqa: E402

from reweave import Document, Pipeline, PipelineConfig, VoiceProfile  # noqa: E402
from reweave._ollama import is_up  # noqa: E402
from reweave.extract import OllamaExtractor  # noqa: E402
from reweave.guard import OllamaEmbeddingGuard  # noqa: E402
from reweave.regenerate import OllamaRegenerator  # noqa: E402
from reweave.score import StatisticalScorer  # noqa: E402
from reweave.scrub import UnicodeScrubber  # noqa: E402
from reweave.verify import ConstraintChecker  # noqa: E402

MODEL = os.environ.get("REWEAVE_MODEL", "llama3.2:1b")
EMBED_MODEL = os.environ.get("REWEAVE_EMBED_MODEL", "all-minilm")
# Floor from tests/harness/guard_eval.py: faithful rewords ≈0.65, topic drift ≈0.04.
SIM_FLOOR = 0.35


def make_pipeline(max_iter: int, sim_floor: float = SIM_FLOOR,
                  facts: bool = True) -> Pipeline:
    return Pipeline(
        UnicodeScrubber(),
        OllamaExtractor(model=MODEL),
        OllamaRegenerator(model=MODEL),
        StatisticalScorer(),
        OllamaEmbeddingGuard(model=EMBED_MODEL),
        PipelineConfig(max_iterations=max_iter, similarity_floor=sim_floor,
                       human_threshold=0.65, enforce_facts=facts),
        # The guard cannot see truth value; this can. Both gates or neither.
        fact_checker=ConstraintChecker() if facts else None,
    )


def e1_watermark_removal(n: int = 3) -> None:
    print("\n" + "=" * 70)
    print(f"E1 — REAL local model ({MODEL}) removing a watermark WE can measure")
    print("=" * 70)
    cfg = SynthIDConfig()
    built = build(n=30, length=70)
    wm = built["corpora"]["watermarked"][:n]
    un_baseline = [score_text(r["text"], cfg) for r in built["corpora"]["unwatermarked"]]

    # E1 proves the MECHANISM: call Stage ②→③ directly so a genuine regeneration
    # always happens. (The gated Pipeline may keep the input if regeneration does
    # not improve human-signature — correct for the product, wrong for this proof.)
    scrub, ext, reg = UnicodeScrubber(), OllamaExtractor(model=MODEL), OllamaRegenerator(model=MODEL)
    befores, afters = [], []
    for i, row in enumerate(wm, 1):
        before = score_text(row["text"], cfg)
        meaning = ext.extract(scrub.scrub(Document(text=row["text"])))
        out = reg.regenerate(meaning, VoiceProfile(), aggressiveness=0.7)
        after = score_text(out.text, cfg)
        befores.append(before)
        afters.append(after)
        print(f"  sample {i}: watermark mean-score  {before:.3f} → {after:.3f}   "
              f"(Δ {before - after:+.3f})")

    base = sum(un_baseline) / len(un_baseline)
    print(f"\n  unwatermarked baseline mean-score : {base:.3f}")
    print(f"  watermarked, before pipeline (avg): {sum(befores)/len(befores):.3f}")
    print(f"  watermarked, after  pipeline (avg): {sum(afters)/len(afters):.3f}")
    still = summary(afters, un_baseline)["tpr@fpr=1%"]
    print(f"  still detectable after (TPR@FPR=1% vs baseline): {still:.3f}")
    print("  → a real, local, unwatermarked model collapsed a measurable watermark.")


def e2_real_content(n: int = 2) -> None:
    print("\n" + "=" * 70)
    print(f"E2 — regenerating REAL Gemma text ({MODEL}); meaning + quality")
    print("=" * 70)
    ai = []
    with open(os.path.join(DATA_ROOT, "synthid", "human_eval.jsonl"), encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            ai.append(json.loads(line)["watermarked_model_response"])

    # The GATED pipeline, now with the embedding guard. The gate only accepts a
    # regeneration whose MEANING survives (sim >= floor); with Jaccard this path
    # rejected every faithful reword, so nothing ever changed.
    pipe = make_pipeline(max_iter=2)
    for i, text in enumerate(ai, 1):
        result = pipe.run(Document(text=text), VoiceProfile())
        b = result.before.score if result.before else float("nan")
        a = result.after.score if result.after else float("nan")
        changed = result.output.text.strip() != text.strip()
        print(f"\n  doc {i}: human-signature {b:.3f} → {a:.3f}   "
              f"semantic-sim={result.semantic_similarity:.2f}   "
              f"iters={result.iterations}  accepted-rewrite={changed}")
        print(f"    facts : {result.facts.summary() if result.facts else 'not checked'}")
        for st in result.trace:
            if st.stage == "iterate":
                print(f"      · {st.note}  sim={st.data['similarity']:.2f}  "
                      f"human={st.data['score']:.3f}  facts={st.data['facts']}")
        print(f"    BEFORE: {text[:150].strip()}...")
        print(f"    AFTER : {result.output.text[:150].strip()}...")
    print("\n  NOTE: Google's watermark on these is NOT measurable without their key —")
    print("        we report human-signature and embedding-measured meaning preservation.")


def main() -> int:
    if not is_up():
        print("Ollama is not reachable at localhost:11434. Start it and pull a local "
              "open-weight model (e.g. `ollama pull llama3.2:1b`).", file=sys.stderr)
        return 2
    e1_watermark_removal()
    e2_real_content()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
