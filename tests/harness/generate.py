"""Build ground-truth corpora: watermarked vs unwatermarked token sequences,
using the pure-Python SynthID reference over a word LM trained on real English.
"""

from __future__ import annotations

import json
import os
import random

from synthid_ref import SynthIDConfig, context_seed, tournament_sample
from wordlm import WordLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "dataset"))
GEN_DIR = os.path.join(DATA_ROOT, "generated")


def load_corpus_texts(limit: int = 1500) -> list[str]:
    """Real English to train the LM: the unwatermarked Gemma responses."""
    path = os.path.join(DATA_ROOT, "synthid", "human_eval.jsonl")
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            rec = json.loads(line)
            texts.append(rec["unwatermarked_model_response"])
    return texts


def generate_sequence(lm: WordLM, cfg: SynthIDConfig, length: int,
                      rng: random.Random, watermarked: bool) -> list[str]:
    words: list[str] = [lm.start_word(rng)]
    for _ in range(length - 1):
        prev = words[-1]
        if watermarked:
            seed = context_seed(words, cfg.key, cfg.H)
            dist_words, dist_wts = lm.next_dist_words(prev)
            words.append(tournament_sample(dist_words, dist_wts, seed, cfg, rng))
        else:
            words.append(lm.sample_next_word(prev, rng))
    return words


def build(n: int = 120, length: int = 70, cfg: SynthIDConfig | None = None,
          seed: int = 7) -> dict:
    cfg = cfg or SynthIDConfig()
    rng = random.Random(seed)
    lm = WordLM().fit(load_corpus_texts())

    os.makedirs(GEN_DIR, exist_ok=True)
    corpora = {}
    for label, wm in (("watermarked", True), ("unwatermarked", False)):
        rows = []
        for _ in range(n):
            words = generate_sequence(lm, cfg, length, rng, watermarked=wm)
            rows.append({"words": words, "text": " ".join(words)})
        path = os.path.join(GEN_DIR, f"{label}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        corpora[label] = rows
    return {"lm": lm, "cfg": cfg, "corpora": corpora}


if __name__ == "__main__":
    out = build()
    print("vocab:", out["lm"].vocab_size)
    for k, v in out["corpora"].items():
        print(k, len(v), "sequences ->", os.path.join(GEN_DIR, f"{k}.jsonl"))
