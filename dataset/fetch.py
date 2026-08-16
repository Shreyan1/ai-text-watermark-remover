"""Fetch/rebuild the datasets. Stdlib only.

  * dataset/synthid/human_eval.jsonl  — 3,000 watermarked vs unwatermarked Gemma 7B
    responses, from google-deepmind/synthid-text (the paper's human-eval data).
  * dataset/seed_human.jsonl          — genuine human prose (public domain: Darwin,
    Origin of Species, 1859) as multi-paragraph samples for the Track B scorer test.

Run:  python3 dataset/fetch.py
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTHID_URL = "https://raw.githubusercontent.com/google-deepmind/synthid-text/main/data/human_eval.jsonl"
DARWIN_URL = "https://www.gutenberg.org/cache/epub/1228/pg1228.txt"


def fetch_synthid() -> None:
    out_dir = os.path.join(HERE, "synthid")
    os.makedirs(out_dir, exist_ok=True)
    data = urllib.request.urlopen(SYNTHID_URL, timeout=60).read()
    path = os.path.join(out_dir, "human_eval.jsonl")
    with open(path, "wb") as fh:
        fh.write(data)
    print(f"synthid/human_eval.jsonl  ({len(data.splitlines())} records, {len(data)} bytes)")


def build_human_seed(n: int = 24) -> None:
    raw = urllib.request.urlopen(DARWIN_URL, timeout=60).read().decode("utf-8", "replace")
    start, end = raw.find("*** START"), raw.find("*** END")
    body = raw[raw.find("\n", start) + 1:end]
    paras = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
    good = [p for p in paras if 250 <= len(p) <= 700 and p.count(".") >= 2
            and p[:1].isupper() and "CHAPTER" not in p and "_" not in p[:3]]
    samples, i = [], 20
    while i + 1 < len(good) and len(samples) < n:
        samples.append(good[i] + "\n\n" + good[i + 1])
        i += 2
    path = os.path.join(HERE, "seed_human.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps({
                "text": s, "label": "human",
                "source": "Darwin, On the Origin of Species (1859), public domain, Project Gutenberg #1228",
            }) + "\n")
    print(f"seed_human.jsonl          ({len(samples)} multi-paragraph human samples)")


if __name__ == "__main__":
    fetch_synthid()
    build_human_seed()
