# dataset/

Data for validating the pipeline. Rebuild everything with `python3 dataset/fetch.py`.

## Contents

| Path | What | Provenance |
|---|---|---|
| `synthid/human_eval.jsonl` | 3,000 records: `watermarked_model_response` vs `unwatermarked_model_response` (Gemma 7B), plus preference ratings | [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text), the Nature paper's human-eval data. *gitignored (8.7 MB); fetch to rebuild.* |
| `seed_human.jsonl` | 24 multi-paragraph human prose samples | Darwin, *On the Origin of Species* (1859), public domain, Project Gutenberg #1228 |
| `generated/` | Harness-produced watermarked/unwatermarked corpora + `report.json` | Built by the harness; *gitignored.* |

## Why these

- **Both fields of `human_eval.jsonl` are AI** (watermarked / unwatermarked Gemma).
  We do **not** hold Google's key, so their real watermark is **not measurable**, exactly the real-world constraint. This data serves as (a) the AI side of the
  Track B scorer test and (b) real content for the end-to-end meaning test, and
  (c) the English corpus the harness word-LM is trained on.
- **`seed_human.jsonl` is genuine human text** for the Track B human side. Register
  differs from ELI5 (essay vs Q&A), a documented confound, not a calibrated
  boundary. For a register-matched test, substitute human ELI5 answers.

## Ground truth lives in the harness: not here

Because no third party's keyed watermark is measurable without their key, the only
clean before/after comes from watermarking with **our own** key. That is what
`tests/harness/` does with a pure-Python SynthID reference. See `../RESULTS.md`.
