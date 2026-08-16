# Self-watermark test harness

The only way to measure removal of a *keyed* watermark honestly is to hold the
key yourself. This harness builds that ground truth with a pure-Python SynthID
reference — no ML dependencies. Measured results: **[../../RESULTS.md](../../RESULTS.md)**.

## Files

| File | What |
|---|---|
| `synthid_ref.py` | Faithful pure-Python SynthID-Text: g-values, sliding-window seed, Tournament sampling, mean-score detector. Tokens are words, so it can score *any* text. |
| `wordlm.py` | Word-level bigram LM (the `p_LM` Tournament draws from), trained on real English from the dataset. |
| `metrics.py` | AUROC and TPR@FPR (the paper's headline metrics), pure Python. |
| `generate.py` | Builds watermarked vs unwatermarked corpora with keys we hold. |
| `attacks.py` | The substrate attack: regenerate a fraction `p` of tokens. |
| `evaluate.py` | Track A (watermark detection + removal + I4) and Track B (human vs AI scorer). |
| `end_to_end.py` | E1 + E2: the **real** local model (Ollama) driving removal and regeneration. |

## Run

```bash
python3 dataset/fetch.py                              # data (once)
PYTHONPATH=src python3 tests/harness/evaluate.py      # Track A + Track B
PYTHONPATH=src python3 tests/harness/end_to_end.py    # E1 + E2 (needs Ollama up)
```

## What ground truth means here

You cannot verify that a real Claude/Gemini watermark is gone without the issuer's
secret key. So we watermark with **our** key using `synthid_ref.py`, then measure
the mean-score before and after regeneration. That before/after is the honest
signal. Everything involving third-party (Google) text can only report
human-signature and meaning preservation — never keyed-watermark removal.

## Deliberate simplifications (and why they're honest)

- **Word-level tokens**, not subword. Buys the ability to score arbitrary text.
  Makes the signal *cleaner* than production, not weaker — we measure mechanism.
- **Modest tournament depth** (`m=6`, production ≈30). Keeps `2^m` candidate
  sampling tractable; the mechanism and its removal are identical in shape.
- **Bigram `p_LM`.** Enough entropy to embed a watermark; the removal result does
  not depend on LM sophistication.

These are stated in `synthid_ref.py` and `dataset/README.md` so nobody mistakes
the harness's clean separation for Google's real operating point.
