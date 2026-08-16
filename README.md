# AI Text Watermark Remover

**Remove statistical text watermarks — SynthID-Text, KGW, Unigram — by rebuilding
text from its meaning instead of editing its surface.** Scheme-blind, runs
entirely on local open-weight models, and every claim below is a number you can
reproduce.

> Watermark removal here is a *side effect*, not a trick. The system regenerates
> the token sequence from a meaning representation. Every token-level watermark
> lives in that token sequence — so it does not survive, whether or not we have
> ever heard of the scheme.

```bash
git clone https://github.com/Shreyan1/ai-text-watermark-remover
cd ai-text-watermark-remover && pip install -e .
reweave scrub input.txt                 # strip invisible characters + homoglyphs
reweave score input.txt                 # how AI-uniform does this read?
reweave facts original.txt rewrite.txt  # did a rewrite keep the facts? (--nli for more)
```

---

## Honest scope — read this before the benchmarks

| Ask | Does it? | Evidence |
|---|---|---|
| Remove a watermark **we can measure** | **Yes** | 0.708 → 0.506 vs 0.501 baseline; detection TPR 1.000 → 0.000 |
| Strip invisible characters, homoglyphs, curly quotes | **Yes** | deterministic, zero-dependency |
| Rewrite AI-flat text toward human variance | **Yes** | human-signature 0.493 → 0.647 with facts intact |
| Prove removal of **Google's / Anthropic's live watermark** | **No** | their key is secret, so nobody outside can measure it — including us |
| Detect *which* watermark a text carries, keyless | **No** | cryptographically impossible; we refuse to fake it |
| Guarantee text is "undetectable" | **No** | anyone claiming this is selling you something |

The fourth row is the one that separates this from the "100% undetectable"
tool market. **Keyed watermark removal is unverifiable from outside**, so instead
of asserting it, we implement SynthID-Text ourselves, watermark text with *our*
key, and measure removal against a detector we hold. That is the only honest
before/after that exists.

---

## Benchmarks

Every number is produced by a script in this repo. Nothing is quoted from a
vendor page or estimated.

```bash
python3 dataset/fetch.py
PYTHONPATH=src python3 tests/harness/evaluate.py        # watermark + scorer
PYTHONPATH=src python3 tests/harness/guard_eval.py      # semantic guard
PYTHONPATH=src python3 tests/harness/fact_eval.py --live # fact gate
PYTHONPATH=src python3 tests/harness/nli_benchmark.py   # NLI backend vs SNLI
PYTHONPATH=src python3 tests/harness/end_to_end.py      # full pipeline
```

### 1. Watermark detection — is the ground truth real?

Our pure-Python SynthID-Text reference (tournament sampling, H=4, m=6),
120 watermarked vs 120 unwatermarked sequences:

| metric | value |
|---|---|
| mean score, watermarked | 0.713 |
| mean score, unwatermarked | 0.504 |
| **AUROC** | **1.000** |
| **TPR @ FPR=1%** | **1.000** |

A perfectly detectable watermark — which is what makes the removal number below
mean something.

### 2. Watermark removal — how much regeneration is enough?

Regenerating a fraction *p* of the token substrate:

| p regenerated | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | **1.0** |
|---|---|---|---|---|---|---|---|
| still detected (TPR@1%) | 1.000 | 0.992 | 0.767 | 0.467 | 0.075 | 0.042 | **0.008** |

Detection is near-dead by the halfway point, with **zero knowledge of the
scheme**. This is the substrate attack.

### 3. Editing vs regenerating (why in-place editing fails)

| approach | words changed | still detected |
|---|---|---|
| light copy-edit | ~10% | **0.992** — watermark survives |
| full regeneration | ~100% | **0.008** — watermark gone |

Removal tracks *how many token identities change*. Synonym-swapping tools change
few, so they under-remove. This is why the architecture regenerates and never
edits in place.

### 4. End-to-end with a real local model (`llama3.2:1b`)

| | watermark mean-score |
|---|---|
| unwatermarked baseline | 0.501 |
| watermarked, before | 0.708 |
| watermarked, after regeneration | **0.506** |
| **still detectable (TPR@1%)** | **0.000** |

A real, local, unwatermarked model collapsed a measurable watermark to baseline.

### 5. Human-signature scorer (real human vs real AI text)

24 public-domain human passages vs 60 Gemma 7B responses. Per-feature
discriminative power:

| feature | AUROC | verdict |
|---|---|---|
| **burstiness** | **0.906** | the one reliable signal without an LM |
| type-token ratio | 0.593 | weak |
| em-dash rate | 0.583 | weak |
| rule-of-three | 0.405 | noise |
| entity density | 0.315 | **anti-correlated** |
| numeral density | 0.308 | **anti-correlated** |
| paragraph CV | 0.008 | **strongly anti-correlated** |

**Composite AUROC 0.876.** The anti-correlated rows are a finding, not a bug:
modern instruction-tuned models are *specific and well-structured*, so
"specificity = human" is a false prior that most detector heuristics still
encode. Those three features were dropped from the score.

The scorer **abstains** when under 60% of feature weight is available — short
text cannot support burstiness, and a one-line AI-tell-laden string was scoring
0.850 "human-like" before this was added.

### 6. Meaning preservation — semantic guard

The gate must accept "same meaning, different words" and reject "different
meaning". The number that matters is the **margin**:

| guard | faithful reword | different topic | margin |
|---|---|---|---|
| Jaccard (word overlap) | 0.233 | 0.032 | +0.200 ✗ unusable |
| **Embeddings (`all-minilm`)** | **0.650** | **0.042** | **+0.608** ✓ |

Word-overlap similarity rejects a *correct* rewrite at 0.233, because changing
the words is the job. With Jaccard the gate accepted nothing at all.

### 7. Fact preservation — where embeddings fail

Sentence embeddings encode topic, not truth value. Measured:

| pair (same topic, opposite truth) | embedding similarity |
|---|---|
| "X **are** computers" vs "X are **not** computers" | 0.959 |
| "deployment **succeeded**" vs "deployment **failed**" | 0.898 |
| "revenue **increased** 40%" vs "**decreased** 40%" | 0.776 |

All three sail past any usable floor. This is a documented property of
distributional representations ([arXiv:2507.12782](https://arxiv.org/abs/2507.12782)),
not a bad model choice — so the fix is a second gate, not a better threshold.

**Fact gate results** (rules: numerals, proper nouns, NegEx-based negation,
antonym polarity):

| test (curated set, n=8 — see §8 for real data) | rules | + NLI |
|---|---|---|
| lexical inversions caught | **3/3** | 3/3 |
| **reworded** inversions caught | **0/8** | **8/8** |
| faithful rewrites *not* falsely rejected | 8/8 | 8/8 |

Rules cannot see a reversal written in different words ("sales climbed" → "sales
were disappointing") — that is what NLI adds. Alignment must be embedding-based:
with lexical alignment the reworded pairs never reach the model (8/8 → **2/8**).

On 8 live regenerations of real Gemma text, the gate rejected 2 — both genuine
fact loss (the 1B model dropped "Microsoft, Sony, Unity, Unreal, CPU" from one).

### 8. NLI backend on real labelled data

The 8/8 above is our own curated set and is worth little on its own — it measures
whether the mechanism fires, not whether the model is right. Measured on **SNLI
validation** (Bowman et al., EMNLP 2015 — human-annotated, not written by us),
n=150, `gemma3:4b`:

| metric | value | what it means here |
|---|---|---|
| 3-class accuracy | 0.587 | vs ~0.88 human, ~0.92 SOTA — a 4B general model is not a trained NLI head |
| **contradiction precision** | **0.923** | when it rejects a rewrite, it is right 92% of the time |
| **contradiction recall** | **0.444** | it catches under half of real contradictions |
| unparseable replies | 4/150 | counted, never silently treated as "no finding" |
| cost | 3.0s/pair | on GPU, local |

Confusion (gold → predicted):

| | entail | neutral | contra |
|---|---|---|---|
| **entailment** | 41 | 8 | 0 |
| **neutral** | 22 | 23 | 2 |
| **contradiction** | 3 | 27 | **24** |

**This corrects the impression the 8/8 gives.** On real data the NLI gate does
*not* close the lexical gap — it narrows it, from 0% coverage to roughly 44%, and
it does so at high precision. The dominant error is contradiction → neutral (27
cases): the model is *conservative*, defaulting to "no finding" rather than
falsely accusing a rewrite.

For a gate that blocks a user's work that is the right direction to fail, and
precision is the metric that decides usability — 0.923 means it rarely destroys
good work. But anyone reading "8/8" as "fact corruption is solved" would be
wrong, and the honest summary is:

> **Rules catch what they catch, deterministically. NLI adds partial coverage of
> reworded reversal at low false-alarm cost. Neither is a guarantee.**

Two caveats in both directions. SNLI is *harder* than our actual task — it is
full of subtle neutral distinctions, whereas we only ever compare a sentence to a
rewrite of itself, a narrower and more contrastive distribution. So 0.444 is a
conservative lower bound for this use, not a direct measure of it. Equally, the
curated 8/8 is an upper bound. The truth is between, and we report both rather
than picking the flattering one.

---

## How it works

```
INPUT ─▶ ① Scrub ─▶ ② Extract ─▶ ③ Regenerate ─▶ ④ Score ─▶ ⑤ Gate ─▶ OUTPUT
        (code)      (meaning)     (LOCAL LLM)      (stats)     │
                                                               │
                                        ┌──────────────────────┴────────────┐
                                  guard: same topic?        facts: same claims?
                                  (embeddings)              (rules + NLI)
```

**Stage ③ must use a local, unwatermarked, open-weight model.** Paraphrasing
Claude's output *with Claude* strips the old mark and stamps a fresh one — the
Self-Watermark Trap. `Pipeline.__init__` refuses a regenerator not declared
`is_unwatermarked`, and the Ollama adapter refuses any model whose name contains
`cloud`.

Full design contract: **[ARCHITECTURE.md](ARCHITECTURE.md)** ·
diagrams: **[docs/diagrams.md](docs/diagrams.md)** ·
all numbers: **[RESULTS.md](RESULTS.md)**

---

## Install

Core installs with **zero dependencies** — that is the point; a dependency-free
core cannot rot:

```bash
pip install -e .
```

The import package and CLI are called **`reweave`**: the system does not erase a
mark, it *re-weaves* the text — discarding the token sequence where every
watermark lives and weaving a new one from the meaning. (Install name and import
name differing is idiomatic; cf. `beautifulsoup4` → `bs4`.)

Local models via [Ollama](https://ollama.com) for the model-backed stages:

```bash
ollama pull llama3.2:1b   # regeneration + extraction
ollama pull all-minilm    # semantic guard embeddings (45MB)
ollama pull gemma3:4b     # NLI fact checking (optional)
```

---

## What is proven, and what is not

**Proven here:** the substrate attack removes a keyed watermark we can measure;
a real local model does it end to end; removal scales with fraction-of-tokens
changed; burstiness is the one robust human/AI signal available without an LM;
the embedding guard makes the gate usable where word overlap could not; the rule
checker catches lexical inversion deterministically with no false alarms.

**Not proven here:**

- **Removal of Google's or Anthropic's live watermark.** Their keys are secret,
  so nobody outside can measure it — and any project claiming otherwise has not
  measured it either.
- **Fact preservation as a guarantee.** NLI narrows the reworded-reversal gap
  (0% → ~44% recall at 92% precision), it does not close it. A rewrite can still
  corrupt meaning in ways nothing here detects.
- **A production-grade regenerator.** A 1B model is weak; output quality tracks
  model quality directly.
- **A calibrated human/AI decision boundary.** The scorer separates (AUROC 0.876)
  but a hard threshold needs the perplexity backend and register-matched data.

If you need one sentence: **this removes watermarks it can measure, and is honest
about the ones it cannot.**

---

## A note on use

A genuine rewrite into your own voice *is* authorship — that is the legitimate
use, and it is what this is built for: research on watermark robustness, and
writing that sounds like you.

Using it to claim human authorship where disclosure is legally or academically
required is deception, and that is on the operator, not the tool. Under the
EU AI Act Article 50(2), providers must mark synthetic content; stripping such a
mark to misrepresent provenance may carry legal consequences. The pipeline can
regenerate a token sequence. It cannot write the one line only you could write.

Apache-2.0.
