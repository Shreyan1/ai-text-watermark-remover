# Results — what the harness actually measured

Reproduce:

```bash
python3 dataset/fetch.py                             # get the data
PYTHONPATH=src python3 tests/harness/evaluate.py     # Track A + Track B
PYTHONPATH=src python3 tests/harness/guard_eval.py   # semantic guard comparison
PYTHONPATH=src python3 tests/harness/fact_eval.py --live  # fact gate (--live needs Ollama)
PYTHONPATH=src python3 tests/harness/end_to_end.py   # E1 + E2 (needs Ollama)
```

All numbers below are from a real run (word-level SynthID reference, m=6 tournament
layers, H=4, 120 sequences/class of length 70; local `llama3.2:1b` via Ollama).

## Track A — watermark ground truth (our own key)

**A1 — Detection.** Watermarked vs unwatermarked, mean-score detector:

| mean (watermarked) | mean (unwatermarked) | AUROC | TPR@FPR=1% |
|---|---|---|---|
| 0.713 | 0.504 | **1.000** | **1.000** |

The pure-Python SynthID reference embeds a cleanly detectable watermark. (Cleaner
than production — word-level tokens, modest depth — by design.)

**A2 — Removal.** Regenerate a fraction *p* of the token substrate; watermark decays:

| p (fraction regenerated) | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|---|
| still detected (TPR@1%) | 1.000 | 0.992 | 0.767 | 0.467 | 0.075 | 0.042 | **0.008** |

Regenerate ~half the tokens and detection is already near-dead. This is the
substrate attack working with zero knowledge of the scheme.

**A3 — Invariant I4.** Same data, two effort levels:

- light copy-edit (~10% of words changed): **TPR@1% = 0.992** → watermark *survives*
- full regeneration (~100% changed): **TPR@1% = 0.008** → watermark *gone*

Removal tracks how many token identities change. In-place editing changes few, so
it under-removes; regeneration changes all. That is why the architecture says
regenerate, never edit in place.

## Track B — human-signature scorer (real human vs real AI)

24 public-domain human passages (Darwin) vs 60 Gemma 7B responses.

Per-feature discriminative power (single-feature AUROC, human > AI):

| feature | AUROC | verdict |
|---|---|---|
| **burstiness** | **0.906** | strong — the one reliable signal without an LM |
| type-token ratio | 0.593 | weak positive |
| em-dash rate | 0.583 | weak |
| rule-of-three | 0.405 | ~noise |
| entity density | 0.315 | **anti-correlated** |
| numeral density | 0.308 | **anti-correlated** |
| paragraph CV | 0.008 | **strongly anti-correlated** |

Lesson: modern instruction-tuned models are *specific and well-structured*, so
"specificity/structure = human" is a false prior. After dropping the
anti-correlated features and weighting burstiness-forward, the composite scorer
reached **AUROC 0.876** (human 0.879 vs AI 0.656). Ranking separation is real;
a calibrated hard threshold still needs the perplexity backend and register-matched
data. This is consistent with the wider finding that statistical AI-detection is
unreliable.

**Abstention on short text.** Burstiness carries 45% of the weight and needs ≥2
sentences, so on a short passage the remaining features renormalise into an
inflated score — a one-line string full of AI tells scored 0.850 "human-like".
The scorer now abstains (`Verdict.UNCERTAIN`) when less than 60% of feature
weight is available, and says so. This is the selective-prediction mechanism from
the SynthID-Text paper (Supplementary C.8), and it matches Anthropic's own
statement that watermark detection degrades significantly on short passages: a
short sample simply does not carry enough signal to judge.

## Semantic guard — Jaccard vs embeddings

`tests/harness/guard_eval.py`. The gate needs a similarity that says YES to "same
meaning, different words" and NO to "different meaning". The decisive number is
the **margin** between them, because the floor has to sit in the gap.

| guard | faithful reword | different topic | margin | verdict |
|---|---|---|---|---|
| Jaccard (word overlap) | 0.233 | 0.032 | +0.200 | ✗ rejects faithful rewords — unusable |
| **Embedding (`all-minilm`)** | **0.650** | **0.042** | **+0.608** | ✓ floor at **0.35** separates cleanly |

Jaccard scores a correct reword at 0.233 — below any sane floor — because changing
the words *is* the job. That is why the gated pipeline previously accepted nothing.
The embedding guard triples the margin and makes the gate usable.

**Measured blind spot — negation and factual inversion:**

| pair (same topic, opposite truth) | similarity |
|---|---|
| "X are computers" vs "X are **not** computers" | 0.959 |
| "deployment succeeded" vs "deployment failed" | 0.898 |
| "revenue **increased** 40%" vs "**decreased** 40%" | 0.776 |

Sentence embeddings encode topic, not truth value. **The guard prevents topic
drift, not fact corruption** — a passing similarity does not mean the facts
survived. This is not hypothetical: the 1B model rewrote "Xbox and PlayStation
*are* computers" into "*aren't* computers" and still scored 0.83.

This is a property of the representation, not a bad choice of model. Embeddings
are built on the distributional hypothesis — words are learned from the contexts
they appear in — and a sentence and its negation share their contexts almost
perfectly, which makes such models "insensitive to negation and related phenomena
such as antonymy" ([arXiv:2507.12782](https://arxiv.org/abs/2507.12782); see also
[arXiv:2307.13989](https://arxiv.org/abs/2307.13989)). No floor fixes it. It needs
a second check that looks at what embeddings discard — which is the next section.

## Fact gate — closing the negation blind spot

`tests/harness/fact_eval.py`. `verify/constraint_checker.py` verifies three
things the guard cannot see: numerals, proper nouns, and per-claim polarity.
Negation handling follows **NegEx** (Chapman et al., *J. Biomedical Informatics*
34(5), 2001) — the standard rule-based negation detector — including the general
subset of its published trigger lexicon and, critically, its rule that
*pseudo*-negation takes precedence over negation.

**Recall on the exact pairs the embedding guard waved through:**

| pair | embedding sim | fact gate |
|---|---|---|
| "are computers" vs "are **not** computers" | 0.959 ✗ passed | ✅ rejected |
| "deployment **succeeded**" vs "**failed**" | 0.898 ✗ passed | ✅ rejected |
| "revenue **increased** 40%" vs "**decreased**" | 0.776 ✗ passed | ✅ rejected |
| "was **never** trained on…" vs "was trained on…" | — | ✅ rejected |
| "**All** 12 regions reported" vs "**None** of the 12" | — | ✅ rejected |

**Inversion recall 5/5, where the embedding guard scores 0/5.** Faithful rewords
4/4 accepted and dropped-fact cases 2/2 rejected — 11/11 on the curated set.

**Rejection rate on live regenerations** (`--live`, 8 real Gemma responses
rewritten by `llama3.2:1b`): **2/8**, both genuine fact loss — the 1B dropped
"Microsoft, Sony, Unity, Unreal, CPU" from one and the "PNG/TIF" format names
from another. Deliberately reported as a rejection rate with reasons, not as
precision: there are no ground-truth fidelity labels here, and a gate that
rejects everything is as useless as one that rejects nothing.

**Four false-positive classes were found and fixed by measuring on real text,**
not by inspection — each one would have made the gate reject legitimate work:

| false positive | cause | fix |
|---|---|---|
| doc contradicts *itself* (4/64 of corpus) | overlap coefficient scores a 1-word fragment ("Metaphor:") at 1.0 vs any sentence containing it | set cosine + greedy one-to-one alignment |
| common nouns as "names" (`considerations`, `engine`) | title-case headings make every word look proper | heading detection; colon ends a clause |
| `"Visit our website"` counted as a name | capital explained by the quote, not by being a name | quote-initial capitals need corroboration |
| markdown list ordinals as facts | `**1. Habitat loss:**` — bold markers hid the ordinal from the list-stripper | strip inline markup *before* leading markup |
| "not **just** about X, it's also Y" flagged as inversion | affirming construction read as negation | NegEx pseudo-negation precedence |

Self-consistency (a document checked against itself must always pass) went from
**4/64 failures to 0/64**. That invariant is what surfaced the alignment bug.

**Limit of the rules — and how it was closed.** The rule check is *lexical*: it
catches negation and listed antonym flips. A claim reversed with entirely
different words is invisible to it. Measured on a gap set of 8 such pairs
(`tests/harness/nli_eval.py`):

| checker | lexical inversions | **reworded inversions** | faithful rewrites kept |
|---|---|---|---|
| rules only | 3/3 | **0/8** | 8/8 |
| **rules + NLI** | 3/3 | **8/8** | 8/8 |
| NLI with *lexical* alignment | 3/3 | 2/8 | 8/8 |

The third row is the load-bearing detail. Aligning claims by word overlap fails
on exactly the cases NLI is for: "migration finished ahead of schedule" vs
"migration overran its deadline" share one content word, cosine 0.22, below any
floor — so the pair never reaches the model. **Embeddings do the alignment, NLI
does the judging.** Each instrument is used where it is strong: embeddings are
excellent at "are these about the same thing?" and blind only to polarity, which
is precisely the question NLI then answers.

## NLI backend — measured on real labelled data

The 8/8 above is a set we wrote ourselves and is worth little alone: it shows the
mechanism fires, not that the model is right. It is also trivially overfittable,
since the author of the examples is the author of the system. The counterexample
that forced a real benchmark — gemma3:4b labels

    "migration finished ahead of schedule" / "overran its deadline"   → CONTRADICTION ✓
    "migration finished ahead of its published schedule" /
        "overran the published schedule by several weeks"             → ENTAILMENT ✗

Same meaning, trivially different words, opposite verdict. So the backend is
benchmarked on **SNLI validation** (Bowman et al., EMNLP 2015 — human-annotated,
not ours), n=150, `gemma3:4b`, via `tests/harness/nli_benchmark.py`:

| metric | value |
|---|---|
| 3-class accuracy | **0.587** |
| contradiction precision | **0.923** |
| contradiction recall | **0.444** |
| contradiction F1 | 0.600 |
| unparseable replies | 4/150 |
| cost | 3.0 s/pair (local GPU) |

Confusion (gold → predicted):

| | entailment | neutral | contradiction |
|---|---|---|---|
| **entailment** | 41 | 8 | 0 |
| **neutral** | 22 | 23 | 2 |
| **contradiction** | 3 | 27 | 24 |

**This corrects the 8/8.** On real data NLI does not close the lexical gap — it
narrows it, from 0% coverage to ~44%, at high precision. The dominant error is
contradiction → neutral (27 cases): the model is conservative and defaults to
"no finding" rather than falsely accusing a rewrite. For a gate that blocks a
user's work that is the right failure direction, and precision (0.923) is the
metric that decides usability: it rarely destroys good work.

Both numbers are bounds, and neither alone is the answer. SNLI is *harder* than
our task — it is full of subtle neutral distinctions, while we only ever compare
a sentence to a rewrite of itself, a narrower and more contrastive distribution.
So 0.444 is a conservative lower bound here. The curated 8/8 is an upper bound.
We report both rather than the flattering one.

Improving it is a backend swap, not a redesign (`NLIBackend` is one method): a
trained cross-encoder on MNLI+VitaminC is the obvious next step, and
self-consistency voting (`votes>1`) trades cost for calibration.

**Ceiling, for context.** NLI-based factual consistency is not a solved problem
being approximated badly — it is open. SummaC's state of the art on its
six-dataset benchmark is **74.4% balanced accuracy**
([Laban et al., TACL 2022](https://aclanthology.org/2022.tacl-1.10/)), using
BERT-Large trained on MNLI+VitaminC. Notably VitaminC is built from Wikipedia
revisions where a small edit *flips* the label — contrastive supervision aimed at
exactly our failure case, and it is what lifts that configuration above plain
MNLI (72.8 → 74.4 on Conv).

## End-to-end — real local model (`llama3.2:1b`)

**E1 — watermark removal, measured.** Text watermarked with our key, regenerated
by the real local model, re-scored with our key:

| | watermark mean-score |
|---|---|
| unwatermarked baseline | 0.501 |
| watermarked, before | 0.708 |
| watermarked, after regeneration | **0.506** |
| still detectable (TPR@1%) | **0.000** |

A real, local, unwatermarked model collapsed a measurable watermark to baseline.

**E2 — real Gemma content, gated pipeline with BOTH gates (embedding + facts):**

| doc | human-signature | semantic-sim | facts | rewrite accepted? |
|---|---|---|---|---|
| 1 | 0.493 → **0.647** | 0.91 | preserved | ✅ yes |
| 2 | 0.462 → 0.462 | 1.00 | — | ❌ no — every candidate failed a gate |

Doc 1 previously reached 0.557 at sim 0.83 while **inverting a fact** ("are
computers" → "aren't computers"). With the fact gate added — and the extracted
`Constraints` now pinned into the regenerator prompt — the same document reaches
**0.647 at sim 0.91 with facts intact**. Telling the model what to keep improved
the rewrite; the gate is what makes that verifiable rather than hoped-for.

Doc 2 shows both gates working: iteration 1 was rejected for 2 inverted claims,
iteration 2 for meaning drift (sim 0.63), so the original was kept. Conservative,
and correct — a 1B model is a weak regenerator and output quality tracks model
quality.

## What is proven, and what is not

- **Proven:** the substrate attack removes a keyed watermark we can measure; a real
  local model does it end to end (0.708 → 0.506, still-detected 0.000); removal
  scales with fraction-of-tokens-changed (I4); burstiness is the one robust
  human/AI signal available without an LM; the embedding guard makes the gate
  usable where Jaccard could not; the fact gate catches 5/5 of the inversions the
  embedding guard misses while rejecting 0/4 faithful rewords, and rejects 2/8
  live regenerations for reasons that inspect as genuine fact loss.
- **Not proven here:** removal of *Google's* watermark (their key is unmeasurable —
  the real-world constraint); a production-grade regenerator (needs a stronger
  local model); **fact preservation beyond lexical inversion** — a claim reversed
  with entirely different words still passes, and closing that needs NLI, which
  is itself only ~74% balanced accuracy at the state of the art; a calibrated
  human/AI boundary (needs the perplexity backend).
