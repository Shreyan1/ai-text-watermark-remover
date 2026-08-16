# AI Text Watermark Remover

**Remove statistical text watermarks (SynthID-Text, KGW, Unigram) by rebuilding
text from its meaning instead of editing its surface.** Scheme-blind, runs
entirely on local open-weight models, and every claim below is a number you can
reproduce.

Watermark removal here is a side effect, not a trick. The system regenerates the
token sequence from a meaning representation. Every token-level watermark lives
in that token sequence, so it does not survive, whether or not we have ever heard
of the scheme.

```bash
git clone https://github.com/Shreyan1/ai-text-watermark-remover
cd ai-text-watermark-remover && pip install -e .

reweave meta  --dry-run answer.md   # what provenance does this file carry?
reweave meta  --rename   answer.md  # strip it, and the vendor name in the filename
reweave scrub answer.md             # strip invisible chars and homoglyphs
reweave score answer.md             # how AI-uniform does this read?
reweave facts before.md after.md    # did a rewrite keep the facts?

reweave fix   --rename   answer.md  # all of the above, looped until the file stops changing
```

Start with `meta --dry-run`. On a file saved from a chat UI it usually finds the
source URL sitting in an extended attribute, untouched by anything you do to the
text. `reweave fix` runs the whole sequence in one bounded loop; it is fully
offline unless you add `--regenerate`, which brings in a local model to rewrite
the prose under the fact gate.

---

## Honest scope, before the benchmarks

| Ask | Does it? | Evidence |
|---|---|---|
| Remove a watermark **we can measure** | **Yes** | 0.708 to 0.506, vs 0.501 baseline; detection TPR 1.000 to 0.000 |
| Strip file provenance (source URL, generator tags) | **Yes** | 6 of 7 traces on a simulated chat download; the 7th is reported |
| Strip invisible characters, homoglyphs, curly quotes | **Yes** | deterministic, zero dependencies |
| Rewrite AI-flat text toward human variance | **Yes** | human-signature 0.493 to 0.647 with facts intact |
| Prove removal of a **live vendor watermark** | **No** | their key is secret, so nobody outside can measure it, us included |
| Detect *which* watermark a text carries, keyless | **No** | cryptographically impossible; we refuse to fake it |
| Guarantee text is "undetectable" | **No** | anyone claiming this is selling you something |

Row five is what separates this from the "100% undetectable" tool market. Keyed
watermark removal is unverifiable from outside, so instead of asserting it, we
implement SynthID-Text ourselves, watermark text with *our* key, and measure
removal against a detector we hold. That is the only honest before-and-after
that exists.

---

## Runs offline, and never re-watermarks

Two guarantees, both enforced by tests rather than promised in prose.

**The deterministic core needs no model and no network.** Stripping metadata,
scrubbing characters, scoring, and the default fact check all run on the standard
library alone. `tests/test_offline.py` disables every network socket and then
runs each of them, so a stray dependency on a hosted default fails there instead
of in your offline session.

| command | needs a model? | what it uses |
|---|---|---|
| `reweave meta` | no | filesystem attributes, stdlib |
| `reweave scrub` | no | `unicodedata`, stdlib |
| `reweave score` | no | `re`, `statistics`, stdlib |
| `reweave facts` | no | deterministic rules (`--nli` adds the model path) |
| `reweave fix` | no | the four above, looped (`--regenerate` adds the rewrite) |
| `reweave facts --nli`, `fix --regenerate` | local Ollama | a local open-weight model, never hosted |

**The rewrite cannot re-stamp a watermark.** Regeneration is the one stage that
emits new text, so it is the one that could carry a fresh mark. It refuses any
model that is not local, open-weight, and on a verified-unwatermarked allowlist
(`gemma3`, `llama3.2`, `qwen2.5`, `mistral`, and a few more); the default is
`gemma3:4b`. This is safe for a concrete, checkable reason: statistical text
watermarks (SynthID-Text; the KGW green-list family) are decode-time processors
applied by a serving stack, not weights baked into the model. The open weights
you pull with Ollama carry none, so text generated locally starts unmarked. A
hosted endpoint is refused outright, because it may apply SynthID and it would
see your source text. Enforced at construction in `_ollama.assert_watermark_safe`
and pinned by `tests/test_watermark_guard.py`. Sources: Dathathri et al.,
"Scalable watermarking for identifying large language model outputs", Nature 634
(2024); the `google-deepmind/synthid-text` reference implementation; Kirchenbauer
et al., "A Watermark for Large Language Models" (2023).

What this does not buy you is silence from every third-party detector. Removing a
proactive watermark is not the same as reading as human to an arbitrary
classifier, and no honest tool can promise the latter. That is the job of the
substrate rewrite and the human-signature score, both measured below, not of the
model choice.

---

## Benchmarks

Every number below is produced by a script in this repo. Nothing is quoted from
a vendor page, and nothing is estimated.

```bash
python3 dataset/fetch.py
PYTHONPATH=src python3 tests/harness/evaluate.py         # watermark + scorer
PYTHONPATH=src python3 tests/harness/guard_eval.py       # semantic guard
PYTHONPATH=src python3 tests/harness/fact_eval.py --live # fact gate
PYTHONPATH=src python3 tests/harness/nli_benchmark.py    # NLI vs SNLI
PYTHONPATH=src python3 tests/harness/end_to_end.py       # full pipeline
```

### 1. Watermark detection: is the ground truth real?

Our pure-Python SynthID-Text reference (tournament sampling, H=4, m=6), across
120 watermarked and 120 unwatermarked sequences:

| metric | value |
|---|---|
| mean score, watermarked | 0.713 |
| mean score, unwatermarked | 0.504 |
| **AUROC** | **1.000** |
| **TPR at FPR=1%** | **1.000** |

A perfectly detectable watermark, which is what makes the removal number below
mean anything.

### 2. Watermark removal: how much regeneration is enough?

Regenerating a fraction *p* of the token substrate:

| p regenerated | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | **1.0** |
|---|---|---|---|---|---|---|---|
| still detected (TPR@1%) | 1.000 | 0.992 | 0.767 | 0.467 | 0.075 | 0.042 | **0.008** |

Detection is near dead by the halfway point, with zero knowledge of the scheme.
This is the substrate attack.

### 3. Editing versus regenerating

| approach | words changed | still detected |
|---|---|---|
| light copy-edit | ~10% | **0.992**, watermark survives |
| full regeneration | ~100% | **0.008**, watermark gone |

Removal tracks *how many token identities change*. Synonym-swapping tools change
few, so they under-remove. This is why the architecture regenerates and never
edits in place.

### 4. End to end with a real local model (`llama3.2:1b`)

| | watermark mean-score |
|---|---|
| unwatermarked baseline | 0.501 |
| watermarked, before | 0.708 |
| watermarked, after regeneration | **0.506** |
| **still detectable (TPR@1%)** | **0.000** |

A real, local, unwatermarked model collapsed a measurable watermark to baseline.

### 5. Human-signature scorer, real human versus real AI text

24 public-domain human passages against 60 Gemma 7B responses. Per-feature
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

**Composite AUROC 0.876.** The anti-correlated rows are a finding, not a bug.
Modern instruction-tuned models are specific and well structured, so "specificity
means human" is a false prior that most detector heuristics still encode. Those
three features were dropped from the score.

The scorer **abstains** when under 60% of feature weight is available. Short text
cannot support burstiness, and a one-line string full of AI tells was scoring
0.850 "human-like" before this was added.

**On live 2026 chat output.** The same essay prompt given to three assistants,
downloaded and scored unmodified:

| source | human-signature | verdict | em-dashes |
|---|---|---|---|
| ChatGPT | 0.463 | uncertain | 0 |
| Gemini | 0.609 | uncertain | 2 |
| Grok | 0.609 | uncertain | 1 |

None reached the 0.70 human threshold, and none was confidently called AI either,
which is the honest outcome: the scorer ranks, it does not adjudicate. Note that
two of the three reached for em-dashes in a five-paragraph essay. That habit is
why `em_dash_rate` is a feature at all, and why this repo's own prose avoids it.

### 6. Meaning preservation: the semantic guard

The gate must accept "same meaning, different words" and reject "different
meaning". The number that matters is the margin:

| guard | faithful reword | different topic | margin |
|---|---|---|---|
| Jaccard (word overlap) | 0.233 | 0.032 | +0.200, unusable |
| **Embeddings (`all-minilm`)** | **0.650** | **0.042** | **+0.608** |

Word-overlap similarity rejects a *correct* rewrite at 0.233, because changing
the words is the job. With Jaccard the gate accepted nothing at all.

### 7. Fact preservation: where embeddings fail

Sentence embeddings encode topic, not truth value. Measured:

| pair, same topic and opposite truth | embedding similarity |
|---|---|
| "X **are** computers" vs "X are **not** computers" | 0.959 |
| "deployment **succeeded**" vs "deployment **failed**" | 0.898 |
| "revenue **increased** 40%" vs "**decreased** 40%" | 0.776 |

All three sail past any usable floor. This is a documented property of
distributional representations ([arXiv:2507.12782](https://arxiv.org/abs/2507.12782)),
not a bad model choice, so the fix is a second gate rather than a better
threshold.

Fact gate results. Rules cover numerals, proper nouns, NegEx-based negation, and
antonym polarity:

| curated set, n=8 (see section 8 for real data) | rules | rules + NLI |
|---|---|---|
| lexical inversions caught | **3/3** | 3/3 |
| **reworded** inversions caught | **0/8** | **8/8** |
| faithful rewrites not falsely rejected | 8/8 | 8/8 |

Rules cannot see a reversal written in different words ("sales climbed" becomes
"sales were disappointing"). That is what NLI adds. Alignment has to be
embedding-based: with lexical alignment the reworded pairs never reach the model
at all, and recall collapses to **2/8**.

On 8 live regenerations of real Gemma text, the gate rejected 2, both genuine
fact loss. The 1B model had dropped "Microsoft, Sony, Unity, Unreal, CPU" from
one of them.

### 8. NLI backend on real labelled data

The 8/8 above is our own curated set and is worth little on its own. It measures
whether the mechanism fires, not whether the model is right. Measured on **SNLI
validation** (Bowman et al., EMNLP 2015, human-annotated and not written by us),
n=150, `gemma3:4b`, with the original class-definitions prompt:

| metric | value | what it means here |
|---|---|---|
| 3-class accuracy | 0.587 | against ~0.88 human and ~0.92 SOTA; a 4B general model is not a trained NLI head |
| **contradiction precision** | **0.923** | when it rejects a rewrite, it is right 92% of the time |
| contradiction recall | 0.444 | original prompt; the improvement is below |

The confusion matrix showed the whole problem in one cell: contradiction became
neutral 27 times out of 30 misses. The model was not confusing contradiction
with entailment, it was retreating to "no finding" when unsure. That is a prompt
problem before it is a model problem, so we swept four prompts on the same rows
(`tests/harness/nli_prompt_sweep.py`, n=100), scoring the contradiction class:

| prompt | precision | recall | F1 |
|---|---|---|---|
| original | 1.000 | 0.432 | 0.604 |
| **strict-neutral** (new default) | **0.950** | **0.514** | **0.667** |
| few-shot (`high_recall=True`) | 0.781 | 0.676 | 0.725 |

Naming NEUTRAL as the *narrow* label rather than the safe one, and forcing a
"could both be true at the same moment?" test, buys 8 points of recall for one
false positive in 100. That is now the default. Few-shot reaches 0.68 recall but
drops precision to 0.78, which for a gate that blocks a user's rewrite means 7
good rewrites wrongly rejected per 100, so it ships opt-in rather than as the
default.

For a blocking gate, precision is the metric that decides usability. The honest
summary stands:

> **Rules catch what they catch, deterministically. NLI adds partial coverage of
> reworded reversal at low false-alarm cost. Neither is a guarantee.**

Both figures are bounds. SNLI is *harder* than our actual task, being full of
subtle neutral distinctions, whereas we only ever compare a sentence to a rewrite
of itself, a narrower and more contrastive distribution, so these are
conservative lower bounds for this use. The curated 8/8 is an upper bound. The
truth is between, and we report both rather than picking the flattering one.

### 9. File provenance: what survives a perfect rewrite

Regeneration replaces every token, so no statistical watermark survives it. A
file is not only its text, though. Save an answer from a chat UI and the
operating system records where it came from, somewhere no editor shows you.

These are **real downloads**, kept in `tests/` as fixtures: the same essay task
given to six different assistants, saved the way anyone would save it. None of
them mentions its origin anywhere in the prose. All of them announce it anyway,
and macOS will show you in Finder under Get Info, "Where from":

| file | source URL still attached | traces found |
|---|---|---|
| `ChatGPT-Victoria Memorial Essay.md` | `https://chatgpt.com/` | 3 |
| `Gemini-Victoria Memorial Essay.md` | `https://contribution.usercontent.google.com/download?c=...` | 3 |
| `Grok-Victoria Memorial Kolkata Essay.md` | `https://grok.com/` | 3 |
| `Deepseek-Victoria Memorial Essay.md` | `https://chat.deepseek.com/` | 3 |
| `Qwen-Victoria Memorial Essay.md` | `https://chat.qwen.ai/` | 3 |
| `Kimi-Victoria Memorial Essay.md` | a signed storage URL (see below) | 3 |

Three traces each: the source URL, the `com.apple.quarantine` stamp naming the
browser that downloaded it (`Arc`, with a UUID), and the filename. The vendor
detector knows the non-Western labs too now (Kimi/Moonshot, Qwen, DeepSeek, GLM,
Doubao), which is why Kimi and Qwen files get caught by name and not just by URL.

The Kimi file is the sharpest example, and it is why `meta --dry-run` now points
at *which parts* of a URL identify you. Its "where from" is not a homepage; it is
a ByteDance Volcano Engine (TOS) pre-signed download URL, and inside it:

```
- file-id: 1a00a706-4b72-8ae9-8000-096e74afe606
- credential (X-Tos-Credential): AKLTYTJlNjgwMjY2ZDBkNDFiYmI5YWNi...
- signature (X-Tos-Signature): e2d574d13a04daae005b6b1f0ae1c24f...
- expiry (X-Tos-Date): 20260816T120055Z
```

That is an access credential and a request signature into the vendor's object
storage, keyed to one file, not merely "this came from Kimi". A plain homepage
URL like Qwen's decomposes to nothing sensitive, and the report stays quiet;
this one gets pulled apart component by component.

The Gemini one is worth decoding too. That `c=` parameter is base64, and inside it:

```
bard_storage ... response_data ... e28349d995dad45a00065927d1449194037055f63804c310
```

A unique per-response identifier. Not merely "this came from Gemini" but *which
generation produced it*. Rewrite every sentence in that file and the identifier
is untouched.

Both filenames are the third leak, and the most visible one, since the filename
is what gets shared.

| layer | example | result |
|---|---|---|
| extended attribute | `kMDItemWhereFroms`, the source URL | removed |
| extended attribute | `com.apple.quarantine`, downloading app and time | removed |
| filename | `ChatGPT-Quarterly Notes.md` | reported; renamed only with `--rename` |
| YAML front matter | `generator: ChatGPT`, `model: gpt-5` | removed |
| inline attribution | `<!-- Generated by ChatGPT -->` | removed |
| DOCX | `dc:creator`, `cp:lastModifiedBy`, `Application` | removed |
| PDF | `/Producer`, `/Creator`, XMP packet | **reported, not removed** |

Real front-matter content (`title`, `tags`) and the prose itself are untouched.

Two deliberate refusals in that table. Renaming changes the path the caller
handed us, so it never happens implicitly. And PDF fields need a library that
rebuilds the xref table, so we report them rather than claim a removal we did
not perform.

`scrub_file` re-inspects the file afterwards and reports what actually survived
instead of trusting its own delete calls. That check earned its place
immediately: rewriting the text after stripping attributes caused macOS to
re-create one, so the first version reported a removal that had already been
undone.

---

## How it works

```
FILE -> 0 Meta -> 1 Scrub -> 2 Extract -> 3 Regenerate -> 4 Score -> 5 Gate -> OUT
        (xattrs)   (chars)    (meaning)    (LOCAL LLM)     (stats)      |
                                                                        |
                                              +-------------------------+
                                        guard: same topic?   facts: same claims?
                                        (embeddings)         (rules + NLI)
```

**Stage 3 must use a local, unwatermarked, open-weight model.** Paraphrasing
Claude's output *with Claude* strips the old mark and stamps a fresh one, which
is the Self-Watermark Trap. `Pipeline.__init__` refuses a regenerator not
declared `is_unwatermarked`, and the Ollama adapter refuses any hosted model and
any local model off the verified-unwatermarked allowlist (override per call with
`allow_unlisted`, or `--allow-model` on `reweave fix`). See "Runs offline, and
never re-watermarks" above for the allowlist and the citations behind it.

Full design contract: **[ARCHITECTURE.md](ARCHITECTURE.md)**.
Diagrams: **[docs/diagrams.md](docs/diagrams.md)**.
All numbers: **[RESULTS.md](RESULTS.md)**.

---

## Install

The core installs with **zero dependencies**. That is the point: a
dependency-free core cannot rot.

```bash
pip install -e .
```

The import package and CLI are called `reweave`, because the system does not
erase a mark. It re-weaves the text, discarding the token sequence where every
watermark lives and weaving a new one from the meaning. (An install name that
differs from the import name is idiomatic; compare `beautifulsoup4` and `bs4`.)

Local models via [Ollama](https://ollama.com) for the model-backed stages:

```bash
ollama pull llama3.2:1b   # regeneration and extraction
ollama pull all-minilm    # semantic guard embeddings, 45MB
ollama pull gemma3:4b     # NLI fact checking, optional
```

---

## A note on style

This README contains no em-dashes and no Unicode arrows, and that is deliberate.
Overused em-dashes are one of the most reliable surface tells of unedited AI
prose, and `score/features.py` weights `em_dash_rate` as an AI signal. A tool
that scores that tell while its own front page is full of them has already told
you how much care went into it.

`tests/test_docs_style.py` enforces this on every doc in the repo. We run our own
scorer on our own documentation.

---

## What is proven, and what is not

**Proven here:** the substrate attack removes a keyed watermark we can measure; a
real local model does it end to end; removal scales with the fraction of tokens
changed; burstiness is the one robust human-versus-AI signal available without an
LM; the embedding guard makes the gate usable where word overlap could not; the
rule checker catches lexical inversion deterministically with no false alarms;
file-level provenance is removed and what survives is reported.

**Not proven here:**

- **Removal of a live vendor watermark.** Their keys are secret, so nobody
  outside can measure it, and any project claiming otherwise has not measured it
  either.
- **Fact preservation as a guarantee.** NLI narrows the reworded-reversal gap
  (rules alone catch 0% of it; the tuned default catches about half at high
  precision). It does not close it. A rewrite can still corrupt meaning in ways
  nothing here detects.
- **A production-grade regenerator.** A 1B model is weak, and output quality
  tracks model quality directly.
- **A calibrated human-versus-AI boundary.** The scorer separates (AUROC 0.876),
  but a hard threshold needs the perplexity backend and register-matched data.

If you need one sentence: **this removes watermarks it can measure, and is honest
about the ones it cannot.**

---

## A note on use

A genuine rewrite into your own voice *is* authorship. That is the legitimate
use, and it is what this is built for: research on watermark robustness, and
writing that sounds like you.

Using it to claim human authorship where disclosure is legally or academically
required is deception, and that is on the operator rather than the tool. Under EU
AI Act Article 50(2), providers must mark synthetic content, and stripping such a
mark to misrepresent provenance may carry legal consequences. The pipeline can
regenerate a token sequence. It cannot write the one line only you could write.

Apache-2.0.
