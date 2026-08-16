# AI Text Watermark Remover — Architecture

> The holy-grail design. This document is the contract. Code may be rewritten;
> the invariants in this file are what must survive every model, every
> watermark scheme, and every year to come.
>
> Visual companion: **[docs/diagrams.md](docs/diagrams.md)** ·
> Measured results: **[RESULTS.md](RESULTS.md)**
>
> **On names.** The repository and distribution are
> `ai-text-watermark-remover` — what the thing does, and what people search for.
> The engine inside is **Reweave** (`import reweave`, CLI `reweave`), because it
> does not erase a mark: it *re-weaves* the text, discarding the token sequence
> where every watermark lives and weaving a new one from the meaning. Both names
> appear throughout and refer to the same system.

---

## 0. What this is (and the honest scope)

Reweave is **not a watermark detector**. It is a **regenerator**: it rebuilds a
piece of text from its *meaning* into a genuine human voice. Destroying any
embedded statistical watermark is a *side effect* of that regeneration, not a
targeted operation.

This framing is deliberate and load-bearing. It is the entire reason the system
survives future models. Read §2 before touching anything.

Two feasibility facts drive the whole design:

| Ask | Feasible? | Why |
|---|---|---|
| Detect *which* watermark a text carries, without the issuer's key | **No** | Modern watermarks (SynthID-Text, KGW) are cryptographically keyed. A single document reveals nothing without the secret key. Anyone claiming keyless universal detection is selling snake oil. |
| Remove a watermark | **Yes** | A token-level watermark lives in *which tokens were emitted*. Regenerate the token sequence and the signal dies — 98–100% empirically, across every known scheme, with no knowledge of the scheme. |

So we build the achievable thing (regeneration) and refuse to build the
impossible thing (keyless universal detection). What we *do* offer on the
detection side is an honestly-labelled **human-signature score** — a heuristic
"how AI-uniform does this read" gauge used as an internal quality gate, never
sold as watermark detection.

---

## 1. Prime Directive

> **Attack the substrate, never the scheme.**

Every token-level text watermark — SynthID-Text, KGW/green-list, Unigram, and
anything shaped like them — is *forced* to hide its signal in one shared place:
**the specific sequence of tokens the model emitted.** They differ in how they
bias that sequence; they are identical in depending on it.

Therefore we never model any individual watermark. We attack the one thing they
all stand on: we discard the surface token sequence and regenerate a new one
from meaning. A scheme we have never heard of, shipped by a vendor that does not
yet exist, is defeated the moment its output passes through the pipeline —
because we removed the ground it was standing on without ever looking at it.

**Corollary — the Self-Watermark Trap.** If you regenerate with a *watermarked*
model, you strip the old mark and stamp a fresh one. The regenerator MUST be an
unwatermarked, open-weight, locally-run model. This is a hard invariant, not a
preference. See §5, Stage ③.

---

## 2. Why this survives every future model — the Invariants

These five invariants are the "holy grail." If a future change would violate one
of them, the change is wrong, not the invariant.

- **I1 — Scheme-agnostic core.** The core contains zero watermark-specific code.
  No scheme names, no key logic, no per-scheme branches. It cannot go stale
  against a new scheme because it never encoded any scheme.

- **I2 — Substrate attack, not detector evasion.** We optimise toward a *fixed,
  vendor-independent target* (statistical human-ness), never *against a specific
  detector*. Detector-chasing is the arms race that kills every humanizer; a
  fixed target has no arms race. New detector ships → we do nothing.

- **I3 — Stable Core / Swappable Edge.** Everything that will change over the
  years (models, tokenizers, scorers) lives behind an interface at the edge.
  Everything in the core is a data contract or an orchestration rule. Adding the
  2028 state-of-the-art model is writing one new adapter; the core is untouched.

- **I4 — Meaning is the pivot, surface form is disposable.** The pipeline's
  center of gravity is a meaning representation. Surface tokens (where all
  watermarks and all style tells live) are always regenerated, never edited in
  place. In-place editing leaks the original token statistics; regeneration does
  not.

- **I5 — Measure everything, prove nothing you can't.** Every transform is
  scored before/after on human-signature and semantic-similarity. We report what
  we can measure and explicitly refuse claims we cannot verify (e.g. "the keyed
  watermark is provably gone" — unprovable without the key).

---

## 3. The layered model

```
┌─────────────────────────────────────────────────────────────────┐
│  STABLE CORE  (changes ~never; no dependencies; no scheme code)   │
│                                                                   │
│   core/types.py        the data contracts (Document, Meaning, …)  │
│   core/interfaces.py   the five stage contracts (ABCs)            │
│   core/pipeline.py     orchestration + the convergence loop       │
│   core/registry.py     how edge adapters plug in                  │
└─────────────────────────────────────────────────────────────────┘
                              ▲   ▲   ▲
        implements contracts  │   │   │   implements contracts
                              │   │   │
┌───────────────┐  ┌──────────┴─┐ │ ┌─┴──────────┐  ┌───────────────┐
│  SWAPPABLE EDGE (changes often; where models/deps/schemes live)   │
│                                                                   │
│  scrub/     extract/     regenerate/     score/     guard/        │
│  (code)     (LLM)        (LOCAL LLM)      (stats)    (embeddings)  │
└───────────────────────────────────────────────────────────────────┘
```

The rule that makes it durable: **dependencies point inward.** The edge imports
the core's contracts. The core imports nothing from the edge and nothing from
the outside world. The core has *no third-party dependencies at all*, so it
cannot rot.

---

## 4. Data contracts (the spine)

All types are immutable. The pipeline threads them stage to stage. Full
definitions in `core/types.py`; the shape:

- **`Document`** — text + metadata (source hint, language, model guess). The unit
  in and out of every stage.
- **`Meaning`** — the pivot representation: an ordered set of claims/points/intent
  stripped of surface form. This is where watermarks die. Format is deliberately
  loose (structured outline) so extractors can evolve.
- **`FeatureVector`** — the measurable human/AI signals (see §6).
- **`HumanSignature`** — `score ∈ [0,1]`, a verdict, and the feature breakdown.
  Higher = more human-like. This is our honest, internal "detector."
- **`VoiceProfile`** — the target to regenerate *into*: contraction policy, target
  burstiness, the user's own vocabulary and sample texts, banned AI-tells.
- **`Constraints`** — the must-keep facts of a text: numerals, proper nouns, and a
  polarity fingerprint per claim. The half of meaning embeddings cannot see.
- **`FactReport`** — did the rewrite keep them? Missing numbers, dropped names,
  and the specific claims whose polarity flipped.
- **`TransformResult`** — original, output, before/after `HumanSignature`,
  semantic similarity, `FactReport`, iteration count, and a full trace.

---

## 5. The pipeline — five stages

```
        ┌──────────────────────────── convergence loop (I2) ─────────────────────────┐
        │                                                                              │
INPUT ─▶ ① Scrub ─▶ ② Extract ─▶ ③ Regenerate ─▶ ④ Score ─▶ ⑤ Gate ─┬─ pass ─▶ OUTPUT
        (code)      (LLM)         (LOCAL LLM)      (stats)            │
                                       ▲                             └─ fail ─▶ back to ③
                                       │                                (more aggressive)
                                  VoiceProfile
```

**① Scrub — `scrub/`** *(pure code, zero ML, ships first)*
Kill the character-residue watermark category outright: zero-width chars, NBSP,
homoglyphs, curly→straight quotes, em-dash normalisation, Unicode NFKC. Handles
the entire "character artifact" family deterministically. Also the fast path: if
a caller only wants artifact hygiene, they stop here.

**② Extract — `extract/`** *(LLM, model-agnostic)*
Reduce the (scrubbed) document to `Meaning`. Discarding surface form here is what
severs the token-sequence substrate (I4). An extractor may be an LLM prompted to
outline, or a cheaper structural summariser. Swappable.

**③ Regenerate — `regenerate/`** *(LOCAL, UNWATERMARKED LLM — invariant, see §1)*
Rebuild prose from `Meaning`, conditioned on `VoiceProfile`, explicitly steered
toward high burstiness, high perplexity, the user's vocabulary. The
`aggressiveness` knob (0→1) is what the loop turns up on retries. This is the
only stage that MUST NOT use a watermarked/hosted frontier model.

**④ Score — `score/`** *(stats, no training)*
Compute `HumanSignature` from the feature vector (§6). This is the achievable
half of "detection," repurposed as a quality gate — never exposed as watermark
detection.

**⑤ Gate — `core/pipeline.py`**
Emit when `HumanSignature.score ≥ threshold` **and** meaning survives. Else loop
back to ③ with higher aggressiveness, up to `max_iterations`. This closed loop is
the "ever-evolving" mechanism: it optimises against a fixed statistical target
(I2), so it needs no updates when detectors or models change.

"Meaning survives" is **two orthogonal tests**, because no single measure covers
both axes:

| test | asks | adapter | blind to |
|---|---|---|---|
| `SemanticGuard` | *same topic?* | `guard/embedding_guard.py` | truth value |
| `FactChecker` (rules) | *same numbers, names, polarity?* | `verify/constraint_checker.py` | reworded inversion |
| `FactChecker` (NLI) | *does the rewrite contradict the source?* | `verify/nli_checker.py` | whatever its backend gets wrong |

`verify/composite.py` runs rules first, then NLI. The order is a guarantee, not
tidiness: NLI can only ADD findings, so an unavailable or wrong model can never
*weaken* what the deterministic checker already proved. The union is monotonic in
safety.

The similarity floor is only meaningful with a *meaning*-based guard. Word-overlap
(Jaccard) scores a faithful reword at 0.233 — it punishes exactly what Stage ③ is
for — so the gate accepts nothing. Embeddings measure meaning instead (reword
0.650 vs topic-drift 0.042, floor 0.35).

But embeddings are blind to truth value by construction, not by accident: the
distributional hypothesis learns words from the contexts they occur in, and a
sentence and its negation share their context almost exactly. Measured here at
0.776–0.959 for inverted pairs. That is a documented property of the
representation ([arXiv:2307.13989](https://arxiv.org/abs/2307.13989),
[arXiv:2507.12782](https://arxiv.org/abs/2507.12782)), so no floor can fix it —
it needs a second check that looks at what embeddings discard. `FactChecker` is
that check: numerals, proper nouns, and per-claim polarity, with negation handling
modelled on **NegEx** (Chapman et al., 2001) including its pseudo-negation
precedence rule. A candidate must pass **both** gates.

Those rules are lexical, so they miss a reversal written in different words (0/8
measured). `verify/nli_checker.py` closes that (8/8), following **SummaC**
(Laban et al., TACL 2022) for sentence-level granularity and max-then-mean
aggregation — with one deliberate deviation. SummaC takes a max over the full
M×N pair matrix, which is safe for *entailment* (the best supporter is the right
one to keep) but wrong for *contradiction*: two unrelated sentences in one
document routinely look contradictory, and a single spurious pair would veto a
good rewrite. So contradiction is judged only on aligned pairs. That also turns
M×N model calls into O(M), which is what makes an LLM backend affordable at all.

Alignment must be embedding-based, and the measurement is unambiguous: with
lexical alignment the reworded pairs never reach the model (8/8 → **2/8**).
Embeddings align, NLI judges — neither is asked to do the other's job.

---

## 6. The measurable target (human vs AI)

AI's failure mode is one word: **uniformity** — it regresses to the mean of its
training distribution. Human writing has variance. The scorer measures the gap:

| Feature | Human | AI (unedited) | Computation |
|---|---|---|---|
| Perplexity mean & variance | higher, spiky | lower, flat | per-token surprise under a reference LM |
| Burstiness | high | low | std-dev of sentence length |
| Syntactic burstiness | high | low | std-dev of parse depth |
| Type-token ratio | higher | lower | unique / total tokens (windowed) |
| Paragraph rhythm | uneven | symmetrical | CV of paragraph lengths |
| Punctuation fingerprint | idiosyncratic | em-dash / rule-of-three heavy | char n-gram freq vs baseline |
| Specificity | named entities, real numbers | hedged universals | NE + numeral density |

The feature set is **open** (I1/I3): `FeatureVector` is extensible, features
register themselves, and the score is a transparent weighted blend — no trained
classifier to retrain as models drift. The one feature no tool can synthesise —
*the line only the author could have written* — is explicitly out of scope and
documented as the human's job, which bounds honest claims (§8).

---

## 7. Extension model — adding the 2028 SOTA model

The durability test. To adopt a model that does not exist yet:

1. Write one class in `regenerate/` implementing the `Regenerator` ABC.
2. Register it: `@register("regenerator", "my-2028-model")`.
3. Select it by name/config.

No core file changes. No contract changes. Same for a new scorer feature, a new
extractor, a new scrubber rule. **If adopting a new model requires editing the
core, an invariant has been violated.**

---

## 8. Non-goals — what we refuse to build

- **Keyless universal watermark detection.** Impossible by design; claiming it is
  fraud. We ship an honestly-labelled human-signature heuristic instead.
- **Detector-specific evasion.** Violates I2 and starts an arms race we would
  lose. We optimise to a fixed statistical target, not against any named
  detector.
- **In-place surface editing to "sneak past" a mark.** Violates I4 and leaks the
  original statistics. We regenerate.
- **Authorship laundering as a feature.** A genuine rewrite into your own voice
  *is* authorship — that is legitimate. Using the tool to falsely claim human
  authorship where disclosure is legally or academically required is deception,
  and is the operator's responsibility, not the pipeline's. The tool can
  regenerate a token sequence; it cannot supply the one line only the author
  could write. That remains the human's.

---

## 9. Directory map

```
reweave/
├── ARCHITECTURE.md          ← this file (the contract)
├── README.md
├── docs/
│   ├── principles.md        ← the invariants, expanded
│   └── threat-model.md      ← watermark families + what defeats each
├── src/reweave/
│   ├── core/                ← STABLE. no deps, no scheme code.
│   │   ├── types.py
│   │   ├── interfaces.py
│   │   ├── pipeline.py
│   │   └── registry.py
│   ├── scrub/               ← Stage ①  (pure code, ships first)
│   ├── extract/             ← Stage ②  (LLM)
│   ├── regenerate/          ← Stage ③  (LOCAL unwatermarked LLM)
│   ├── score/               ← Stage ④  (stats)
│   ├── guard/               ← ⑤ gate A: semantic-similarity floor (topic)
│   ├── verify/              ← ⑤ gate B: fact preservation (truth value)
│   │   ├── constraints.py       numbers, names, NegEx polarity
│   │   ├── constraint_checker.py  rules — deterministic, zero-dep
│   │   ├── nli.py / ollama_nli.py NLI backend contract + local backend
│   │   ├── nli_checker.py       SummaC-style entailment check
│   │   └── composite.py         rules ∪ NLI (rules can only be added to)
│   └── cli.py
└── tests/
    └── harness/             ← self-watermark ground-truth test rig
```

---

## 10. Testing philosophy

You cannot verify removal of a *keyed* watermark without the key. So we build our
own ground truth:

1. **Self-watermark harness** (`tests/harness/`): use the open SynthID-Text /
   MarkLLM implementations to produce text *we* watermarked with keys *we* hold.
   Run it through the pipeline, verify with *our* detector that the score
   collapses. The only clean before/after measurement — build it first.
2. **Detector panel**: run outputs through an ensemble of public detectors; track
   score drops (proxy for the wild, since single-detector evasion doesn't
   generalise).
3. **Live text**: real post-Aug-2026 Claude / GPT output → confirm artifacts gone
   and human-signature up. Cannot prove keyed-watermark removal here; measure
   everything else.
4. **Semantic guard** (`guard/`): every run scored for meaning drift. Removal that
   garbles content is a failure, not a success.
5. **Fact gate** (`verify/`): every run checked for dropped numbers, dropped
   names, and inverted claims. Two properties are measured, because either one
   alone is meaningless — recall on inversions the embedding guard waves through,
   **and** the rejection rate on legitimate rewrites. A gate that rejects
   everything is as useless as one that rejects nothing.

A note on why the fact gate is rules and not a model. The principled approach is
NLI-based entailment — SummaC ([Laban et al., TACL 2022](https://aclanthology.org/2022.tacl-1.10/))
or AlignScore (ACL 2023) — and that remains the upgrade path. But it is worth
knowing the ceiling before assuming a model would be strictly better: SummaC's
state-of-the-art result on the six-dataset SummaC inconsistency benchmark is
**74.4% balanced accuracy**. Automated factual-consistency checking is not a
solved problem that we are approximating badly; it is an open one. A
deterministic check with a *stated, narrow* scope is the honest instrument here
(I5) — it never claims the coverage it does not have.
