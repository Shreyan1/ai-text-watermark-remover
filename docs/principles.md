# Principles: the invariants, expanded

These are the rules that make Reweave durable. `ARCHITECTURE.md` states them
tersely; this file is the reasoning, so a future contributor understands *why* a
tempting change is actually a mistake.

## I1: Scheme-agnostic core

**Rule.** The `core/` package contains no watermark-specific code: no scheme
names, no key handling, no per-scheme branches.

**Why.** A watermark detector must be updated for every new scheme. A substrate
attack never encoded any scheme, so there is nothing to update. The moment you
add `if scheme == "synthid"` to the core, you have created maintenance debt that
grows with every vendor and every year. Keep scheme knowledge, if it ever exists
at all, at the edge, in an optional analysis adapter that is *never* on the
removal path.

**Temptation to resist.** "Let's detect the scheme first so we can remove it
more precisely." No. Precise removal is unnecessary (regeneration is total) and
detection is impossible keyless. This temptation reintroduces the arms race.

## I2: Substrate attack, not detector evasion

**Rule.** Optimise toward a fixed, vendor-independent target (statistical
human-ness). Never optimise against a specific named detector.

**Why.** Humanizers that train against a target detector win against that
detector and lose against an ensemble, and lose entirely when the detector
updates. That is an arms race with a structural disadvantage. A fixed target has
no counter-move: when GPTZero v9 ships, human writing still looks like human
writing, and we did nothing. Durability comes from refusing to play.

**Temptation to resist.** "Detector X flags us; let's add a rule to beat X." That
rule rots the day X updates. Instead ask: *is our text genuinely more uniform
than human text on some measurable axis?* If yes, fix the axis in `score/`. If
no, X is wrong and we ignore it.

## I3: Stable Core / Swappable Edge

**Rule.** Dependencies point inward. The edge imports core contracts; the core
imports nothing from the edge and has no third-party dependencies.

**Why.** Everything that dates, models, tokenizers, runtimes, APIs, is at the
edge behind an interface. The core is data shapes and orchestration rules, which
do not date. A dependency-free core cannot break from an upstream release. This
is the single most important structural decision for a "survives all years"
system.

**Test.** Adopting the 2028 SOTA model = one new edge class + one registry line.
If it requires a core edit, an invariant was violated.

## I4: Meaning is the pivot; surface form is disposable

**Rule.** Always regenerate the token sequence from a meaning representation.
Never edit surface tokens in place.

**Why.** All watermarks and all style-tells live in the surface token sequence.
In-place edits (synonym swaps, sentence shuffles) leave most original tokens
intact, so most of the watermark signal and most of the AI-uniformity survive.
Round-tripping through meaning discards the entire surface, so there is nothing
of the original statistics left to detect. This is why paraphrase beats editing
in every published removal study.

## I5: Measure everything; prove nothing you can't

**Rule.** Score every transform before/after (human-signature + semantic
similarity + fact preservation). Report measurements; refuse unverifiable claims.

**Why.** We cannot prove a *keyed* watermark is gone without the key, so we never
claim it. We *can* show the human-signature score rose and meaning was preserved, so we report that. Honesty here is also a product feature: it is the difference
between Reweave and the SEO "watermark remover" sites that assert removal they
never measured.

**Corollary: a check must state its own scope.** "Meaning preserved" meant
*topic* preserved until we measured it: sentence embeddings score a claim and its
negation at 0.959, so the guard was silently certifying something it could not
see. The fix was not a better threshold but a second check plus an explicit,
written blind spot. A measurement that overstates its coverage is worse than no
measurement, because it converts a known unknown into false assurance.

**Corollary: prefer published prior art to invented heuristics.** Where a rule
is uncertain, use the validated one and cite it. Negation detection follows NegEx
(Chapman et al., 2001) rather than a list we made up; the reason the fact gate is
needed at all is a documented property of distributional embeddings, not our
observation alone. Where we do generalise beyond a source, we say which part is
ours.

---

## The Self-Watermark Trap (a corollary of I1+I4)

Regenerating with a *watermarked* model strips the old mark and applies a fresh
one. Stage ③ must use a local, unwatermarked, open-weight model. This is why the
architecture forbids routing regeneration through a hosted frontier API, not for
cost or latency, but because it silently re-watermarks and defeats the purpose.
