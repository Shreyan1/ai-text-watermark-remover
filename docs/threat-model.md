# Threat model: watermark families and what defeats each

Reweave's core never reasons about specific schemes (Invariant I1). This document
exists only so contributors understand *what is out there* and can see that a
single substrate attack covers all of it. Nothing here is imported by `core/`.

## Family A: Character / encoding artifacts

**What.** Zero-width spaces (U+200B/C/D), non-breaking spaces, homoglyph
substitutions (Cyrillic а for Latin a), curly quotes, unusual Unicode, trailing
variation selectors. Often not deliberate watermarks at all, just chat-UI and
word-processor residue. Sometimes deliberate (invisible-character tagging).

**Defeated by.** Stage ① `scrub/`, deterministically and completely. NFKC
normalisation + a homoglyph map + zero-width strip. No model needed.

**Detectability by us.** Total, these are visible to a byte scan. This is the
*only* family we can honestly "detect."

## Family B: Statistical / generative, distribution-biasing

### B1: Green-list / red-list (Kirchenbauer et al., 2023; "KGW", "Unigram")

**What.** A secret key pseudo-randomly partitions the vocabulary into green/red at
each step; the sampler is biased toward green. Over enough tokens the green ratio
is statistically improbable for human text. Detection needs the key, not the
model.

**Empirical fragility.** ~100% removal after a single paraphrase pass in
published evaluations. Unigram slightly more robust than KGW but still falls.

### B2: Tournament sampling (SynthID-Text; Google DeepMind, Nature 2024)

**What.** Draws multiple candidate tokens from the model's own distribution and
runs a keyed knockout tournament (m≈30 layers, Bernoulli(0.5) g-values). Because
every candidate came from the genuine distribution, output stays natural.
Non-distortionary in the 2-competitor config; adds repeated-context masking for
sequence-level non-distortion. Shipped in Claude (Aug 2026) and Gemini.

**Empirical fragility.** ~98.3% removal after one paraphrase pass. Cross-lingual
summarization drives detection to near chance (AUROC ≈ 0.53). ETH Zürich: >90%
removal with off-the-shelf paraphrasers, *worse* than competing schemes.

**Known theoretical weakness.** Under mean-score detection, stacking tournament
layers *decreases* separation (arXiv 2603.03410's layer-inflation attack). The
shipped config uses a learned Bayesian scorer, not mean score, so this is
narrower than headlines imply, but it confirms the signal is brittle.

### B3: Semantic / sentence-level watermarks (SIR, XSIR, AliMark, SynGuard ...)

**What.** Embed signal in sentence embeddings or semantic structure rather than
exact tokens, specifically to survive paraphrase.

**Empirical fragility.** More robust to monolingual paraphrase, but cross-lingual
summarization still collapses even XSIR (AUROC 0.827 -> 0.53). Meaning-preserving
*restructuring* (not just word swaps) degrades them.

## Why one substrate attack covers B1-B3

All of B rides on a correlation between the emitted surface sequence and a secret
key. Reweave's pivot-through-meaning regeneration (Invariant I4) produces a *new*
surface sequence that was never correlated with any key, token-level (B1/B2)
directly, and semantic (B3) to the extent regeneration restructures rather than
paraphrases. This is why the core needs no per-family code: the attack is on the
shared dependency, not on any family's mechanism.

## What we cannot claim

- We cannot *prove* a keyed B-family mark is gone without the issuer's key. We
  measure human-signature and semantic drift and report those (Invariant I5).
- B3 schemes that survive heavy restructuring are the hardest case; aggressive
  regeneration (high `aggressiveness`) is the lever, at higher semantic-drift
  risk, which the guard stage bounds.

## Sources

- Dathathri et al., *Scalable watermarking for identifying LLM outputs*, Nature 634 (2024), SynthID-Text
- Kirchenbauer et al., *A Watermark for Large Language Models* (2023), KGW
- arXiv 2603.03410, theoretical analysis + layer-inflation attack
- arXiv 2607.16010, forensic-readiness eval (98.3% / 100% removal rates)
- arXiv 2510.24789, cross-lingual summarization removal attack
- ETH Zürich SRI Lab, probing SynthID-Text (>90% paraphrase removal)
