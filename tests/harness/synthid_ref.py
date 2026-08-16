"""A faithful, pure-Python reference implementation of SynthID-Text.

Zero dependencies (hashlib + random only). This is the *ground truth* for the
harness: we watermark with keys WE hold, so we can measure before/after honestly
— the one thing you cannot do to a real Claude/Gemini watermark without the key.

It implements the exact quantities from Dathathri et al., Nature 634 (2024):

  * g-value            g_ℓ(x, r), a keyed Bernoulli(0.5) pseudo-random bit
  * random seed        r_t = h(x_{t-H..t-1}, key) , sliding-window over context
  * Tournament sampling  draw M=2^m candidates from p_LM, run an m-layer knockout
  * mean-score detector  Score(x) = (1/mT) Σ_t Σ_ℓ g_ℓ(x_t, r_t)

Tokens here are WORDS (strings), not subword ids. That is the one deliberate
simplification, and it buys a crucial property: the detector can score ANY text
by tokenising it the same way, so we can watermark a passage, hand it to a real
local model to rewrite, and measure the collapse. Real SynthID is weaker
per-token (subword entropy, RLHF); the harness measures mechanism, not Google's
exact operating point. See dataset/README.md.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _prf_unit(key: bytes, *parts) -> float:
    """Keyed pseudo-random function -> uniform float in [0, 1). Deterministic.
    Accepts int and str parts (words are str, layer/seed are int)."""
    h = hashlib.sha256(key)
    for p in parts:
        if isinstance(p, int):
            h.update((p & _MASK64).to_bytes(8, "little"))
        else:
            h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")  # field separator, avoids concatenation collisions
    return int.from_bytes(h.digest()[:8], "little") / 2**64


def context_seed(context_words: list[str], key: bytes, H: int) -> int:
    """r_t: hash of the last H context words with the key (sliding window)."""
    window = context_words[-H:] if H > 0 else context_words
    h = hashlib.sha256(key)
    h.update(b"seed")
    for w in window:
        h.update(str(w).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest()[:8], "little")


def g_value(token: str, layer: int, seed: int, key: bytes) -> int:
    """g_ℓ(x, r): keyed Bernoulli(0.5) bit for (token, layer, seed)."""
    return 1 if _prf_unit(key, token, layer, seed) >= 0.5 else 0


@dataclass
class SynthIDConfig:
    m_layers: int = 6          # tournament depth (production ≈ 30; we keep it tractable)
    H: int = 4                 # sliding-window context size for the seed
    key: bytes = b"reweave-harness-key-v1"
    repeated_context_masking: bool = False  # Algorithm 3; off by default for clarity


def tournament_sample(
    dist_words: list[str],
    dist_weights: list[float],
    seed: int,
    cfg: SynthIDConfig,
    rng: random.Random,
) -> str:
    """Emit one token by Tournament sampling over the LLM distribution p_LM.

    Draw M=2^m candidates from p_LM, then run an m-layer knockout where layer ℓ
    picks the higher g_ℓ (ties broken randomly). The survivor is the emitted token.
    """
    m = cfg.m_layers
    candidates = rng.choices(dist_words, weights=dist_weights, k=2**m)
    for layer in range(1, m + 1):
        winners: list[str] = []
        for i in range(0, len(candidates), 2):
            a, b = candidates[i], candidates[i + 1]
            ga = g_value(a, layer, seed, cfg.key)
            gb = g_value(b, layer, seed, cfg.key)
            if ga > gb:
                winners.append(a)
            elif gb > ga:
                winners.append(b)
            else:
                winners.append(a if rng.random() < 0.5 else b)
        candidates = winners
    return candidates[0]


def mean_score(words: list[str], cfg: SynthIDConfig) -> float:
    """The detector. Needs only the tokens, the key, and the seed rule, NOT the LM.

    Returns mean g-value in [0,1]; watermarked text trends > 0.5, human/unwatermarked
    text sits at ≈ 0.5.
    """
    if not words:
        return 0.0
    total = 0
    count = 0
    seen: set[int] = set()
    for t, tok in enumerate(words):
        seed = context_seed(words[:t], cfg.key, cfg.H)
        if cfg.repeated_context_masking:
            if seed in seen:
                continue
            seen.add(seed)
        for layer in range(1, cfg.m_layers + 1):
            total += g_value(tok, layer, seed, cfg.key)
            count += 1
    return total / count if count else 0.0


def score_text(text: str, cfg: SynthIDConfig) -> float:
    """Convenience: tokenise arbitrary text the harness way, then mean-score it.
    This is what lets us measure a watermark before/after a real model rewrite."""
    from wordlm import tokenize
    return mean_score(tokenize(text), cfg)
