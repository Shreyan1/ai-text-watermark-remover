"""Removal attacks — the token-substrate attack, parameterised by how much of the
sequence we regenerate. Operates in word space against the harness's ground-truth
watermark. This is where Invariant I4 is proven with numbers: removal strength
tracks the *fraction of tokens whose identity changes*.
"""

from __future__ import annotations

import random

from wordlm import WordLM


def paraphrase_remove(words: list[str], p: float, lm: WordLM, rng: random.Random) -> list[str]:
    """Regenerate a fraction ~p of positions by resampling from the (unwatermarked)
    LM given the preceding word. The new tokens were never Tournament-selected, so
    their g-correlation with the key is gone. p=1.0 ≈ full regeneration."""
    out = list(words)
    for i in range(len(out)):
        if rng.random() < p:
            prev = out[i - 1] if i > 0 else None
            out[i] = lm.sample_next_word(prev, rng)
    return out
