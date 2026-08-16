"""A tiny word-level bigram language model, the p_LM that Tournament sampling
draws candidates from. Zero dependencies.

Built from real English (the unwatermarked Gemma responses in the dataset) so the
distribution has realistic entropy. Word-level, not subword: honest and simple.
The watermark needs entropy to embed signal, so a real-text bigram is enough to
demonstrate the whole mechanism.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z]+'?[a-z]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class WordLM:
    def __init__(self, top_k: int = 60, smoothing: float = 0.4) -> None:
        self.top_k = top_k          # cap candidate support per context (speed)
        self.smoothing = smoothing  # unigram backoff weight
        self.id_of: dict[str, int] = {}
        self.word_of: list[str] = []
        self._bigram: dict[int, Counter] = defaultdict(Counter)
        self._unigram: Counter = Counter()
        self._uni_ids: list[int] = []
        self._uni_wts: list[float] = []

    def _id(self, word: str) -> int:
        i = self.id_of.get(word)
        if i is None:
            i = len(self.word_of)
            self.id_of[word] = i
            self.word_of.append(word)
        return i

    def fit(self, texts: list[str]) -> "WordLM":
        for text in texts:
            toks = [self._id(w) for w in tokenize(text)]
            for a, b in zip(toks, toks[1:]):
                self._bigram[a][b] += 1
                self._unigram[b] += 1
            if toks:
                self._unigram[toks[0]] += 1
        self._uni_ids = list(self._unigram.keys())
        total = sum(self._unigram.values()) or 1
        self._uni_wts = [c / total for c in self._unigram.values()]
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.word_of)

    def next_dist(self, prev_id: int | None) -> tuple[list[int], list[float]]:
        """Return (candidate ids, weights) for p_LM(· | prev). Mixes the observed
        successors of `prev` with unigram smoothing so there is always entropy."""
        succ = self._bigram.get(prev_id) if prev_id is not None else None
        support: dict[int, float] = {}
        if succ:
            for tok, cnt in succ.most_common(self.top_k):
                support[tok] = float(cnt)
        # Unigram backoff, guarantees candidates even for unseen/edge contexts.
        for tok, w in zip(self._uni_ids[: self.top_k], self._uni_wts[: self.top_k]):
            support[tok] = support.get(tok, 0.0) + self.smoothing * w * 100.0
        ids = list(support.keys())
        wts = list(support.values())
        return ids, wts

    def sample_next(self, prev_id: int | None, rng: random.Random) -> int:
        ids, wts = self.next_dist(prev_id)
        return rng.choices(ids, weights=wts, k=1)[0]

    def start_token(self, rng: random.Random) -> int:
        return rng.choices(self._uni_ids, weights=self._uni_wts, k=1)[0]

    def detokenize(self, ids: list[int]) -> str:
        return " ".join(self.word_of[i] for i in ids)

    # --- word-level API (tokens are strings), so the watermark can score any text ---

    def next_dist_words(self, prev_word: str | None) -> tuple[list[str], list[float]]:
        prev_id = self.id_of.get(prev_word) if prev_word is not None else None
        ids, wts = self.next_dist(prev_id)
        return [self.word_of[i] for i in ids], wts

    def sample_next_word(self, prev_word: str | None, rng: random.Random) -> str:
        words, wts = self.next_dist_words(prev_word)
        return rng.choices(words, weights=wts, k=1)[0]

    def start_word(self, rng: random.Random) -> str:
        return self.word_of[self.start_token(rng)]
