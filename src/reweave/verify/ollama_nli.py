"""NLI via a local instruct model — the backend that works with what's installed.

This is an LLM prompted to emit one of three labels, NOT a trained NLI
cross-encoder. That distinction is worth stating plainly because it changes what
the numbers mean:

  * A cross-encoder trained on MNLI(+VitaminC) returns a calibrated distribution
    over the three classes. You can threshold it.
  * An instruct model returns a hard label. Confidence has to come from
    self-consistency (sample k times, count votes), which is what `votes > 1`
    does here — at k× the cost.

Why it is still the right default: it needs no torch, no transformers, no model
download beyond an Ollama pull, and it keeps the zero-dependency promise of the
core intact (I3 — the heavy option lives behind the `[verify]` extra). And for
short claim pairs, which is all this checker ever asks about, a modern instruct
model is a genuinely strong zero-shot NLI classifier.

The model MUST be local and open-weight for the same reason Stage ③ must be —
except the reason here is privacy, not watermarking: this backend sees the user's
source text. A judge's output is a label, not prose, so it cannot re-watermark
anything; but sending the document to a hosted endpoint to be judged still sends
the document.
"""

from __future__ import annotations

import re

from .._ollama import DEFAULT_HOST, generate
from .nli import NLIBackend, NLIScores

# Asking for a bare one-word answer does NOT work across model families:
# reasoning models (qwen3, deepseek-r1) are trained to think first and will spend
# the whole budget doing so. Rather than fight that — reasoning genuinely helps
# NLI accuracy — we let the model reason and require a DELIMITED final answer.
# One protocol, works for both families, and the delimiter makes parsing exact
# instead of hunting for a label inside prose that names all three.
_SYSTEM = (
    "You are a natural language inference classifier. Given a PREMISE and a "
    "HYPOTHESIS, decide:\n"
    "ENTAILMENT - the hypothesis must be true if the premise is true\n"
    "CONTRADICTION - the hypothesis cannot be true if the premise is true\n"
    "NEUTRAL - neither; the hypothesis adds or omits information but does not "
    "conflict\n\n"
    "Judge only whether both can be true at once. Different wording with the "
    "same meaning is ENTAILMENT. Wording that reverses the direction, outcome, "
    "or polarity of a claim is CONTRADICTION — even with no negation word.\n\n"
    "Think briefly if you need to, then end your reply with exactly:\n"
    "ANSWER: <ENTAILMENT|CONTRADICTION|NEUTRAL>"
)

_LABELS = ("contradiction", "entailment", "neutral")
_ANSWER = re.compile(r"answer\s*:\s*\**\s*(entailment|contradiction|neutral)", re.I)


class OllamaNLIBackend(NLIBackend):
    """Zero-shot NLI by prompting a local open-weight instruct model."""

    name = "ollama-nli"
    #: One-hot output, so any non-zero contradiction mass is the whole vote.
    default_contradiction_threshold = 0.5

    def __init__(
        self,
        # A NON-reasoning instruct model by design. Measured on the gap set:
        # gemma3:4b answers in ~2.8s/pair with 0 parse failures; qwen3:4b (a
        # reasoning model) took 46s/pair and STILL ran out of budget mid-thought
        # on one, because it is trained to deliberate before answering. For a
        # per-claim gate, an answer beats an argument.
        model: str = "gemma3:4b",
        host: str = DEFAULT_HOST,
        votes: int = 1,
        num_predict: int = 400,
        think: bool | None = False,
    ) -> None:
        if "cloud" in model.lower():
            raise ValueError(
                f"refusing model {model!r}: a hosted model would receive the "
                "user's source text for judging. Use a local open-weight model."
            )
        self.model = model
        self.host = host
        #: >1 turns the hard label into a vote share (self-consistency), at cost.
        self.votes = max(1, votes)
        #: Must be generous: a reasoning model truncated mid-thought emits no
        #: answer at all, and a silently-unparsed pair defaults to "no finding".
        self.num_predict = num_predict
        self.think = think
        #: Unparseable replies are COUNTED, not swallowed. A backend quietly
        #: abstaining on every pair looks identical to a clean bill of health.
        self.parse_failures = 0
        self.calls = 0
        self._cache: dict[tuple[str, str], NLIScores] = {}

    def _ask(self, premise: str, hypothesis: str, temperature: float) -> str:
        self.calls += 1
        raw = generate(
            f"PREMISE: {premise}\nHYPOTHESIS: {hypothesis}",
            model=self.model, host=self.host,
            temperature=temperature, num_predict=self.num_predict,
            system=_SYSTEM, think=self.think,
        )
        m = None
        for m in _ANSWER.finditer(raw):  # last delimited answer wins
            pass
        if m:
            return m.group(1).lower()

        # No delimiter. Fall back to a bare label ONLY if the reply is short
        # enough that it cannot be reasoning that merely names the classes.
        low = raw.strip().lower()
        if len(low) <= 40:
            for label in _LABELS:  # contradiction first — it contains no other label
                if label in low:
                    return label
        self.parse_failures += 1
        return "neutral"  # abstain, but the failure is on the record

    def predict(self, premise: str, hypothesis: str) -> NLIScores:
        key = (premise, hypothesis)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        if self.votes == 1:
            scores = NLIScores.one_hot(self._ask(premise, hypothesis, 0.0))
        else:
            tally = {lab: 0 for lab in _LABELS}
            for i in range(self.votes):
                # First vote greedy, the rest sampled — otherwise every vote is
                # identical and self-consistency measures nothing.
                tally[self._ask(premise, hypothesis, 0.0 if i == 0 else 0.7)] += 1
            n = float(self.votes)
            scores = NLIScores(
                entailment=tally["entailment"] / n,
                neutral=tally["neutral"] / n,
                contradiction=tally["contradiction"] / n,
            )
        self._cache[key] = scores
        return scores
