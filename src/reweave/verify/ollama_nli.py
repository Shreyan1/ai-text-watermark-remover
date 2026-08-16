"""NLI via a local instruct model, the backend that works with what's installed.

This is an LLM prompted to emit one of three labels, NOT a trained NLI
cross-encoder. That distinction is worth stating plainly because it changes what
the numbers mean:

  * A cross-encoder trained on MNLI(+VitaminC) returns a calibrated distribution
    over the three classes. You can threshold it.
  * An instruct model returns a hard label. Confidence has to come from
    self-consistency (sample k times, count votes), which is what `votes > 1`
    does here, at k× the cost.

Why it is still the right default: it needs no torch, no transformers, no model
download beyond an Ollama pull, and it keeps the zero-dependency promise of the
core intact (I3, the heavy option lives behind the `[verify]` extra). And for
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

from .._ollama import DEFAULT_HOST, assert_local, generate
from .nli import NLIBackend, NLIScores

# Asking for a bare one-word answer does NOT work across model families:
# reasoning models (qwen3, deepseek-r1) are trained to think first and will spend
# the whole budget doing so. Rather than fight that, reasoning genuinely helps
# NLI accuracy, we let the model reason and require a DELIMITED final answer.
# One protocol, works for both families, and the delimiter makes parsing exact
# instead of hunting for a label inside prose that names all three.
#
# The prompt below is the "strict-neutral" wording, chosen by measurement. The
# original prompt just defined the three classes; on SNLI it retreated to NEUTRAL
# whenever unsure, which is where nearly all the missed contradictions went
# (confusion matrix: contradiction -> neutral was the dominant error). Naming
# NEUTRAL as the narrow label rather than the safe one, and forcing the
# "could both be true at the same moment?" test, lifted contradiction recall
# from 0.43 to 0.51 at a cost of a single false positive in 100. See
# tests/harness/nli_prompt_sweep.py for the A/B/C/D comparison.
_SYSTEM = (
    "You are a natural language inference classifier. Given a PREMISE and a "
    "HYPOTHESIS, decide:\n"
    "ENTAILMENT - if the premise is true, the hypothesis must also be true\n"
    "CONTRADICTION - the premise and the hypothesis cannot both be true of the "
    "same situation\n"
    "NEUTRAL - both could be true at once; the hypothesis just adds detail the "
    "premise does not settle\n\n"
    "NEUTRAL is the narrowest label, not the safe one. Before choosing it, ask: "
    "could both sentences describe the same situation at the same moment? If they "
    "could not, the answer is CONTRADICTION, even when no negation word appears "
    "and even when only one detail conflicts. Different wording with the same "
    "meaning is ENTAILMENT.\n\n"
    "Think briefly if you need to, then end your reply with exactly:\n"
    "ANSWER: <ENTAILMENT|CONTRADICTION|NEUTRAL>"
)

# Opt-in higher-recall variant. Six labelled examples weighted toward hard
# contradictions push recall to ~0.68, but precision falls to ~0.78 (7 false
# positives in 100). For a gate that BLOCKS a user's rewrite a false positive is
# a rejected good rewrite, so this is not the default; it is offered for callers
# who would rather over-flag and review than miss an inversion.
_FEW_SHOT = (
    "\n\nExamples:\n"
    "PREMISE: A man is playing a guitar on stage.\n"
    "HYPOTHESIS: A man is performing music.\nANSWER: ENTAILMENT\n\n"
    "PREMISE: A man is playing a guitar on stage.\n"
    "HYPOTHESIS: A man is asleep in bed.\nANSWER: CONTRADICTION\n\n"
    "PREMISE: A man is playing a guitar on stage.\n"
    "HYPOTHESIS: The man is playing his own songs.\nANSWER: NEUTRAL\n\n"
    "PREMISE: Two children run across a grassy field.\n"
    "HYPOTHESIS: The children are sitting still indoors.\nANSWER: CONTRADICTION\n\n"
    "PREMISE: Sales climbed steadily through the summer.\n"
    "HYPOTHESIS: Sales were disappointing all summer.\nANSWER: CONTRADICTION\n\n"
    "PREMISE: A woman in a red coat waits at a bus stop.\n"
    "HYPOTHESIS: A woman waits for the bus in the rain.\nANSWER: NEUTRAL\n"
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
        high_recall: bool = False,
    ) -> None:
        # A judge emits a label, not prose, so it cannot re-watermark anything;
        # the allowlist is therefore not enforced here. But a hosted judge still
        # receives the user's source text, so refuse that.
        assert_local(model)
        self.model = model
        self.host = host
        #: high_recall appends few-shot examples: recall ~0.68 vs ~0.51, at the
        #: cost of precision ~0.78 vs ~0.95. Off by default because the gate
        #: blocks user work and a false positive rejects a good rewrite.
        self.system = _SYSTEM + (_FEW_SHOT if high_recall else "")
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
            system=self.system, think=self.think,
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
            for label in _LABELS:  # contradiction first, it contains no other label
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
                # First vote greedy, the rest sampled, otherwise every vote is
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
