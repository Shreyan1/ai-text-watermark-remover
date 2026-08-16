"""Smoke tests — prove the stable core + working stages hold together.

These use only zero-dependency pieces (scrubber, scorer) plus the wiring stubs,
so they run with no models installed. They validate the CONTRACTS, not removal
quality (that needs the harness + a real regenerator).
"""

from __future__ import annotations

from reweave import Document, Pipeline, PipelineConfig, VoiceProfile
from reweave.core.registry import KIND_REGENERATOR, available, resolve
from reweave.extract import SentenceFallbackExtractor
from reweave.guard import JaccardGuard
from reweave.regenerate import EchoStubRegenerator
from reweave.score import StatisticalScorer
from reweave.scrub import UnicodeScrubber
from reweave.verify import (
    CompositeChecker,
    ConstraintChecker,
    NLIBackend,
    NLIChecker,
    NLIScores,
)


class _FakeNLI(NLIBackend):
    """Deterministic stand-in so the NLI contract is tested without a model."""

    name = "fake"

    def __init__(self, label: str = "neutral") -> None:
        self.label = label
        self.seen: list[tuple[str, str]] = []

    def predict(self, premise: str, hypothesis: str) -> NLIScores:
        self.seen.append((premise, hypothesis))
        return NLIScores.one_hot(self.label)


def test_scrubber_strips_invisibles_and_homoglyphs():
    dirty = "Hеllo​world​— test"  # Cyrillic 'е', ZWSP, em dash, NBSP
    out = UnicodeScrubber().scrub(Document(text=dirty))
    assert "​" not in out.text
    assert " " not in out.text
    assert "—" not in out.text  # normalised to hyphen
    assert out.text.startswith("Hello")  # homoglyph folded
    assert out.meta["scrubbed"] is True


def test_scorer_ranks_human_above_ai():
    ai = (
        "Leadership is important. Communication is important. Teamwork is important. "
        "Furthermore, it is a testament to robust synergy. Moreover, we must leverage "
        "our ecosystem to unlock a seamless landscape."
    )
    human = (
        "I fired the whole plan on a Tuesday. Rebuilt it in three days flat, running "
        "on cold coffee and spite. It worked — barely — and the client never noticed "
        "the seams. That near-miss taught me more than any postmortem ever has."
    )
    s = StatisticalScorer()
    assert s.score(Document(text=human)).score > s.score(Document(text=ai)).score


def test_registry_discovers_adapters():
    regs = available(KIND_REGENERATOR)
    assert (KIND_REGENERATOR, "echo-stub") in regs
    assert resolve(KIND_REGENERATOR, "echo-stub") is EchoStubRegenerator


def test_pipeline_refuses_watermarked_regenerator():
    class Watermarked(EchoStubRegenerator):
        name = "watermarked"
        is_unwatermarked = False

    try:
        Pipeline(
            UnicodeScrubber(), SentenceFallbackExtractor(), Watermarked(),
            StatisticalScorer(), JaccardGuard(),
        )
    except ValueError as e:
        assert "Self-Watermark Trap" in str(e)
    else:
        raise AssertionError("pipeline must refuse a watermarked regenerator")


def test_fact_checker_catches_the_embedding_blind_spot():
    """The three pairs that scored 0.776-0.959 on the embedding guard."""
    c = ConstraintChecker()
    inversions = [
        ("Xbox and PlayStation are computers that run an operating system.",
         "Xbox and PlayStation aren't computers that run an operating system."),
        ("The deployment succeeded and the service came back at full capacity.",
         "The deployment failed and the service came back at full capacity."),
        ("Revenue increased 40% in the third quarter on enterprise renewals.",
         "Revenue decreased 40% in the third quarter on enterprise renewals."),
    ]
    for src, cand in inversions:
        r = c.check(Document(text=src), Document(text=cand))
        assert not r.ok, f"missed inversion: {cand}"
        assert r.inversions


def test_fact_checker_accepts_faithful_reword():
    """A gate that rejects legitimate rewrites shuts the pipeline down."""
    c = ConstraintChecker()
    r = c.check(
        Document(text="Revenue increased 40% in the third quarter, driven by "
                      "enterprise renewals."),
        Document(text="Enterprise renewals drove the quarter. Revenue climbed "
                      "40% in Q3."),
    )
    assert r.ok, r.summary()


def test_fact_checker_ignores_pseudo_negation():
    """NegEx precedence: 'not only/just' ADDS to a claim, it does not deny it."""
    c = ConstraintChecker()
    r = c.check(
        Document(text="The ordering takes into account the alphabetical order "
                      "of the letter names."),
        Document(text="The ordering is not just about the alphabetical order of "
                      "the letter names; it also reflects them."),
    )
    assert r.ok, r.summary()


def test_fact_checker_is_self_consistent():
    """A document must never contradict itself — the invariant that caught the
    degenerate-alignment bug on real markdown output."""
    c = ConstraintChecker()
    text = (
        "**Hardware:**\n\n* **Closed systems:** Xbox and PlayStation are closed "
        "systems, so you don't get the same access as on PCs.\n"
        "* **Tools:** Microsoft and Sony provide the official SDKs.\n\n"
        "1. Habitat loss: species in habitats destroyed at a rapid rate are more "
        "endangered.\n"
    )
    r = c.check(Document(text=text), Document(text=text))
    assert r.ok, r.summary()


def test_pipeline_gate_rejects_fact_corrupting_candidate():
    """The fact checker must actually veto in the pipeline, not just report."""
    class Inverter(EchoStubRegenerator):
        name = "inverter"
        is_unwatermarked = True

        def regenerate(self, meaning, voice, aggressiveness=0.5):
            return Document(text="The deployment failed. The rollout was halted "
                                 "before the release went out to customers.")

    src = Document(text="The deployment succeeded. The rollout was completed "
                        "before the release went out to customers.")
    pipe = Pipeline(
        UnicodeScrubber(), SentenceFallbackExtractor(), Inverter(),
        StatisticalScorer(), JaccardGuard(),
        config=PipelineConfig(max_iterations=1, similarity_floor=0.0,
                              human_threshold=0.0),
        fact_checker=ConstraintChecker(),
    )
    result = pipe.run(src, VoiceProfile())
    assert "succeeded" in result.output.text, "gate let a fact inversion through"
    assert result.facts is not None


def test_nli_checker_flags_contradiction():
    src = Document(text="Sales climbed steadily through the summer months.")
    cand = Document(text="Sales were disappointing throughout the summer months.")
    backend = _FakeNLI("contradiction")
    r = NLIChecker(backend).check(src, cand)
    assert backend.seen, "the aligned pair never reached the NLI backend"
    assert not r.ok and r.inversions


def test_nli_checker_passes_entailment():
    src = Document(text="Sales climbed steadily through the summer months.")
    cand = Document(text="Sales rose through the summer months, steadily.")
    r = NLIChecker(_FakeNLI("entailment")).check(src, cand)
    assert r.ok and not r.inversions


def test_composite_rules_stand_alone_when_nli_fails():
    """A broken NLI backend must never weaken the deterministic guarantee."""
    class Broken(NLIBackend):
        name = "broken"

        def predict(self, premise, hypothesis):
            raise RuntimeError("backend down")

    src = Document(text="Revenue increased 40% in the third quarter this year.")
    cand = Document(text="Revenue decreased 40% in the third quarter this year.")
    comp = CompositeChecker(ConstraintChecker(), NLIChecker(Broken()), fail_open=True)
    r = comp.check(src, cand)
    assert not r.ok, "rules finding was lost when NLI errored"
    assert r.detail.get("nli_ran") is False

    # And by default the failure is loud, not silent.
    strict = CompositeChecker(ConstraintChecker(), NLIChecker(Broken()))
    try:
        strict.check(src, cand)
    except RuntimeError:
        pass
    else:
        raise AssertionError("fail_open=False must propagate backend errors")


def test_composite_adds_nli_findings_to_rules():
    src = Document(text="The migration finished ahead of its published schedule.")
    cand = Document(text="The migration overran the published schedule badly.")
    rules_only = ConstraintChecker().check(src, cand)
    assert rules_only.ok, "precondition: rules cannot see this reword"

    comp = CompositeChecker(ConstraintChecker(), NLIChecker(_FakeNLI("contradiction")))
    r = comp.check(src, cand)
    assert not r.ok and r.detail["nli_only_inversions"] >= 1


def test_pipeline_runs_end_to_end_with_stubs():
    pipe = Pipeline(
        UnicodeScrubber(), SentenceFallbackExtractor(), EchoStubRegenerator(),
        StatisticalScorer(), JaccardGuard(),
        config=PipelineConfig(max_iterations=2),
    )
    result = pipe.run(Document(text="This is a test. It has two sentences."), VoiceProfile())
    assert result.output.text
    assert result.before is not None and result.after is not None
    assert result.iterations >= 0
    assert len(result.trace) >= 2
