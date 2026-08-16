"""Data contracts, the spine of the pipeline.

STABLE CORE. No third-party dependencies, no watermark-scheme code. Every type
here is immutable; the pipeline threads them stage to stage. If a stage needs a
new field, extend the type here, never smuggle stage-specific state through
`meta` dicts as a shortcut, because that erodes the contract (Invariant I3).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping


class Verdict(str, Enum):
    """Honest labels for the human-signature heuristic. Never 'watermarked' —
    we cannot know that keyless (Invariant I5)."""

    HUMAN_LIKE = "human-like"
    AI_LIKE = "ai-like"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Document:
    """A unit of text plus what little we know about its provenance.

    `meta` is for descriptive hints (source, language, model guess), never for
    threading pipeline state between stages.
    """

    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def with_text(self, text: str) -> "Document":
        return replace(self, text=text)


@dataclass(frozen=True)
class Point:
    """One atomic unit of meaning: a claim, fact, instruction, or intent,
    stripped of surface form. The thing that survives when tokens are discarded.
    """

    intent: str  # what this conveys, in the extractor's own words
    salience: float = 1.0  # relative importance, 0..1
    constraints: Mapping[str, Any] = field(default_factory=dict)  # must-keep facts, numbers, names


@dataclass(frozen=True)
class Meaning:
    """The pivot representation. Regenerating from this, rather than editing the
    original surface, is what severs the watermark substrate (Invariant I4).

    Deliberately loose so extractors can evolve without a contract change.
    """

    points: tuple[Point, ...]
    order: str = "as-given"  # "as-given" | "logical" | "free", how strictly to preserve sequence
    register: str = "neutral"  # detected register hint for the regenerator
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureVector:
    """The measurable human/AI signals. OPEN by design (Invariant I1): add fields
    freely; the scorer blends whatever is present. `extra` holds
    experimental features before they earn a first-class slot.
    """

    perplexity_mean: float | None = None
    perplexity_var: float | None = None
    burstiness: float | None = None  # std-dev of sentence length
    syntactic_burstiness: float | None = None  # std-dev of parse depth
    type_token_ratio: float | None = None
    paragraph_cv: float | None = None  # coefficient of variation of paragraph lengths
    em_dash_rate: float | None = None
    rule_of_three_rate: float | None = None
    entity_density: float | None = None
    numeral_density: float | None = None
    extra: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HumanSignature:
    """Our honest, internal 'detector'. `score` is 0..1, higher = more human-like.
    Used as a quality gate; never exposed as watermark detection."""

    score: float
    verdict: Verdict
    features: FeatureVector
    rationale: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Constraints:
    """The must-keep facts of a text, extracted deterministically from its surface.

    This is the half of meaning that embeddings cannot see. Sentence embeddings
    encode *topic*, so "revenue increased 40%" and "revenue decreased 40%" sit
    0.776 apart, same topic, opposite truth. Constraints capture the truth part:
    the numbers, the names, and the polarity of each claim.

    Deliberately surface-derived and zero-dependency: unlike the meaning pivot,
    facts are things a regex can actually pin down, and a check that cannot fail
    silently is worth more than a cleverer one that can.
    """

    numerals: frozenset[str] = frozenset()  # normalised numeric values ("40", "1000")
    entities: frozenset[str] = frozenset()  # proper nouns, lowercased for matching
    #: (content-word key, is_negated) per claim, the polarity fingerprint.
    claims: tuple[tuple[frozenset[str], bool], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.numerals or self.entities or self.claims)


@dataclass(frozen=True)
class FactReport:
    """Did the regeneration keep the facts? The gate's second, orthogonal test.

    `SemanticGuard` answers "is this still about the same thing?"; this answers
    "is it still saying the same thing about it?". A candidate must pass both.
    """

    ok: bool
    missing_numerals: tuple[str, ...] = ()
    missing_entities: tuple[str, ...] = ()
    #: (source claim, candidate claim) pairs whose polarity flipped.
    inversions: tuple[tuple[str, str], ...] = ()
    numeral_coverage: float = 1.0
    entity_coverage: float = 1.0
    detail: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return "facts preserved"
        parts = []
        if self.inversions:
            parts.append(f"{len(self.inversions)} inverted claim(s)")
        if self.missing_numerals:
            parts.append(f"dropped numbers: {', '.join(self.missing_numerals)}")
        if self.missing_entities:
            parts.append(f"dropped names: {', '.join(self.missing_entities)}")
        return "; ".join(parts) or "fact check failed"


@dataclass(frozen=True)
class VoiceProfile:
    """The target to regenerate INTO. The regenerator is steered by this; the
    scorer's weights may be derived from it. Sample texts are the strongest
    signal, the user's actual writing."""

    contractions: bool = True
    target_burstiness: float | None = None  # None = "as human as possible"
    vocabulary: frozenset[str] = frozenset()  # words the author actually uses
    banned_terms: frozenset[str] = frozenset()  # AI tells: delve, tapestry, seamless, …
    sample_texts: tuple[str, ...] = ()  # the author's own writing, for style transfer
    meta: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_BANNED_TERMS: frozenset[str] = frozenset({
    "delve", "showcase", "leverage", "ecosystem", "tapestry", "robust",
    "seamless", "landscape", "navigate", "unlock", "testament", "realm",
    "underscore", "elevate", "boasts", "furthermore", "moreover",
})


@dataclass(frozen=True)
class MetadataFinding:
    """One provenance trace found outside the prose.

    Text scrubbing is not enough. A markdown file saved from a chat UI carries
    its origin in the *filesystem*: on macOS, `kMDItemWhereFroms` holds the exact
    source URL and `com.apple.quarantine` names the downloading app. Neither is
    visible in an editor, neither is touched by rewriting a single word, and both
    survive every stage of this pipeline. Verified on a real download here:

        com.apple.metadata:kMDItemWhereFroms -> https://www.nature.com/...
        com.apple.quarantine                 -> 0083;6a7aef14;Preview;

    A pipeline that rewrites the text and leaves that in place has not removed
    provenance; it has only removed the part you could see.
    """

    layer: str    # "xattr" | "frontmatter" | "inline" | "docx" | "pdf"
    key: str      # attribute / field name
    value: str    # what it revealed (truncated for display)
    removable: bool = True

    def __str__(self) -> str:
        v = self.value if len(self.value) <= 90 else self.value[:87] + "..."
        return f"[{self.layer}] {self.key}: {v}"


@dataclass(frozen=True)
class MetadataReport:
    """What provenance a file carries, and what was removed."""

    path: str
    findings: tuple[MetadataFinding, ...] = ()
    removed: tuple[MetadataFinding, ...] = ()
    #: Traces we can see but cannot strip without rewriting the file format.
    unremovable: tuple[MetadataFinding, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return "no provenance metadata found"
        parts = [f"{len(self.findings)} trace(s)"]
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.unremovable:
            parts.append(f"{len(self.unremovable)} NOT removable")
        return ", ".join(parts)


@dataclass(frozen=True)
class StageTrace:
    """One stage's record, for auditability (Invariant I5)."""

    stage: str
    adapter: str
    note: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformResult:
    """The end-to-end record. Everything measured, nothing unverifiable claimed."""

    original: Document
    output: Document
    before: HumanSignature | None
    after: HumanSignature | None
    semantic_similarity: float | None
    iterations: int
    converged: bool
    trace: tuple[StageTrace, ...] = ()
    #: None when no fact checker was wired in.
    facts: FactReport | None = None

    @property
    def improved(self) -> bool:
        if self.before is None or self.after is None:
            return False
        return self.after.score > self.before.score
