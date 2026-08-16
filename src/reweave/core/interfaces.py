"""The five stage contracts.

STABLE CORE. These abstract base classes are the seam between the durable core
and the swappable edge (Invariant I3). Adapters at the edge implement these; the
core orchestrates them and knows nothing else about them.

Adding a new model, scorer, or scrubber = implement one of these + register it
(see registry.py). It must NEVER require editing the core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import (
    Constraints,
    Document,
    FactReport,
    HumanSignature,
    Meaning,
    MetadataReport,
    VoiceProfile,
)


class Scrubber(ABC):
    """Stage ①. Deterministic character/encoding hygiene. Pure code, no model.
    Defeats the character-artifact watermark family (threat-model Family A)."""

    name: str = "scrubber"

    @abstractmethod
    def scrub(self, doc: Document) -> Document:
        ...


class MetadataScrubber(ABC):
    """Stage ⓪. Provenance that lives OUTSIDE the prose.

    Stage ① cleans characters inside the text. This cleans everything wrapped
    around it: filesystem extended attributes, container metadata (DOCX, PDF),
    YAML front matter, and generator comments. These survive regeneration
    completely, you can rewrite every word and the file still says where it
    came from, so they need their own pass.

    Separate from `Scrubber` because it works on PATHS, not `Document`. A
    Document has already lost the file it came from, and that file is exactly
    where this metadata lives.
    """

    name: str = "metadata"

    @abstractmethod
    def inspect(self, path: str) -> MetadataReport:
        """Report provenance traces without modifying anything."""
        ...

    @abstractmethod
    def scrub_file(self, path: str) -> MetadataReport:
        """Remove what can be removed; report what cannot."""
        ...


class Extractor(ABC):
    """Stage ②. Reduce a document to its Meaning. Discarding surface form here is
    what severs the token-sequence substrate (Invariant I4)."""

    name: str = "extractor"

    @abstractmethod
    def extract(self, doc: Document) -> Meaning:
        ...


class Regenerator(ABC):
    """Stage ③. Rebuild prose from Meaning, steered by VoiceProfile.

    HARD INVARIANT: implementations MUST use a local, unwatermarked, open-weight
    model. Routing this through a watermarked/hosted frontier model re-stamps a
    fresh watermark and defeats the entire pipeline (the Self-Watermark Trap).
    An implementation that calls a watermarked API is a bug, not a variant.

    `aggressiveness` (0..1) controls how far from the source surface to roam;
    the convergence loop turns it up on retries.
    """

    name: str = "regenerator"
    #: Implementations MUST set this truthfully. The pipeline refuses to run a
    #: regenerator whose backing model is watermarked.
    is_unwatermarked: bool = False

    @abstractmethod
    def regenerate(
        self,
        meaning: Meaning,
        voice: VoiceProfile,
        aggressiveness: float = 0.5,
    ) -> Document:
        ...


class Scorer(ABC):
    """Stage ④. Compute the human-signature heuristic. Statistical, no training,
    no scheme knowledge. This is the achievable half of 'detection', used only as
    an internal quality gate (Invariant I5)."""

    name: str = "scorer"

    @abstractmethod
    def score(self, doc: Document, voice: VoiceProfile | None = None) -> HumanSignature:
        ...


class SemanticGuard(ABC):
    """Meaning-preservation floor. Removal that garbles content is a failure. The
    gate rejects any regeneration whose similarity to the source drops below the
    configured floor."""

    name: str = "guard"

    @abstractmethod
    def similarity(self, a: Document, b: Document) -> float:
        """Return semantic similarity in 0..1 (1 = identical meaning)."""
        ...


class FactChecker(ABC):
    """The gate's second, orthogonal test, did the facts survive?

    A `SemanticGuard` measures topic distance and is blind to truth value: the
    harness measured "X are computers" vs "X are NOT computers" at 0.959
    similarity. Passing the guard therefore does not mean the claims survived.
    This contract closes that gap: numbers, names, and the polarity of each
    claim must still hold in the candidate.

    Implementations should be deterministic and dependency-free where possible.
    A fact check that can silently fail is worse than none, because it converts
    an known unknown into a false assurance (Invariant I5).
    """

    name: str = "factchecker"

    @abstractmethod
    def constraints(self, doc: Document) -> Constraints:
        """Extract the must-keep facts from a document."""
        ...

    @abstractmethod
    def check(self, source: Document, candidate: Document) -> FactReport:
        """Verify the source's constraints survive in the candidate."""
        ...
