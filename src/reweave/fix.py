"""`reweave fix` - the whole workflow as one bounded convergence loop.

The five commands people run by hand,

    meta --dry-run  ->  meta --rename  ->  scrub  ->  score  ->  facts

collapse into a single pass here, repeated until the file stops changing or a
pass budget runs out. It is bounded by construction, three independent ways:

  1. the deterministic stages (⓪ metadata, ① unicode) are idempotent, so a
     second pass over an already-clean file finds nothing to do;
  2. the loop stops the instant a pass makes no change (a fixed point), so it
     never spins waiting on something it cannot fix;
  3. `max_passes` is a hard backstop, mostly for the regeneration path.

Offline by default. Metadata stripping and unicode scrubbing need no model and
no network, and the Ollama-backed stages (③ extract/regenerate, the embedding
guard) are imported and built ONLY when `regenerate=True`. So `reweave fix` with
no flags never touches the network, which is the point of having it.

What "fixed" means differs by stage, and the report says so honestly:
  * metadata and unicode either converge to clean or the residual is named;
  * the human-signature score can only be nudged by scrubbing punctuation. Truly
    moving it needs a rewrite, so without --regenerate the score is REPORTED, not
    looped on (looping cannot change it). With --regenerate the rewrite runs
    under the fact gate: a candidate that flips a fact or drifts in meaning is
    rejected, never written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core.types import Document, VoiceProfile
from .scrub.metadata_scrubber import FileMetadataScrubber
from .scrub.unicode_scrubber import UnicodeScrubber
from .score import StatisticalScorer


@dataclass
class FixPass:
    """One trip through the stages, for the convergence report."""

    index: int
    meta_removed: int = 0
    meta_survived: int = 0
    renamed_to: str | None = None
    scrub_changed: bool = False
    regenerated: bool = False
    facts_ok: bool | None = None
    similarity: float | None = None
    score: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.meta_removed or self.renamed_to
                    or self.scrub_changed or self.regenerated)


@dataclass
class FixResult:
    path: str                       # final path (may differ after --rename)
    passes: list[FixPass] = field(default_factory=list)
    converged: bool = False
    final_score: float = 0.0
    regenerate: bool = False
    residual: list[str] = field(default_factory=list)  # what is honestly not fixed


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _meta_residual(path: str, rename: bool) -> list:
    """Provenance still present that we WOULD strip in this mode.

    Excludes a filename tag when --rename was not given (we chose to leave it,
    it is not a failure to converge) and format traces we never claim to remove
    (PDF fields). Everything else remaining means the loop has not finished.
    """
    out = []
    for f in FileMetadataScrubber().inspect(path).findings:
        if not f.removable:
            continue
        if f.layer == "filename" and not rename:
            continue
        out.append(f)
    return out


def run_fix(
    path: str,
    *,
    rename: bool = False,
    regenerate: bool = False,
    model: str = "gemma3:4b",
    allow_unlisted: bool = False,
    threshold: float = 0.70,
    similarity_floor: float = 0.82,
    max_passes: int = 3,
    host: str = "http://localhost:11434",
    start_aggressiveness: float = 0.45,
    aggressiveness_step: float = 0.18,
) -> FixResult:
    scorer = StatisticalScorer()
    voice = VoiceProfile()
    result = FixResult(path=path, regenerate=regenerate)
    regenerate_requested = regenerate  # may be cleared below if Ollama is down

    # Build the model-backed stages ONCE, lazily, and only if asked. This import
    # is what keeps the default path offline: nothing here loads unless the
    # caller opted into regeneration.
    extractor = regenerator = guard = fact_checker = None
    if regenerate:
        from ._ollama import is_up
        from .extract.ollama_extractor import OllamaExtractor
        from .guard import OllamaEmbeddingGuard
        from .regenerate.ollama_regenerator import OllamaRegenerator
        from .verify import ConstraintChecker

        if not is_up(host):
            result.residual.append(
                f"--regenerate needs Ollama at {host}, which is not reachable; "
                "ran the offline stages only")
            regenerate = False
            result.regenerate = False
        else:
            # The regenerator is the one that EMITS text, so it is the one the
            # watermark allowlist gates. Extraction shares the model so a single
            # local pull covers the whole path.
            regenerator = OllamaRegenerator(model=model, host=host,
                                            allow_unlisted=allow_unlisted)
            extractor = OllamaExtractor(model=model, host=host)
            guard = OllamaEmbeddingGuard(host=host)
            fact_checker = ConstraintChecker()

    for i in range(1, max_passes + 1):
        fp = FixPass(index=i)

        # ⓪ Metadata: strip xattrs / frontmatter / inline, optionally rename.
        meta = FileMetadataScrubber(rename=rename)
        mreport = meta.scrub_file(path)
        fp.meta_removed = len(mreport.removed)
        fp.meta_survived = len([f for f in mreport.unremovable if f.removable
                                and not (f.layer == "filename" and not rename)])
        if meta.renamed_to:
            fp.renamed_to = meta.renamed_to
            path = meta.renamed_to
            result.path = path

        # ① Unicode: strip invisibles / homoglyphs, normalise punctuation.
        text = _read(path)
        scrubbed = UnicodeScrubber().scrub(Document(text=text)).text
        if scrubbed != text:
            _write(path, scrubbed)
            fp.scrub_changed = True
            text = scrubbed

        # ④ Score the current state.
        sig = scorer.score(Document(text=text))
        fp.score = sig.score

        # ③+⑤ Optional rewrite, only while below threshold, always fact-gated.
        if regenerate and sig.score < threshold:
            meaning = extractor.extract(Document(text=text))
            aggr = min(1.0, start_aggressiveness + (i - 1) * aggressiveness_step)
            cand = regenerator.regenerate(meaning, voice, aggr)
            sim = guard.similarity(Document(text=text), cand)
            facts = fact_checker.check(Document(text=text), cand)
            fp.similarity = sim
            fp.facts_ok = facts.ok
            # Accept only a candidate that keeps the meaning AND the facts AND
            # actually scores better. Otherwise the original text stands.
            if sim >= similarity_floor and facts.ok:
                cand_sig = scorer.score(cand)
                if cand_sig.score > sig.score:
                    _write(path, cand.text)
                    text = cand.text
                    sig = cand_sig
                    fp.regenerated = True
            fp.score = sig.score

        result.passes.append(fp)
        result.final_score = sig.score

        # Convergence: deterministic stages clean, and either we are not chasing
        # the score or it has cleared the bar.
        meta_clean = not _meta_residual(path, rename)
        unicode_clean = UnicodeScrubber().scrub(Document(text=text)).text == text
        score_ok = (not regenerate) or sig.score >= threshold
        if meta_clean and unicode_clean and score_ok:
            result.converged = True
            break
        # Nothing changed this pass and we are still not converged: further
        # passes would repeat identically, so stop rather than burn the budget.
        if not fp.changed:
            break

    # Honest residual for whatever did not reach "clean".
    for f in _meta_residual(path, rename):
        result.residual.append(f"metadata still present: {f}")
    if result.final_score < threshold:
        if regenerate:
            result.residual.append(
                f"human-signature {result.final_score:.3f} below {threshold:.2f} after "
                f"{len(result.passes)} pass(es); the rewrite could not clear the bar")
        elif regenerate_requested:
            # Asked for a rewrite, but Ollama was unreachable (noted above).
            result.residual.append(
                f"human-signature {result.final_score:.3f} below {threshold:.2f}; start "
                "Ollama and re-run with --regenerate to lift it, or edit by hand")
        else:
            result.residual.append(
                f"human-signature {result.final_score:.3f} below {threshold:.2f}; this is "
                "prose-level and cannot be scrubbed. Re-run with --regenerate, or edit by hand")
    return result
