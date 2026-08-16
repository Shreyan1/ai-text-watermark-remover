"""Minimal CLI, wires registered adapters into a Pipeline by name.

Two useful commands work today with zero extra dependencies:
  * `reweave scrub` , Stage ① only (artifact hygiene)
  * `reweave score` , Stage ④ only (human-signature heuristic)

`reweave run` needs a real extractor + regenerator (the model-backed stages) and
will tell you so until those backends are installed.
"""

from __future__ import annotations

import argparse
import sys

from .core.registry import (
    resolve,
    KIND_SCRUBBER,
    KIND_SCORER,
    KIND_FACTCHECKER,
    KIND_METADATA,
)
from .core.types import Document

# Import built-in adapters so their @register decorators fire. The core never
# imports the edge; the application entrypoint does (dependency-inward, I3).
from . import scrub as _scrub  # noqa: F401
from . import score as _score  # noqa: F401
from . import extract as _extract  # noqa: F401
from . import regenerate as _regenerate  # noqa: F401
from . import guard as _guard  # noqa: F401
from . import verify as _verify  # noqa: F401


def _read(path: str | None) -> str:
    if path and path != "-":
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    return sys.stdin.read()


def cmd_scrub(args: argparse.Namespace) -> int:
    scrubber = resolve(KIND_SCRUBBER, args.adapter)()
    out = scrubber.scrub(Document(text=_read(args.input)))
    sys.stdout.write(out.text)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    scorer = resolve(KIND_SCORER, args.adapter)()
    sig = scorer.score(Document(text=_read(args.input)))
    print(f"human-signature: {sig.score:.3f}  ({sig.verdict.value})")
    if sig.rationale.get("abstained"):
        print(f"  ! abstained, only {sig.rationale['coverage']:.0%} of feature weight "
              f"available (text too short for burstiness); score is unreliable")
    for k, v in sig.rationale.get("contributions", {}).items():
        print(f"  {k:>26}: {v:+.3f}")
    return 0


def cmd_meta(args: argparse.Namespace) -> int:
    """Stage ⓪ - provenance in the file, not the prose."""
    from .scrub.metadata_scrubber import XattrUnsupported

    scrubber = resolve(KIND_METADATA, args.adapter)(rename=args.rename)
    try:
        report = scrubber.inspect(args.path) if args.dry_run else scrubber.scrub_file(args.path)
    except XattrUnsupported as e:
        print(f"! {e}", file=sys.stderr)
        return 2

    verb = "found" if args.dry_run else "handled"
    print(f"{args.path}: {report.summary()}")
    if report.clean:
        return 0

    for f in (report.findings if args.dry_run else report.removed):
        print(f"  {'·' if args.dry_run else '✓'} {f}")
    for f in report.unremovable:
        print(f"  ! SURVIVED  {f}")

    if getattr(scrubber, "renamed_to", None):
        print(f"  -> renamed to {scrubber.renamed_to}")
    elif any(f.layer == "filename" for f in report.unremovable):
        suggested = scrubber.suggested_name(args.path)
        print(f"    the filename itself names the source; suggested: {suggested!r}")
        print("    (pass --rename to apply; not done implicitly)")

    if report.unremovable:
        print("\n  Some traces were not removed. macOS re-creates "
              "com.apple.provenance on write;\n  PDF fields need a library that "
              "rebuilds the xref table. Reported, not hidden.")
    if args.dry_run:
        print(f"\n  ({verb} only - re-run without --dry-run to remove)")
    return 0


def cmd_facts(args: argparse.Namespace) -> int:
    if args.nli:
        from .guard import OllamaEmbeddingGuard
        from .verify import CompositeChecker, ConstraintChecker, NLIChecker, OllamaNLIBackend
        checker = CompositeChecker(
            ConstraintChecker(),
            # Embedding alignment is not optional: with lexical alignment the
            # reworded pairs never reach the model (8/8 -> 2/8, measured).
            NLIChecker(OllamaNLIBackend(model=args.nli_model),
                       embedder=OllamaEmbeddingGuard().embed),
        )
    else:
        checker = resolve(KIND_FACTCHECKER, args.adapter)()
    with open(args.source, encoding="utf-8") as fh:
        src = Document(text=fh.read())
    cand = Document(text=_read(args.candidate))
    r = checker.check(src, cand)

    print(f"facts: {'PRESERVED' if r.ok else 'CORRUPTED'}, {r.summary()}")
    print(f"  numerals kept: {r.numeral_coverage:.0%}   names kept: {r.entity_coverage:.0%}")
    for s, c in r.inversions:
        print("\n  INVERTED CLAIM")
        print(f"    source   : {s}")
        print(f"    candidate: {c}")
    if not r.ok and not args.nli:
        print("\n  NOTE: rules only, this catches negation and known antonym flips.")
        print("  A claim reversed in entirely different words needs --nli (measured 0/8 -> 8/8).")
    return 0 if r.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    print(
        "reweave run needs a model-backed extractor and regenerator.\n"
        "Install the local model extra and select real adapters:\n"
        "  pip install -e '.[regenerate]'\n"
        "Until then, 'reweave scrub' and 'reweave score' work standalone.",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reweave", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("meta", help="Stage ⓪ - strip file provenance (xattrs, frontmatter)")
    pm.add_argument("path", help="file to inspect or clean")
    pm.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    pm.add_argument("--rename", action="store_true",
                    help="also rename the file if it names its source (e.g. ChatGPT-*.md)")
    pm.add_argument("--adapter", default="file")
    pm.set_defaults(func=cmd_meta)

    ps = sub.add_parser("scrub", help="Stage ① - character/encoding hygiene")
    ps.add_argument("input", nargs="?", default="-")
    ps.add_argument("--adapter", default="unicode")
    ps.set_defaults(func=cmd_scrub)

    pc = sub.add_parser("score", help="Stage ④, human-signature heuristic")
    pc.add_argument("input", nargs="?", default="-")
    pc.add_argument("--adapter", default="statistical")
    pc.set_defaults(func=cmd_score)

    pf = sub.add_parser("facts", help="⑤, did a rewrite keep the source's facts?")
    pf.add_argument("source", help="the original text")
    pf.add_argument("candidate", nargs="?", default="-", help="the rewrite (default: stdin)")
    pf.add_argument("--adapter", default="constraint")
    pf.add_argument("--nli", action="store_true",
                    help="also run entailment (catches reworded reversal; needs Ollama)")
    pf.add_argument("--nli-model", default="gemma3:4b")
    pf.set_defaults(func=cmd_facts)

    pr = sub.add_parser("run", help="full pipeline (needs model backends)")
    pr.add_argument("input", nargs="?", default="-")
    pr.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
