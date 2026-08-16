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


def _shell_quote(path: str) -> str:
    return f'"{path}"' if " " in path else path


def cmd_meta(args: argparse.Namespace) -> int:
    """Stage ⓪ - provenance in the file, not the prose."""
    from .report import S, render_finding
    from .scrub.metadata_scrubber import XattrUnsupported

    scrubber = resolve(KIND_METADATA, args.adapter)(rename=args.rename)
    try:
        report = scrubber.inspect(args.path) if args.dry_run else scrubber.scrub_file(args.path)
    except XattrUnsupported as e:
        print(f"! {e}", file=sys.stderr)
        return 2

    print()
    print(S.b(args.path))
    if report.clean:
        print(f"  {S.green}no provenance found{S.off}, this file gives nothing away")
        print()
        return 0

    n = len(report.findings)
    if args.dry_run:
        removable = sum(1 for f in report.findings if f.removable)
        note = "all removable" if removable == n else f"{n - removable} cannot be removed"
        print(f"  {n} trace{'s' if n != 1 else ''} found, {note}")
    else:
        print(f"  {len(report.removed)} of {n} removed")
    print()

    shown = report.findings if args.dry_run else report.removed
    suggestion = scrubber.suggested_name(args.path)
    for i, f in enumerate(shown, 1):
        # Only offer the rename hint when a rename has not already happened.
        sugg = (suggestion if f.layer == "filename" and not getattr(
            scrubber, "renamed_to", None) else None)
        for line in render_finding(f, i, removed=not args.dry_run, suggested=sugg):
            print(line)

    survivors = [f for f in report.unremovable if f not in shown]
    if survivors:
        print(f"  {S.yellow}still present{S.off}")
        for f in survivors:
            print(f"    {f.key}: {f.value[:60]}")
        print(S.d("    PDF fields need a library that rebuilds the xref table."))
        print(S.d("    Reported rather than hidden."))
        print()

    # What to do next, spelled out as a command that can be pasted.
    if getattr(scrubber, "renamed_to", None):
        print(f"  {S.green}renamed{S.off} to {scrubber.renamed_to}")
    elif args.dry_run:
        has_name = any(f.layer == "filename" for f in report.findings)
        cmd = f"reweave meta {'--rename ' if has_name else ''}{_shell_quote(args.path)}"
        print(S.d("  nothing was changed (--dry-run)"))
        print(f"  to clean it:  {S.b(cmd)}")
    print()
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


def cmd_fix(args: argparse.Namespace) -> int:
    """The whole workflow as one bounded loop: strip -> scrub -> score (-> rewrite).

    Operates in place on the given file (like `meta` and `scrub` already do).
    Offline unless --regenerate is passed.
    """
    from ._ollama import WatermarkRiskError
    from .fix import run_fix
    from .scrub.metadata_scrubber import XattrUnsupported

    try:
        res = run_fix(
            args.path,
            rename=args.rename,
            regenerate=args.regenerate,
            model=args.model,
            allow_unlisted=args.allow_model,
            threshold=args.threshold,
            max_passes=args.max_passes,
        )
    except XattrUnsupported as e:
        print(f"! {e}", file=sys.stderr)
        return 2
    except WatermarkRiskError as e:
        print(f"! {e}", file=sys.stderr)
        return 2

    mode = "regenerate" if res.regenerate else "offline"
    print(f"{args.path}: fixing ({mode})")
    for p in res.passes:
        bits = []
        bits.append(f"metadata {p.meta_removed} removed" if p.meta_removed else "metadata clean")
        bits.append("unicode scrubbed" if p.scrub_changed else "unicode clean")
        if p.regenerated:
            bits.append(f"rewritten (sim {p.similarity:.2f}, facts "
                        f"{'ok' if p.facts_ok else 'FAILED'})")
        elif p.facts_ok is False:
            bits.append("rewrite rejected (facts flipped)")
        bits.append(f"score {p.score:.3f}")
        line = f"  pass {p.index}: " + ", ".join(bits)
        if p.renamed_to:
            line += f"\n           -> renamed to {p.renamed_to}"
        print(line)

    verdict = "converged" if res.converged else "stopped (budget or fixed point)"
    print(f"{verdict} after {len(res.passes)} pass(es). "
          f"final human-signature: {res.final_score:.3f}")
    if res.residual:
        print("\n  residual (reported, not hidden):")
        for r in res.residual:
            print(f"    - {r}")
    # A low score in offline mode is expected information, not a failure: the
    # user asked to strip and scrub, and that succeeded. Failure is metadata we
    # could not remove, or a --regenerate run that never cleared the bar.
    meta_left = any(r.startswith("metadata still present") for r in res.residual)
    score_fail = res.regenerate and res.final_score < args.threshold
    return 0 if (res.converged and not meta_left and not score_fail) else 1


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

    px = sub.add_parser(
        "fix", help="run the whole workflow in a bounded loop (offline unless --regenerate)")
    px.add_argument("path", help="file to clean, in place")
    px.add_argument("--rename", action="store_true",
                    help="also rename the file if it names its source")
    px.add_argument("--regenerate", action="store_true",
                    help="rewrite the prose to lift the score (needs Ollama; fact-gated)")
    px.add_argument("--model", default="gemma3:4b",
                    help="local open-weight model for regeneration (allowlisted)")
    px.add_argument("--allow-model", action="store_true",
                    help="accept a model not on the verified-unwatermarked allowlist")
    px.add_argument("--threshold", type=float, default=0.70,
                    help="human-signature target for --regenerate")
    px.add_argument("--max-passes", type=int, default=3,
                    help="hard cap on loop passes (default 3)")
    px.set_defaults(func=cmd_fix)

    pr = sub.add_parser("run", help="full pipeline (needs model backends)")
    pr.add_argument("input", nargs="?", default="-")
    pr.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
