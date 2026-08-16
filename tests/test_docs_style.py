"""Run our own AI-tell detector on our own documentation.

`score/features.py` weights `em_dash_rate` as an AI signal, and the regenerator's
system prompt bans AI tells outright. Shipping documentation full of those same
tells is not a cosmetic problem: it is the most visible possible evidence that
the tool's own standards were not applied to its own output. A reader who spots
it has every reason to doubt the numbers too.

So the rule is enforced rather than intended. These characters are the ones that
survive copy-paste out of a chat window and mark text as unedited machine output:

    em dash        the single most cited surface tell of AI prose
    en dash        same family, same giveaway in running text
    unicode arrows people type two characters; renderers do not make one
    ellipsis char  typed as three periods by humans
    curly quotes   smart-quote substitution from a rich-text source

Prose has to be rewritten to not need them, which is the actual work. Swapping
an em dash for a hyphen keeps the same telltale rhythm and fixes nothing.

The table below uses escape sequences, not the literal characters, and that is
deliberate. A bulk find-and-replace over the repo rewrote this very file on the
first attempt, turning the banned arrow into the ASCII arrow it recommends, so
the test then banned its own advice. Data that describes forbidden characters
must not itself be written in them.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED = {
    "—": "em dash, use a comma, colon, or a new sentence",
    "–": "en dash, use a hyphen or the word 'to'",
    "→": "rightwards arrow, use ->",
    "←": "leftwards arrow, use <-",
    "…": "ellipsis character, use three periods",
    "’": "curly apostrophe, use a straight one",
    "“": "curly open quote, use a straight one",
    "”": "curly close quote, use a straight one",
}

#: Nothing is exempt by name. The one file that was (tests/harness/README.md)
#: turned out to be ordinary prose, exempted on a wrong assumption that it held
#: character data. An allow-list is where a standard quietly goes to die.
EXEMPT: set[str] = set()


def _is_fixture(rel: str) -> bool:
    """Evidence, not prose.

    `tests/*.md` are real files downloaded from ChatGPT, Gemini, and Grok, kept
    byte for byte. Two of the three use em dashes and one uses a curly
    apostrophe, which is the tell showing up in the wild on live 2026 output.
    Editing them to satisfy a style rule would destroy the thing they exist to
    demonstrate. Scoped to the top level of tests/ so it cannot creep.
    """
    return rel.startswith("tests/") and "/" not in rel[len("tests/"):]


def _docs() -> list[Path]:
    return [
        p for p in ROOT.rglob("*.md")
        if ".git" not in p.parts
        and "node_modules" not in p.parts
        and str(p.relative_to(ROOT)) not in EXEMPT
        and not _is_fixture(str(p.relative_to(ROOT)))
    ]


def test_docs_carry_no_ai_surface_tells():
    failures: list[str] = []
    for path in _docs():
        text = path.read_text(encoding="utf-8")
        for ch, advice in BANNED.items():
            if ch not in text:
                continue
            line_no = next(
                (i for i, ln in enumerate(text.splitlines(), 1) if ch in ln), 0
            )
            count = text.count(ch)
            rel = path.relative_to(ROOT)
            failures.append(f"{rel}:{line_no} has {count}x {advice}")

    assert not failures, (
        "Documentation contains the AI surface tells this project detects:\n  "
        + "\n  ".join(failures)
    )


def test_readme_scores_well_on_our_own_scorer():
    """Dogfooding: the front page should pass the bar we set for other text."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from reweave.core.types import Document
    from reweave.score import StatisticalScorer

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sig = StatisticalScorer().score(Document(text=readme))
    em = sig.features.em_dash_rate or 0.0

    assert em == 0.0, f"README em-dash rate is {em:.4f}, should be exactly 0"
    assert sig.score >= 0.70, (
        f"README scores {sig.score:.3f} on our own human-signature heuristic; "
        "if we cannot clear our own bar, the bar is not credible"
    )
