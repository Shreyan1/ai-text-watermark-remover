"""The deterministic core must run with the network physically unplugged.

The claim on the tin is that `meta`, `scrub`, `score`, `facts`, and the default
`fix` loop need no model and no connection. That is easy to believe and easy to
break: one stray import of an Ollama-backed default, and a command that used to
work on a plane starts hanging on a socket. So the guarantee is enforced, not
documented and hoped for.

The enforcement is blunt on purpose: every way Python opens a network socket is
replaced with something that raises. Anything reaching for the network fails the
test loudly, here, rather than silently in a user's offline session. Local
subprocesses (macOS routes xattr reads through /usr/bin/xattr) are not sockets
and stay allowed, which is correct: "offline" means no network, not no processes.
"""

from __future__ import annotations

import socket
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted on the offline path")

    # Cover the constructors urllib and everything else actually use.
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    yield


def _dirty_file() -> str:
    d = Path(tempfile.mkdtemp())
    f = d / "notes.md"
    f.write_text(
        "---\ntitle: Real Title\ngenerator: ChatGPT\n---\n\n"
        "The revenue rose 40% in Q3—a big jump. It’s the best quarter yet.\n",
        encoding="utf-8",
    )
    return str(f)


def test_meta_inspect_is_offline(no_network):
    from reweave.scrub.metadata_scrubber import FileMetadataScrubber

    report = FileMetadataScrubber().inspect(_dirty_file())
    assert any(f.layer == "frontmatter" for f in report.findings)


def test_scrub_is_offline(no_network):
    from reweave.core.types import Document
    from reweave.scrub.unicode_scrubber import UnicodeScrubber

    out = UnicodeScrubber().scrub(Document(text="a—b, it’s fine"))
    assert "—" not in out.text and "’" not in out.text


def test_score_is_offline(no_network):
    from reweave.core.types import Document
    from reweave.score import StatisticalScorer

    sig = StatisticalScorer().score(Document(text=_dirty_file_text()))
    assert 0.0 <= sig.score <= 1.0


def test_facts_constraint_is_offline(no_network):
    from reweave.core.types import Document
    from reweave.verify import ConstraintChecker

    r = ConstraintChecker().check(
        Document(text="The tower is 92 metres tall and made of marble."),
        Document(text="Built of marble, the tower stands 92 metres high."),
    )
    assert r.ok


def test_fix_default_loop_is_offline(no_network):
    from reweave.fix import run_fix

    res = run_fix(_dirty_file())  # regenerate defaults False
    assert res.converged
    assert res.regenerate is False


def _dirty_file_text() -> str:
    return (
        "The revenue rose 40% in Q3. It jumped hard. Nobody expected that, and "
        "yet here we are, staring at the biggest quarter the team has ever run."
    )
