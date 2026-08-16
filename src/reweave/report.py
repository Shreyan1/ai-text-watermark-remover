"""Turning findings into something a person actually wants to read.

The first version of the `meta` output printed the raw shape of the data: the
xattr name, the truncated value, a repeated parenthetical about where it lives.
It was correct and nearly unreadable. Someone running this for the first time
wants three answers, in this order:

    what did you find?   why does it matter?   what do I do now?

The attribute name (`com.apple.metadata:kMDItemWhereFroms`) answers none of them,
so it stops being the headline and becomes a footnote. "SOURCE URL" is the
headline, one plain sentence says why it matters, and the run ends with the exact
command that fixes it.

Two things are decoded rather than dumped, because dumping them threw away the
most interesting part of the finding:

  * `com.apple.quarantine` is `flags;hex-time;app;uuid`. Rendered, that is the
    app that downloaded the file and the minute it happened.
  * an opaque base64 blob in a URL often is not opaque. Gemini's `c=` parameter
    decodes to `bard_storage / response_data / <id>`, a unique per-response
    identifier. Printing 128 characters of base64 hid the fact that the file is
    pinned to one specific generation.

Colour is used only when stdout is a terminal, and never when NO_COLOR is set
(https://no-color.org).
"""

from __future__ import annotations

import base64
import datetime
import os
import re
import sys

_WIDTH = 78


def _supports_colour() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class _Style:
    """ANSI codes, or empty strings when the output is not a terminal."""

    def __init__(self, enabled: bool) -> None:
        self.bold = "\033[1m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.off = "\033[0m" if enabled else ""

    def b(self, s: str) -> str:
        return f"{self.bold}{s}{self.off}"

    def d(self, s: str) -> str:
        return f"{self.dim}{s}{self.off}"


S = _Style(_supports_colour())


#: Headline + one plain sentence, per trace. Keyed by xattr name where the layer
#: alone is not specific enough (a "SOURCE URL" and a "DOWNLOAD STAMP" are both
#: xattrs but mean very different things to the reader).
_MEANING = {
    "com.apple.metadata:kMDItemWhereFroms": (
        "SOURCE URL",
        "Where the file was downloaded from. Rewriting the text never touches it.",
    ),
    "user.xdg.origin.url": (
        "SOURCE URL",
        "Where the file was downloaded from. Rewriting the text never touches it.",
    ),
    "user.xdg.referrer.url": (
        "REFERRER URL", "The page you came from when you downloaded it."),
    "com.apple.quarantine": (
        "DOWNLOAD STAMP", "Which app downloaded it, and when."),
    "com.apple.metadata:kMDItemDownloadedDate": (
        "DOWNLOAD DATE", "The moment the file arrived on this machine."),
    "com.apple.metadata:kMDItemCreator": (
        "CREATOR TAG", "The application that claims to have made the file."),
    "com.apple.metadata:kMDItemAuthors": (
        "AUTHOR TAG", "An author recorded alongside the file."),
}

_BY_LAYER = {
    "filename": ("FILENAME", "The one trace anyone you share the file with will see."),
    "frontmatter": ("FRONT MATTER", "A metadata key in the document header naming its source."),
    "inline": ("INLINE ATTRIBUTION", "A line in the prose crediting the tool that wrote it."),
    "docx": ("DOCUMENT PROPERTIES", "Author and application fields stored inside the .docx."),
    "pdf": ("PDF METADATA", "Producer and creator fields in the PDF info dictionary."),
    "xattr": ("EXTENDED ATTRIBUTE", "Filesystem metadata carried alongside the bytes."),
}

_WHERE = {
    "xattr": "extended attribute",
    "filename": "the name itself",
    "frontmatter": "document header",
    "inline": "in the prose",
    "docx": "inside the container",
    "pdf": "info dictionary",
}


def describe(finding) -> tuple[str, str]:
    """Headline and plain-language reason for one finding."""
    if finding.key in _MEANING:
        return _MEANING[finding.key]
    return _BY_LAYER.get(finding.layer, (finding.key.upper(), ""))


def where(finding) -> str:
    """Short right-hand tag saying which layer this lives in."""
    if finding.line is not None:
        return f"line {finding.line}"
    return _WHERE.get(finding.layer, finding.layer)


def decode_quarantine(value: str) -> str | None:
    """`0081;6a8195ec;Arc;UUID` -> `Arc, downloaded 16 Aug 2026 16:20`."""
    parts = value.split(";")
    if len(parts) < 3:
        return None
    stamp, app = parts[1], parts[2] or "an unknown app"
    try:
        when = datetime.datetime.fromtimestamp(int(stamp, 16))
    except (ValueError, OSError, OverflowError):
        return app
    return f"{app}, downloaded {when.strftime('%d %b %Y %H:%M')}"


def decode_opaque(value: str) -> list[str] | None:
    """Pull readable strings out of a base64 blob, or None if it stays opaque.

    Vendor "opaque" tokens frequently are not. Worth trying before printing a
    wall of base64 that tells the reader nothing.
    """
    if len(value) < 16 or not re.fullmatch(r"[A-Za-z0-9_\-+/=]+", value):
        return None
    for pad in range(4):
        try:
            blob = base64.urlsafe_b64decode(value + "=" * pad)
        except Exception:  # noqa: BLE001 - not base64, that is a normal answer
            continue
        found = [m.decode() for m in re.findall(rb"[\x20-\x7e]{4,}", blob)]
        if found:
            return found
        break
    return None


def wrap_value(value: str, indent: str, width: int = _WIDTH) -> list[str]:
    """Wrap a long value, filling each line.

    `textwrap` alone is not enough: a URL is one enormous "word" with no spaces,
    so it has to be hard-broken. Ordinary prose still wraps on word boundaries,
    which the first version of this got wrong by breaking on every space and
    printing one word per line.
    """
    room = max(24, width - len(indent))
    out: list[str] = []
    line = ""
    for word in value.split(" "):
        while len(word) > room:  # unbreakable run, e.g. a signed URL
            if line:
                out.append(indent + line)
                line = ""
            out.append(indent + word[:room])
            word = word[room:]
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= room:
            line += " " + word
        else:
            out.append(indent + line)
            line = word
    if line:
        out.append(indent + line)
    return out or [indent + value]


#: Why one component of a URL matters. Keys are the label prefix used by
#: `metadata_scrubber._url_provenance_parts`.
_PART_MEANING = {
    "response-id": "pins the exact generation that produced this file",
    "file-id": "a unique id for this exact file or response",
    "credential": "an access key tied to the vendor account",
    "signature": "signs the request, proving it came from that session",
    "expiry": "when the signed link was minted",
    "filename": "the name the vendor gave it",
    "token": "an opaque identifier that can be traced back",
}


def part_meaning(label: str) -> str:
    return _PART_MEANING.get(label.split(" ", 1)[0], "")


def render_finding(finding, index: int, *, removed: bool = False,
                   suggested: str | None = None) -> list[str]:
    """One finding as a block of lines."""
    head, why = describe(finding)
    tag = where(finding)
    mark = f"{S.green}[ok]{S.off}" if removed else f"[{index}]"

    # Headline row, with the layer tag pushed to the right margin.
    left = f"  {mark}  {S.b(head)}"
    plain_len = len(re.sub(r"\033\[[0-9;]*m", "", left))
    pad = max(2, _WIDTH - plain_len - len(tag))
    lines = [left + " " * pad + S.d(tag)]
    if why:
        lines.append(S.d(f"       {why}"))
    lines.append("")

    # The value itself.
    if finding.key == "com.apple.quarantine":
        pretty = decode_quarantine(finding.value)
        lines += wrap_value(pretty or finding.value, "       ")
    else:
        lines += wrap_value(finding.value, "       ")

    # What is inside it, when there is anything worth naming.
    if finding.highlights:
        lines.append("")
        lines.append(S.d("       what it contains"))
        width = max(len(lbl.split(" ", 1)[0]) for lbl, _ in finding.highlights)
        for label, snippet in finding.highlights:
            short = label.split(" ", 1)[0]
            lines.append(f"         {short:<{width}}  {snippet}")
            meaning = part_meaning(label)
            if meaning:
                lines.append(S.d(f"         {'':<{width}}  {meaning}"))

    if suggested:
        lines.append("")
        lines.append(f"       suggested  {S.b(suggested)}")

    # The technical name last: useful, but it is a footnote, not the headline.
    if finding.layer in ("xattr", "frontmatter", "docx", "pdf"):
        lines.append(S.d(f"       ({finding.key})"))
    lines.append("")
    return lines
