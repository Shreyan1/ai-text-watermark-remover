# Design brief: the web demo

A static site for GitHub Pages that lets someone try this tool without installing
it, and that is honest about the one thing a browser cannot do.

Hand this document to a designer. It covers what the site is, the hard technical
constraints, what each panel receives and produces, and the tone to aim for.

---

## What it is

One page, five tools, no backend.

Someone arrives holding a file they got out of a chat window, or a block of text
they pasted. They want to know what it gives away and what to do about it. The
site answers that in a few seconds, then points at the command line for the part
it cannot reach.

## Two rules that cannot be bent

**1. Nothing leaves the browser.** No upload, no API call, no analytics on
content. All analysis runs in client-side JavaScript on the visitor's machine.
This is not a nice-to-have: the tool exists for people handling text they do not
want to hand to a third party, and a demo that quietly posts their document to a
server would contradict the entire project. The page should say so plainly, near
the drop zone, and it should be true.

**2. The browser cannot see extended attributes.** This is the important one.

Files downloaded from a chat UI carry their source URL in filesystem metadata
(`kMDItemWhereFroms` on macOS). It is invisible in any editor and it survives
rewriting every word of the text. It is the most striking thing this tool finds.

A browser's File API exposes the name, size, type, and bytes of a file. It does
not expose extended attributes, and there is no permission or flag that changes
this. Drag the very same file into this page and the source URL is simply not
there to read.

So the site must not imply it checked and found nothing. That would be a false
clean bill of health, which is the failure this project cares most about
avoiding. Design a dedicated panel that explains the gap and shows what the CLI
finds instead, using the real examples below. Treat it as the strongest reason to
install, not as an apology.

## The five panels

Each maps to one command. Numbers 2 to 5 run fully in the browser.

### 1. Provenance (`reweave meta`) - partly possible

| layer | in the browser? | why |
|---|---|---|
| source URL, download stamp | **no** | extended attributes are not exposed to web pages |
| filename | **yes** | `File.name` is available, and it often names the vendor outright |
| front matter | **yes** | it is in the file content |
| inline attribution | **yes** | it is in the file content |

So this panel does three of four layers on a real file, and for the fourth it
shows a worked example rather than a blank. Suggested treatment: a short list of
what was found, then a clearly separated block titled something like "what this
page cannot see", holding the real fixture output. Two examples worth showing,
both genuine, both from files in `tests/`:

*Gemini.* The where-from URL carries a parameter that decodes to a unique
per-response identifier, `0e28349d995dad45a00065927d1449194037055f63804c310`.
Not merely "this came from Gemini" but which generation produced it.

*Kimi.* The where-from URL is a pre-signed object-storage link holding an access
credential (`X-Tos-Credential`), a file id, an expiry, and a request signature.

Both survive a complete rewrite of the prose. Neither is visible in an editor.

The filename check should feel immediate: drop in `Gemini-Whatever.md` and the
page names the vendor and proposes a clean filename straight away.

### 2. Character hygiene (`reweave scrub`) - fully possible

Input: pasted text or a file. Output: cleaned text, plus a count of what was
removed by category.

Detects and removes zero-width characters (ZWSP, ZWNJ, ZWJ, word joiner, BOM,
soft hyphen), bidirectional controls, any other Unicode format character,
homoglyphs (Cyrillic and Greek look-alikes folded to ASCII), and typographic
punctuation (curly quotes, en and em dashes, ellipsis character, exotic spaces).

This one is visually rewarding and should probably lead, because invisible
characters are invisible until something shows them. A before-and-after view with
the removed characters marked in place makes the point instantly. A copy button
on the cleaned output is essential.

### 3. Human signature (`reweave score`) - fully possible

Input: text. Output: a score from 0 to 1, a verdict, and a per-feature
contribution breakdown.

Features: burstiness (variation in sentence length, by far the strongest signal,
AUROC 0.906 in our benchmark), type-token ratio, em-dash rate, rule-of-three
rate, paragraph variation, entity density, numeral density.

Two things the design must handle honestly. The score abstains when the text is
too short to measure burstiness, and an abstention has to look different from a
confident answer rather than showing a number that means little. And the label is
"how uniform does this read", not "is this AI". The tool measures a statistical
property, it does not detect authorship, and the copy should not overclaim.

### 4. Fact preservation (`reweave facts`) - fully possible

Input: two texts, before and after. Output: whether the facts survived, coverage
percentages for numbers and names, and any inverted claims side by side.

The deterministic rules port cleanly to JavaScript: numeral extraction, entity
extraction, negation detection, and a curated antonym list. The stronger
entailment check needs a local model and belongs on the CLI, so this panel should
name that boundary the way the tool itself does.

A two-pane diff layout suits this. Inverted claims are the alarming result and
should be unmissable.

### 5. Full run (`reweave fix`) - partly possible

Chains the above: strip what is strippable, scrub, score, report. Good as a
summary view. It cannot rewrite prose, which needs a local model, so the panel
should end by showing the command that does.

## Design direction

Minimal, closer to a measuring instrument than a marketing page. The project's
credibility rests on not overclaiming, and the site should feel the same.

Some specifics worth carrying over:

- **Avoid em dashes in the copy.** The tool scores em-dash rate as a signal of
  unedited machine text. Using them on the site would be self-refuting. The
  repository enforces this on its own documentation with a test.
- **Show severity, do not shout it.** A homepage URL and an embedded access
  credential are not the same finding and should not look the same.
- **Every result says what to do next**, ideally as a command that can be copied.
- **Empty states matter.** "No traces found" and "this page cannot check that"
  are different statements and must never look alike.
- Monospace for values, findings, and commands. Proportional for explanation.
- Dark and light both, since the audience lives in terminals.

## Technical constraints

- Static hosting on GitHub Pages. No backend, no build step that needs a server.
- Client-side JavaScript only. No sending file content anywhere.
- Works offline once loaded. No CDN dependency at runtime.
- Accessible: keyboard reachable, real contrast, no colour-only meaning.
- Responsive, because people will open this on a phone to see what it does.

## Source of truth

The algorithms live in `src/reweave/`, and the browser versions must match them
rather than approximate them:

| panel | module |
|---|---|
| provenance | `scrub/metadata_scrubber.py` |
| character hygiene | `scrub/unicode_scrubber.py` |
| human signature | `score/features.py`, `score/human_signature.py` |
| fact preservation | `verify/constraints.py`, `verify/constraint_checker.py` |

Benchmarks and measured numbers are in `RESULTS.md`, and anything quoted on the
site should come from there rather than being invented.
