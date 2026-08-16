"""Deterministic constraint extraction, the facts a rewrite must not lose.

Shared by the extractor (which attaches constraints to Meaning so the
regenerator is *told* what to keep) and the checker (which verifies they
survived). One implementation, both directions, populate and verify can never
drift apart.

Zero dependencies, all regex and set logic. That is deliberate: unlike style or
meaning, facts are the one thing a rule can actually pin down, and a check that
cannot fail silently is worth more than a cleverer one that can.

Grounding, the rules here follow published prior art rather than intuition:

  * Negation detection follows NegEx (Chapman, Bridewell, Hanbury, Cooper &
    Buchanan, "A Simple Algorithm for Identifying Negated Findings and Diseases
    in Discharge Summaries", J. Biomedical Informatics 34(5), 2001), including
    its pseudo-negation-takes-precedence rule and the general-language subset of
    its published trigger lexicon (github.com/chapmanbe/negex).
  * The reason this check has to exist at all, that embedding similarity is
    blind to negation and antonymy, is a documented property of distributional
    representations, not a quirk of our chosen model. See "This is not correct!
    Negation-aware Evaluation of Language Generation Systems" (arXiv:2307.13989)
    and "Learning Robust Negation Text Representations" (arXiv:2507.12782).
"""

from __future__ import annotations

import re

from ..core.types import Constraints

#: A colon ends a clause for our purposes. LLM prose is full of
#: "Development Tools: Console development requires…", where the word after the
#: colon is capitalised because it opens a clause, not because it is a name.
#: Requiring trailing whitespace keeps "3:30" and "9:1" intact.
_SENT_SPLIT = re.compile(r"(?<=[.!?:])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_NUMERAL = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
# Proper-noun runs: capitalised words, optionally joined by internal lowercase
# connectors ("Bank of England"). Also catches all-caps acronyms and CamelCase.
_PROPER = re.compile(r"\b(?:[A-Z][a-z’']+|[A-Z]{2,}|[A-Z][a-z]+[A-Z]\w*)\b")

# ── Negation detection ──────────────────────────────────────────────────────
# Modelled on NegEx (Chapman et al. 2001), the standard rule-based negation
# detector, rather than invented from scratch. NegEx splits its lexicon into
# pre-negation triggers (PREN), post-negation triggers (POST) and *pseudo*-
# negation triggers (PSEU), phrases that contain a negation word but do not
# negate anything, and applies the rule that PSEU takes PRECEDENCE over PREN.
# That precedence rule is the important part and is what we implement.
#   Trigger lexicon: github.com/chapmanbe/negex (negex_triggers.txt)

#: Single-token negators. `n't` is handled separately: the apostrophe form varies
#: (' vs ’) and it attaches to the verb rather than standing alone.
_NEGATORS = frozenset({
    "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither",
    "nor", "cannot", "cant", "wont", "didnt", "doesnt", "dont", "isnt", "arent",
    "wasnt", "werent", "hasnt", "havent", "hadnt", "shouldnt", "wouldnt",
    "couldnt", "without", "lacks", "lacking", "fails", "failed", "unable",
    "rarely", "hardly", "barely", "seldom",
    # From NegEx PREN, general-language subset:
    "denies", "denied", "absent",
})

#: Multi-word negation triggers (NegEx PREN, general-language subset).
_NEGATOR_PHRASES: tuple[tuple[str, ...], ...] = (
    ("absence", "of"), ("free", "of"), ("negative", "for"),
    ("no", "evidence"), ("fails", "to", "reveal"), ("never", "had"),
)

#: Pseudo-negation: contains a negator but asserts nothing negative. NegEx's PSEU
#: list is clinical, so we take the subset whose logic transfers to general prose
#: and deliberately DROP the rest. "no increase"/"no change" are PSEU in NegEx
#: because they do not negate the clinical *concept*; for claim-polarity
#: comparison they are exactly the inversions we want to catch, so they stay
#: negations here. The scalar-additive family generalises from NegEx's "not only"
#:, same construction, and it ADDS to a claim rather than denying it.
#:
#: This was the single biggest false-positive source measured on real model
#: output: the 1B rewrote "the order takes into account letter names" into "the
#: order is not just about letter names; it's also influenced by …", which
#: affirms the source and was still flagged as an inversion.
_PSEUDO_NEGATION: tuple[tuple[str, ...], ...] = (
    # NegEx PSEU, verbatim:
    ("not", "only"), ("not", "necessarily"),
    ("not", "certain", "if"), ("not", "certain", "whether"),
    ("without", "difficulty"),
    # Same scalar-additive construction as "not only":
    ("not", "just"), ("not", "merely"), ("not", "simply"),
    ("not", "solely"), ("not", "alone"), ("no", "longer", "just"),
)

#: Opposing pairs that flip a claim while keeping every content word and the
#: topic intact, exactly the failure embeddings cannot see. Each entry is a
#: pair of mutually exclusive polarity groups.
#:
#: Antonymy is not an afterthought to negation, it is the same blind spot: the
#: distributional hypothesis that text embeddings are built on learns words from
#: the contexts they appear in, and antonyms share their contexts almost
#: perfectly, so the resulting models are "insensitive to negation and related
#: phenomena such as antonymy" (Learning Robust Negation Text Representations,
#: arXiv:2507.12782). Hence both halves of the polarity check.
#:
#: This is a curated list, not a lexicon, WordNet antonym pairs would be the
#: principled source, at the cost of the zero-dependency guarantee. Covers the
#: common quantitative and outcome flips; see the honest-limits note in
#: constraint_checker.ConstraintChecker.
_ANTONYMS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"increased", "increase", "rose", "rise", "grew", "growth", "up",
                "gained", "gain", "higher", "more", "surged", "climbed"}),
     frozenset({"decreased", "decrease", "fell", "fall", "shrank", "decline",
                "declined", "down", "lost", "loss", "lower", "less", "dropped", "sank"})),
    (frozenset({"succeeded", "success", "successful", "passed", "worked", "won"}),
     frozenset({"failed", "failure", "unsuccessful", "broke", "crashed", "lost"})),
    (frozenset({"always", "all", "every", "everyone", "everything"}),
     frozenset({"never", "none", "no", "nobody", "nothing"})),
    (frozenset({"before", "earlier", "prior", "preceded"}),
     frozenset({"after", "later", "subsequent", "followed"})),
    (frozenset({"above", "over", "exceeds", "exceeded", "greater"}),
     frozenset({"below", "under", "beneath", "fewer", "smaller"})),
    (frozenset({"true", "correct", "right", "valid", "accurate"}),
     frozenset({"false", "incorrect", "wrong", "invalid", "inaccurate"})),
    (frozenset({"allowed", "permitted", "enabled", "supported", "included"}),
     frozenset({"forbidden", "prohibited", "disabled", "unsupported", "excluded"})),
    (frozenset({"open", "opened", "started", "began", "launched"}),
     frozenset({"closed", "shut", "stopped", "ended", "halted"})),
)

#: Number words, so a reword that spells "three" still matches a source "3".
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000", "million": "1000000",
    "billion": "1000000000",
}

#: Words too common to be worth aligning on, and capitalised words that are
#: usually sentence-openers rather than names.
_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "as", "by", "from", "that", "this", "these", "those", "it",
    "its", "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "will", "would", "can", "could", "should",
    "may", "might", "must", "so", "than", "then", "there", "their", "they",
    "them", "we", "our", "us", "you", "your", "i", "he", "she", "his", "her",
    "what", "which", "who", "when", "where", "how", "why", "all", "any",
    "some", "such", "just", "also", "about", "into", "over", "after", "more",
    "most", "other", "only", "up", "out", "very", "s", "t",
})
_NON_ENTITY_CAPS = frozenset({
    "the", "this", "that", "these", "those", "it", "he", "she", "they", "we",
    "you", "i", "a", "an", "and", "but", "if", "so", "in", "on", "at", "for",
    "there", "here", "what", "when", "where", "why", "how", "who", "which",
    "his", "her", "their", "our", "its", "my", "your", "one", "two", "three",
    "first", "second", "third", "next", "then", "now", "yes", "no", "not",
    "some", "many", "most", "every", "each", "both", "all", "after", "before",
    "while", "since", "because", "although", "however", "instead", "still",
    "also", "by", "from", "with", "without", "as", "to", "of", "or",
})


def _norm_number(raw: str) -> str:
    """1,000 -> 1000 ; 40.0 -> 40 ; so formatting changes don't read as loss."""
    s = raw.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


#: Markdown emphasis/heading/list markers. LLM output is full of these, and left
#: in place they wreck two things: `**Hardware:**` stops looking sentence-initial
#: (so "Hardware" gets counted as a proper noun), and heading fragments become
#: pseudo-claims that hijack alignment. Strip before anything else looks at text.
_MD_INLINE = re.compile(r"[*_`~]{1,3}")
_MD_LEAD = re.compile(r"^\s*(?:[-*+•>]\s+|#{1,6}\s+|\d+[.)]\s+)")


def _demarkup(line: str) -> str:
    """Inline markers first, THEN leading ones, `**1. Habitat loss:**` only
    reveals its list ordinal once the bold markers are gone. Getting this order
    wrong makes enumeration numbering look like facts, and a rewrite that drops
    the numbering then reads as fact loss.

    Note `_MD_LEAD` requires the digit to be followed by `.` or `)` AND a space,
    so a real sentence opening "3 people died" keeps its number.
    """
    return _MD_LEAD.sub("", _MD_INLINE.sub("", line)).strip()


def normalised(text: str) -> str:
    """Text with markup removed, the single surface every extractor reads, so
    numerals, entities and claims can never disagree about what the text says."""
    return "\n".join(s for s in (_demarkup(l) for l in text.split("\n")) if s)


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for block in text.split("\n"):
        block = _demarkup(block)
        if not block:
            continue
        for s in _SENT_SPLIT.split(block):
            s = s.strip(" \t-•*:")
            if s:
                out.append(s)
    return out


def numerals(text: str) -> set[str]:
    """Numeric values, digits and spelled-out words alike, normalised. Reads the
    de-marked-up surface so list ordinals are not mistaken for facts."""
    clean = normalised(text)
    found = {_norm_number(m) for m in _NUMERAL.findall(clean)}
    for w in _WORD.findall(clean.lower()):
        if w in _NUM_WORDS:
            found.add(_NUM_WORDS[w])
    return found


def _is_heading(sent: str) -> bool:
    """Title-case fragment rather than a sentence. LLM output is full of these
    ("Game Engine", "Key Considerations"), and every word after the first looks
    like a proper noun to a capitalisation rule, which is how common nouns end
    up in the must-keep set and get a faithful rewrite rejected."""
    words = _WORD.findall(sent)
    if not words or len(words) > 8 or sent.rstrip().endswith((".", "!", "?")):
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / len(words) >= 0.5


def entities(text: str) -> set[str]:
    """Proper nouns, lowercased. A capital only counts as a name if it appears
    capitalised somewhere it *had* to be a choice, i.e. mid-sentence, in a real
    sentence. Sentence openers and title-case headings need corroboration. That
    is the cheap way to tell "Microsoft" from "Hardware" with no POS tagger."""
    mid: set[str] = set()
    unproven: set[str] = set()
    for sent in sentences(text):
        words = _PROPER.findall(sent)
        if not words:
            continue
        heading = _is_heading(sent)
        first_tok = _WORD.match(sent.strip())
        head = first_tok.group(0) if first_tok else ""
        for m in _PROPER.finditer(sent):
            w = m.group(0)
            low = w.lower()
            if low in _NON_ENTITY_CAPS:
                continue
            # A capital right after an opening quote is explained by the quote,
            # not by a name: `a call to action, such as "Visit our website"`.
            prev = sent[:m.start()].rstrip()
            quoted_open = prev.endswith(('"', "“", "'", "‘", "(", "["))
            if heading or quoted_open or (w == head and sent.strip().startswith(w)):
                unproven.add(low)
            else:
                mid.add(low)
    # Words seen ONLY in an opener or a heading are discarded: their capital is
    # explained by position, so it is no evidence of a name. `unproven` is kept
    # separate rather than folded in, because that asymmetry IS the rule.
    return mid


def _matches_at(toks: list[str], i: int, phrase: tuple[str, ...]) -> bool:
    return toks[i:i + len(phrase)] == list(phrase)


def _is_negated(words: list[str]) -> bool:
    """NegEx-style: mask pseudo-negation spans FIRST, then look for triggers in
    what survives. The precedence ordering is the whole point, 'not only' must
    consume its 'not' before the negation pass ever sees it."""
    toks = [w.replace("’", "'").replace("'", "").lower() for w in words]
    masked = [False] * len(toks)

    for i in range(len(toks)):
        for phrase in _PSEUDO_NEGATION:
            if _matches_at(toks, i, phrase):
                for j in range(i, min(i + len(phrase), len(toks))):
                    masked[j] = True

    for i, tok in enumerate(toks):
        if masked[i]:
            continue
        if words[i].replace("’", "'").endswith("n't") or tok in _NEGATORS:
            return True
        if any(_matches_at(toks, i, p) for p in _NEGATOR_PHRASES):
            return True
    return False


def claim_key(sentence: str) -> frozenset[str]:
    """Content words of a claim, what we align two sentences on. Negators and
    antonym members are excluded so a flipped claim still aligns with its
    source (otherwise the inversion would look like an unrelated sentence and
    escape as 'dropped' rather than 'inverted')."""
    polarity_words = set()
    for a, b in _ANTONYMS:
        polarity_words |= a | b
    out = set()
    for w in _WORD.findall(sentence.lower()):
        bare = w.replace("’", "'").replace("'", "")
        if bare in _STOP or bare in _NEGATORS or bare in polarity_words:
            continue
        if bare.endswith("nt") and bare[:-2] in {"is", "are", "was", "were", "do", "does", "did"}:
            continue
        out.add(bare)
    return frozenset(out)


def polarity_signature(sentence: str) -> tuple[bool, tuple[int, ...]]:
    """(negated?, which side of each antonym pair the sentence sits on).

    Side is 0 = neither, 1 = first group, 2 = second group, 3 = both (ambiguous,
    never counted as a flip).
    """
    words = _WORD.findall(sentence.lower())
    bare = {w.replace("’", "'").replace("'", "") for w in words}
    sides: list[int] = []
    for a, b in _ANTONYMS:
        hit_a, hit_b = bool(bare & a), bool(bare & b)
        sides.append(3 if (hit_a and hit_b) else 1 if hit_a else 2 if hit_b else 0)
    return _is_negated(words), tuple(sides)


def polarity_flipped(src: str, cand: str) -> bool:
    """True when two aligned claims assert opposite things."""
    s_neg, s_sides = polarity_signature(src)
    c_neg, c_sides = polarity_signature(cand)
    if s_neg != c_neg:
        return True
    for s, c in zip(s_sides, c_sides):
        if s in (1, 2) and c in (1, 2) and s != c:
            return True
    return False


#: Claims below this many content words carry too little to align reliably.
MIN_CLAIM_WORDS = 3


def extract_constraints(text: str) -> Constraints:
    """The full must-keep fingerprint of a text."""
    claims: list[tuple[frozenset[str], bool]] = []
    for sent in sentences(text):
        key = claim_key(sent)
        if len(key) >= MIN_CLAIM_WORDS:
            claims.append((key, _is_negated(_WORD.findall(sent.lower()))))
    return Constraints(
        numerals=frozenset(numerals(text)),
        entities=frozenset(entities(text)),
        claims=tuple(claims),
    )
