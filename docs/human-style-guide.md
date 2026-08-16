# Writing guide: prose that does not read as machine-generated

Paste this file into the model before asking for an essay, article, or report.
It works as a system prompt, as a preamble to your actual request, or as a
reference to hand back when a draft comes out flat.

Every rule here comes from comparing real human writing against real model
output and measuring what actually differs. The rules are ordered by how much
they matter. If the model can only follow one, it should be rule 1.

---

## Rule 1: Vary sentence length, hard

This matters more than everything else in this file combined. Machine prose
settles into a steady rhythm, most sentences landing within a word or two of each
other. Human prose lurches. A four-word sentence, then a thirty-word one that
keeps going past the point where a model would have stopped.

**Do not write:**

> The library opened in 1936. It served the neighbourhood for decades. Its
> reading room became a local landmark.

Three sentences, six to eight words each. Nothing is wrong with any of them, and
the evenness is the problem.

**Write:**

> The library opened in 1936. For decades it was the only place in the
> neighbourhood where you could sit down for three hours without buying
> anything, which is most of why the reading room ended up mattering to people.

Four words, then thirty-one.

**Concrete instruction:** in every paragraph, include at least one sentence under
eight words and at least one over twenty-five. Never let three consecutive
sentences land within five words of each other.

---

## Rule 2: Do not pile on facts to sound credible

This one is counterintuitive and it is the mistake most "write like a human"
attempts make. Model output is already *more* fact-dense than human writing, not
less. Adding more names, dates, and figures pushes prose further from how people
actually write, not closer.

**Do not write:**

> Designed by architect William Emerson and constructed between 1906 and 1921
> from Makrana marble, the 184-foot structure spans 57 acres and houses 28,394
> artefacts across 25 galleries.

Every clause carries a data point. Real writers do not sustain that.

**Write:**

> Emerson designed it, and it took fifteen years to build. The marble came from
> the same quarries as the Taj. Standing in front of it you mostly notice that
> nobody builds like this any more, for reasons that are not only about money.

**Concrete instruction:** let some sentences carry no data at all. A sentence
whose only job is to react, connect, or change direction is not filler. It is
what human writing is mostly made of.

---

## Rule 3: Let paragraphs be uneven

The tidy shape (an introduction, three body paragraphs of similar size, a
conclusion) is a strong machine signature. Human documents are lopsided because
the writer got absorbed in one part and hurried through another.

**Concrete instruction:** vary paragraphs from one sentence to eight. Include at
least one paragraph that is a single line. Do not give every section the same
weight, and do not open every paragraph with a topic sentence.

---

## Rule 4: Never use these words

These are the strongest vocabulary tells. Not one of them is a bad word in
isolation. Together they are a fingerprint.

```
delve        showcase     leverage      ecosystem     tapestry
robust       seamless     landscape     navigate      unlock
testament    realm        underscore    elevate       boasts
furthermore  moreover     intricate     crucial       vital
pivotal      myriad       plethora      profound      nuanced
```

Also avoid these constructions:

- "It stands as a testament to..."
- "In today's fast-paced world..."
- "It is important to note that..."
- "This raises the question of whether..."
- "Not only ... but also ..."
- "serves as a reminder that..."

**Do not write:**

> The bridge stands as a testament to robust engineering and showcases a
> seamless blend of form and function.

**Write:**

> The bridge has held for ninety years, through two floods that took out
> everything else on that stretch of river. It also happens to look good.

---

## Rule 5: Never use these characters

These survive copy and paste out of a chat window and mark text as unedited
machine output. Named by codepoint so there is no ambiguity about which
character is meant.

| Character | Codepoint | Use instead |
|---|---|---|
| em dash | U+2014 | a comma, a colon, or two sentences |
| en dash | U+2013 | a hyphen, or the word "to" |
| curly quotes | U+2018 U+2019 U+201C U+201D | straight quotes `'` and `"` |
| ellipsis | U+2026 | three periods `...` |
| right arrow | U+2192 | `->` |
| non-breaking space | U+00A0 | a normal space |

The em dash deserves special mention. Readers now treat it as the single clearest
sign that text came from a model, whether or not that is statistically fair. Two
per document is enough to get a piece dismissed.

**Do not write** (where `[U+2014]` stands for the em dash character itself, which
this file will not print):

> The result was clear `[U+2014]` the method worked.

**Write:**

> The result was clear. The method worked.

Note the fix is to restructure, not to substitute. Swapping in a hyphen keeps the
same interrupted rhythm and gives away just as much.

---

## Rule 6: Break the pattern of three

Grouping in threes is a real habit of machine prose, and once you notice it you
cannot stop noticing it.

**Do not write:**

> The policy was efficient, equitable, and enforceable.

**Write:**

> The policy was efficient and enforceable. Whether it was fair is still argued
> about.

**Concrete instruction:** use two items, or four. When you do use three, make
them different lengths, and do not make the third one the neat summarising one.

---

## Rule 7: Stop when the argument stops

Machine writing closes by restating what it just said. People simply stop.

**Do not write:**

> In conclusion, the reforms represent a significant milestone that will continue
> to shape the sector for years to come.

**Write:** nothing. Delete the paragraph. Whatever came before it was the ending.

**Concrete instruction:** never begin a final paragraph with "In conclusion", "In
summary", "Ultimately", or "Overall". If the last paragraph only repeats earlier
points in different words, cut it entirely.

---

## Rule 8: Use contractions

"It's", "doesn't", "won't", "there's". Their absence reads as stiff in almost
every register short of a legal filing.

**Do not write:**

> It is not clear that the committee did not know.

**Write:**

> It's not clear the committee didn't know.

---

## Rule 9: Let something stay unresolved

Models smooth everything into balance. Every argument gets its counterargument,
every claim its qualification. The evenness is itself a tell.

**Concrete instruction:** state at least one thing plainly without hedging it.
Leave at least one question open instead of answering it in the same paragraph.
It is fine to say a thing is bad, or that nobody really knows.

**Do not write:**

> While the policy had certain advantages, it also presented challenges, and
> opinions on its overall effectiveness remain varied.

**Write:**

> The policy did not work. People disagree about why, and I do not think the
> honest answer has been written down yet.

---

## Checklist before returning a draft

Run through this and fix anything that fails:

1. Is there a sentence under eight words? Is there one over twenty-five?
2. Do any three consecutive sentences have nearly the same length? Fix them.
3. Are the paragraphs different sizes, including at least one very short one?
4. Search for every word in the rule 4 list. Zero hits.
5. Search for the characters in rule 5. Zero hits.
6. Are there any three-item lists? Make them two or four.
7. Does the final paragraph add anything new? If not, delete it.
8. Are contractions present?
9. Is at least one statement unhedged?

---

## One thing this guide cannot do

Following every rule here makes prose read more like a person wrote it. It does
not make it "undetectable", and no set of writing rules can promise that. Anyone
selling that promise is selling something.

Also worth knowing: if you downloaded a file from a chat interface, the file
itself records where it came from, in metadata that no amount of rewriting
touches. That is a separate problem from the prose, and it needs a separate tool.
