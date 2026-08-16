# Architecture diagrams

Mermaid renders natively on GitHub. Companion to [ARCHITECTURE.md](../ARCHITECTURE.md);
numbers referenced here are measured in [RESULTS.md](../RESULTS.md).

---

## 1. The pipeline: how a document flows

```mermaid
flowchart LR
    IN([Input text<br/>possibly watermarked]) --> S1

    subgraph STAGE1["① Scrub"]
        S1["Unicode hygiene<br/><i>pure code, no model</i>"]
    end

    S1 --> BASE{{"④ Score<br/>baseline"}}
    BASE -->|"already human-like<br/>score ≥ 0.65"| OUT
    BASE -->|"reads AI-flat"| S2

    subgraph STAGE2["② Extract"]
        S2["Document -> Meaning<br/><i>local LLM</i>"]
    end

    S2 --> MEAN[["Meaning<br/>atomic points, no surface form"]]
    MEAN --> S3

    subgraph STAGE3["③ Regenerate"]
        S3["Meaning -> fresh prose<br/><b>LOCAL unwatermarked model</b>"]
    end

    S3 --> S4["④ Score<br/>human-signature"]
    S4 --> S5{"⑤ Gate"}
    S5 -->|"human ≥ 0.65<br/>AND topic kept<br/>AND facts kept"| OUT([Output])
    S5 -->|"fail -> raise aggressiveness<br/>max 4 iterations"| S3

    GUARD["Semantic guard<br/><i>embeddings, same topic?</i>"] -.->|"similarity ≥ 0.35"| S5
    FACTS["Fact checker<br/><i>rules, same claims?</i>"] -.->|"no dropped/inverted facts"| S5
    MEAN -.->|"Constraints:<br/>must-keep numbers & names"| S3
    VOICE[/"VoiceProfile<br/>vocabulary, banned tells"/] -.-> S3

    style STAGE3 fill:#2d3748,stroke:#e53e3e,stroke-width:3px,color:#fff
    style MEAN fill:#1a365d,color:#fff
    style OUT fill:#22543d,color:#fff
    style IN fill:#4a5568,color:#fff
    style FACTS fill:#553c1a,color:#fff
```

The red border on Stage ③ marks the **Self-Watermark Trap**: regenerating with a
watermarked model strips the old mark and stamps a fresh one, so that stage must
run a local open-weight model. `Pipeline.__init__` refuses a regenerator not
declared `is_unwatermarked`.

---

## 1b. Why the gate needs two tests

```mermaid
flowchart TB
    C["Candidate rewrite"] --> G & F & N

    G["<b>Semantic guard</b><br/>embedding cosine"]
    F["<b>Rules</b><br/>numerals · names · NegEx polarity"]
    N["<b>NLI</b><br/>entailment on aligned claims"]

    G --> GQ{"Same topic?"}
    F --> FQ{"Numbers, names,<br/>polarity intact?"}
    N --> NQ{"Does it<br/>contradict?"}

    GQ -->|"reword 0.650<br/>drift 0.042"| GOK["✓ topic drift"]
    GQ -.->|"'are' vs 'are not'<br/>= 0.959"| GBAD["✗ blind to truth value"]

    FQ -->|"3/3 lexical<br/>0 false alarms"| FOK["✓ dropped facts<br/>+ lexical flips"]
    FQ -.->|"0/8 reworded"| FBAD["✗ blind to reworded reversal"]

    NQ -->|"8/8 reworded"| NOK["✓ reworded reversal"]
    NQ -.->|"bounded by backend;<br/>needs embedding alignment"| NBAD["✗ 2/8 if aligned lexically"]

    GOK --> PASS(["Emit only if ALL pass"])
    FOK --> PASS
    NOK --> PASS

    style GBAD fill:#742a2a,color:#fff
    style FBAD fill:#742a2a,color:#fff
    style NBAD fill:#742a2a,color:#fff
    style PASS fill:#22543d,color:#fff
    style GOK fill:#1a365d,color:#fff
    style FOK fill:#1a365d,color:#fff
    style NOK fill:#1a365d,color:#fff
```

Each instrument's blind spot is another's competence, and the red boxes are the
honest residue. Embedding blindness to negation is a documented property of
distributional representations, not a tuning failure, no similarity floor fixes
it. The rules are lexical: NegEx negation plus a curated antonym list, so a
reversal in different words is invisible to them. NLI sees that, but only because
**embeddings do its alignment**, align lexically and it collapses to 2/8, since
the reworded pairs never reach the model.

Order matters: rules run first and NLI can only *add* findings, so a model that
is unavailable or wrong can never weaken the deterministic guarantee.

---

## 2. Why it survives future models: the substrate attack

```mermaid
flowchart TB
    subgraph SCHEMES["Watermark schemes, all different"]
        K["KGW / green-list<br/>biases vocabulary split"]
        SY["SynthID-Text<br/>keyed tournament"]
        SE["Semantic watermarks<br/>SIR / XSIR"]
        FUT["A 2028 scheme<br/>that does not exist yet"]
    end

    K --> SUB
    SY --> SUB
    SE --> SUB
    FUT --> SUB

    SUB[["THE SHARED SUBSTRATE<br/>the specific token sequence emitted"]]

    SUB --> ATK{"Reweave:<br/>discard surface,<br/>regenerate from meaning"}
    ATK --> GONE([Signal has nothing<br/>left to ride on])

    style SUB fill:#742a2a,color:#fff,stroke:#e53e3e,stroke-width:3px
    style ATK fill:#1a365d,color:#fff
    style GONE fill:#22543d,color:#fff
    style FUT stroke-dasharray: 5 5
```

They differ in *how* they bias the sequence; they are identical in *depending* on
it. We never model a scheme, we remove the ground it stands on. That is why a
scheme nobody has published yet is already handled.

---

## 3. Stable core / swappable edge: dependencies point inward

```mermaid
flowchart TB
    subgraph EDGE["SWAPPABLE EDGE, changes often, has dependencies"]
        direction LR
        SC["scrub/<br/>unicode"]
        EX["extract/<br/>ollama"]
        RG["regenerate/<br/>ollama"]
        ZS["score/<br/>statistical"]
        GD["guard/<br/>embedding"]
    end

    subgraph CORE["STABLE CORE, zero dependencies, no scheme code"]
        direction LR
        T["types.py<br/>data contracts"]
        I["interfaces.py<br/>5 stage ABCs"]
        P["pipeline.py<br/>orchestration + loop"]
        R["registry.py<br/>plug-in resolution"]
    end

    SC -->|implements| I
    EX -->|implements| I
    RG -->|implements| I
    ZS -->|implements| I
    GD -->|implements| I

    NEW["A 2028 model adapter"] -.->|"one class<br/>+ one @register line"| I

    style CORE fill:#1a365d,color:#fff,stroke:#63b3ed,stroke-width:3px
    style EDGE fill:#2d3748,color:#fff
    style NEW stroke-dasharray: 5 5,fill:#22543d,color:#fff
```

The core imports nothing from the edge and nothing third-party, so it cannot rot.
Adopting a new model is one adapter plus one registry line, **if it requires a
core edit, an invariant was violated.**

---

## 4. How the harness proves it: ground truth requires holding the key

```mermaid
flowchart TB
    KEY(["Our own key"]) --> WM
    LM["Word-level LM<br/>trained on real English"] --> WM
    WM["Tournament sampling<br/><i>pure-Python SynthID ref</i>"] --> WMT[["Watermarked text"]]

    WMT --> DET1{{"Detect with our key<br/>score 0.708"}}
    WMT --> PIPE["Reweave pipeline<br/>local model regenerates"]
    PIPE --> OUTT[["Regenerated text"]]
    OUTT --> DET2{{"Detect with our key<br/>score 0.506"}}

    BASE[["Unwatermarked baseline<br/>0.501"]] -.->|"compare"| DET2
    DET2 --> VERDICT([" Watermark gone<br/>still-detected 0.000"])

    GOOG["Real Claude / Gemini text"] -.->|"key is secret, <br/>NOT measurable"| NOPE(["Can only report<br/>human-signature + meaning"])

    style KEY fill:#744210,color:#fff
    style VERDICT fill:#22543d,color:#fff
    style NOPE fill:#742a2a,color:#fff
    style GOOG stroke-dasharray: 5 5
```

You cannot verify removal of a *keyed* watermark without the key, so the harness
watermarks with its own. The right branch is the permanent honesty boundary: for
third-party text we report only what we can actually measure.

---

## 5. The convergence loop: optimising to a fixed target

```mermaid
stateDiagram-v2
    [*] --> Scrubbed
    Scrubbed --> Baseline: score

    Baseline --> Done: already human-like
    Baseline --> Extracted: reads AI-flat

    Extracted --> Regenerated: aggressiveness 0.45
    Regenerated --> Judged: score + guard + facts

    Judged --> Done: human ✓ AND topic ✓ AND facts ✓
    Judged --> Regenerated: fail -> aggressiveness +0.18
    Judged --> BestSoFar: iteration cap reached

    BestSoFar --> Done: emit best meaning-preserving candidate
    Done --> [*]

    note right of Judged
        Target is FIXED: statistical human-ness.
        Never a named detector, that is
        the arms race we refuse to enter.
    end note
```

Because the target is human-ness rather than any particular detector, a new
detector shipping tomorrow requires no change here.

---

## 6. Where the numbers land

```mermaid
flowchart LR
    subgraph PROVEN["✅ Measured"]
        A["Watermark detection<br/>AUROC 1.000"]
        B["Removal by real model<br/>0.708 -> 0.506"]
        C["Invariant I4<br/>10% edit survives<br/>100% regen dies"]
        D["Embedding guard<br/>margin +0.608"]
        I["Fact gate, rules<br/>5/5 lexical inversions<br/>0 false alarms"]
        J["Fact gate, NLI<br/>8/8 reworded reversals<br/>rules alone: 0/8"]
    end

    subgraph OPEN["🚧 Not proven"]
        E["Google's watermark<br/>key unmeasurable"]
        F["Fidelity beyond the backend<br/>NLI SOTA is 74.4%"]
        G["Production regenerator<br/>needs bigger local model"]
        H["Calibrated AI boundary<br/>needs perplexity backend"]
    end

    style PROVEN fill:#22543d,color:#fff
    style OPEN fill:#742a2a,color:#fff
```
