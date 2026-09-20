# LinkedIn post

Paste-ready. Pick one. Add the repo link at the end and a screenshot of the
`python -m src.bench` table as the image — the table is the post's best asset.

---

## Option A — the numbers lead (recommended)

My RAG pipeline makes **0.75 LLM calls per query.**

Less than one. Here's how that happens.

TypeSafe AI shipped Jev this week — a "System One" model. It doesn't generate
text. You hand it a state and a set of typed questions, and it returns every
answer in one parallel pass with calibrated confidence, in a schema it can't
break out of.

So I went looking for a system drowning in questions that aren't really
generative. Corrective RAG is exactly that. Per query it makes three judgment
calls:

→ Should we retrieve at all?
→ Is this chunk actually relevant? (once per chunk)
→ Is the generated answer grounded?

With 5 chunks that's ~7 model calls, and exactly one of them writes something a
user reads. The other six are the system asking itself yes/no questions in
English, waiting on a token stream, then parsing JSON back out of prose. The
judge can also hallucinate its own verdict — which is a strange way to build a
hallucination check.

I moved all six to Jev and left the LLM with the one job it's needed for.

Same retriever, same generator, same graph. Only the decision layer changed:

• p50 latency — 4.0× lower
• worst case — 19.6s → 3.2s
• LLM calls per query — 5.38 → 0.75

The tail matters more than the median. An LLM judge's latency compounds across N
chunks because you're waiting on N token streams. Typed gates fan out flat.

And escalation stops being a vibe. Jev returns calibrated probabilities, so the
policy is a threshold: below 0.60 confidence, a human reads it. With an LLM
judge you get a confident sentence either way.

One caveat I'll state plainly: Jev is early access and I don't have a key yet.
Generation and the LLM-judge baseline are live on Groq and genuinely measured.
The Jev gates are stubbed at 300ms/call — the pessimistic end of the published
70–500ms — so the numbers under-sell rather than over-sell. The benchmark prints
PARTIAL and names the stub whenever that's what ran. Swap a key in and it prints
measured figures.

The design rule the whole thing follows: **a generative model should never be
asked a question whose answer space is already known.**

Code, benchmark harness, and the full honest-status table:
[link]

#RAG #LLM #AIEngineering #Python

---

## Option B — shorter, the idea leads

We've been using generative models to answer yes/no questions, and paying
generative latency for it.

A Corrective RAG pipeline makes three judgments per query — should we retrieve,
is this chunk relevant (once per chunk), is the answer grounded. That's ~7 model
calls where exactly one writes something a user reads. The other six are the
system asking itself questions in English and parsing JSON back out of prose.

TypeSafe AI's Jev is a "System One" model — no text generation, just typed
questions in, typed answers plus calibrated confidence out, one parallel pass.
That's the exact shape of a RAG decision gate.

So I rebuilt the decision layer on it. Same retriever, same generator, same
graph:

• LLM calls per query: 5.38 → 0.75
• Worst case latency: 19.6s → 3.2s

Below one LLM call per query, because most queries never reach the generative
model. "hey, thanks!" exits at triage. A request to waive a policy escalates to
a human before retrieval. An ungrounded answer gets suppressed after generation.

Honest caveat: Jev is early access and I don't have a key. The Groq half is
measured; the Jev gates are stubbed at the pessimistic end of the published
latency range, and the benchmark labels itself PARTIAL when that's what ran.

Rule I'd keep either way: never ask a generative model a question whose answer
space you already know.

[link]

#RAG #LLM #AIEngineering

---

## Notes on posting

- **Lead image:** screenshot the `python -m src.bench` output table. Numbers in
  an image stop the scroll better than numbers in text.
- **Keep the caveat paragraph.** It's the part that reads as an engineer rather
  than a marketer, and it's the first thing a technical reviewer checks for.
- **First comment:** drop the repo link there too — LinkedIn suppresses reach on
  posts with outbound links in the body.
- **Timing:** Tue–Thu, 8–10am in your target market's timezone.
- If anyone asks "why not just use a small LLM as the judge" — the baseline
  already *is* one. `gpt-oss-20b` on Groq LPUs, about as fast and cheap as an
  LLM judge gets. That's a good comment to be ready for.
