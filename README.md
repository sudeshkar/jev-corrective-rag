# jev-corrective-rag

Corrective RAG where every decision gate is a typed **System One** model call
instead of an LLM judge.

**0.75 LLM calls per query.** Below one, because most queries never reach the
generative model at all.

---

## The problem

Corrective RAG is a good pattern with an expensive habit. Per query it makes
three judgment calls, and in every implementation I have seen, each one is
another LLM round trip:

1. **Should we retrieve at all?** — routing
2. **Is this chunk actually relevant?** — grading, once per chunk
3. **Is the generated answer grounded?** — hallucination check

With 5 retrieved chunks that is ~7 model calls, of which exactly **one** writes
anything a user reads. The other six are the system asking itself yes/no
questions in English, waiting on a token stream, then parsing JSON back out of
prose — and the judge can hallucinate its own verdict, or return a fenced code
block, or drop a field.

You are paying generative-model latency for a classification problem.

## The idea

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is a
System One model: it does not generate text. You give it a state and a set of
**typed questions**, and it returns every answer in a single parallel pass, with
calibrated confidence, in a schema it cannot deviate from.

That is exactly the shape of a RAG decision gate. So all six judgment calls move
to Jev, and the LLM is left with the one job it is actually needed for — writing
the sentence.

This is not an LLM replacement, and the repo does not treat it as one. It is a
filter that holds back the traffic which never needed a generative model, and
forwards only what does. Three layers: observe, **judge (System 1)**, reason
(System 2) — the naming is Kahneman's, and it is the blueprint the pipeline
follows literally.

```
question
   │
   ├── [Jev] triage ............ needs KB? domain? urgent? human-only?
   │        └── chitchat and policy exceptions exit here.        0 LLM calls
   │
   ├── BM25 retrieve
   │
   ├── [Jev] grade × N ......... does this passage ACTUALLY answer it?
   │        └── all rejected → widen once → still none?          0 LLM calls
   │
   ├── [LLM]  generate ......... the only generative call in the system
   │
   └── [Jev] verify ............ grounded? on-question? complete?
            └── calibrated confidence → AUTO / REVIEW / ESCALATE
```

The design rule the whole repo follows:

> **A generative model is never asked a question whose answer space is already
> known.**

## Results

8 support queries, same retriever, same generator, same graph. The only variable
is what sits at the three decision points.

**What is structural.** These come out the same on every run, because they follow
from the shape of the graph rather than from how a model felt that afternoon:

| metric | Jev gates | LLM judge | delta |
|---|---:|---:|---:|
| LLM calls / query | 0.75 | 5.38 | **7.2× fewer** |
| cost / query | $0.00015 | $0.00038 | 2.6× better |

**What is not.** Latency is the number to be suspicious of, mine included. Two
runs of the same 8 queries:

| latency | Jev gates | LLM judge | delta |
|---|---:|---:|---:|
| p50, run A | 2,780 ms | 11,014 ms | 4.0× |
| p50, run B | 2,815 ms | 4,613 ms | 1.6× |
| worst case, run A | 3,164 ms | 19,636 ms | 6.2× |
| worst case, run B | 3,078 ms | 16,875 ms | 5.5× |

The gate side barely moves — it is a fixed-latency stub (see
[Honest status](#honest-status)), so it is an assumption, not a measurement. The
judge side moves by a factor of two between runs on shared inference capacity.
**A single median multiplier from this benchmark is not a real number**, so I am
not quoting one.

Reproduce with `python -m src.bench`, and expect your judge column to differ.

Three things that matter more than any single speedup figure:

**The tail, not the median.** An LLM judge's latency compounds across N chunks
because you are waiting on N token streams; typed gates fan out flat. That shape
difference is the durable part — the judge's worst case ran 16–20s across both
runs while the gate path stayed near 3s, and the gap at the tail survived the
run-to-run variance that flattened the median. It is also the part a stubbed
gate can least justify on its own, so treat the magnitude as pending a live key
and the direction as argued from the graph.

**Below one LLM call per query.** `hey, thanks for the help earlier!` costs one
typed call and exits. `Please waive the refund policy as a one-off exception`
escalates to a human *before* retrieval. An ungrounded answer is suppressed
after generation. The generative model is never woken for any of them.

**Escalation becomes a number.** Jev returns calibrated probabilities, so the
policy is a threshold, not a vibe:

```python
if not grounded:                       return "ESCALATE"   # answer suppressed
if confidence < VERIFY_CONF_FLOOR:     return "REVIEW"     # human checks it
return "AUTO_ANSWER"
```

Raise `RELEVANCE_CONF_FLOOR` / `VERIFY_CONF_FLOOR` in `src/pipeline.py` to trade
coverage for precision. With an LLM judge this knob does not exist — you get a
confident sentence either way.

### Why the confidence number is worth gating on

This is the part that makes the threshold defensible rather than decorative.

RLHF trains a model on answers a human rated confident and helpful. Optimise for
sounding confident and you get a model that hallucinates convincingly — its
stated certainty carries little information about whether it is right.

Jev is trained with **RLCD** (Reinforcement Learning from Calibrated Decisions),
where wrong answers are explicitly punished. The consequence is that ambiguity
in the input shows up as a *lower probability score* rather than a confident
fabrication. That is what turns `confidence < 0.60 → human review` into a real
control, instead of a threshold on a number that means nothing.

## Honest status

Read this before quoting any number above.

| Component | State |
|---|---|
| Generation (`openai/gpt-oss-120b` on Groq) | **Live and measured** |
| LLM-judge baseline (`openai/gpt-oss-20b` on Groq) | **Live and measured** |
| Jev decision gates | **Stubbed** — no API key yet, Jev is early access |

The Jev integration in `src/gates.py` is written against the documented SDK
shape and is wired end to end, but has not run against the live API. Without a
`TYPESAFE_API_KEY` the gates fall back to a keyword heuristic that sleeps
**300 ms per call** — the pessimistic end of TypeSafe's published 70–500 ms
range, and consistent with a developer-reported 300 ms measured from Chennai
including transatlantic network latency. Chosen so a stubbed run *under-sells*
the latency case. The benchmark prints `PARTIAL` and names the stub whenever
this is what ran.

Add a key and the same command prints `MEASURED` with real figures.

The baseline is deliberately a steelman: `gpt-oss-20b` on Groq LPUs is about as
fast and as cheap as an LLM judge gets. Beating a slow judge would prove
nothing.

## Quickstart

```bash
git clone https://github.com/sudeshkar/jev-corrective-rag
cd jev-corrective-rag
pip install -r requirements.txt

cp .env.example .env        # add GROQ_API_KEY; TYPESAFE_API_KEY optional
```

```bash
streamlit run app.py                 # visual demo - watch each gate fire
python -m src.cli                    # walk the sample queries
python -m src.cli "your question"    # ask one
python -m src.bench                  # Jev gates vs LLM judge
```

The Streamlit app is the one to open first. It shows the decision layer rather
than just the answer: every typed gate, what it decided, its confidence and its
latency, plus where a query exited. Watching `hey, thanks!` terminate at triage
with zero LLM calls makes the argument faster than any paragraph.

Everything runs without any key at all — both layers fall back to labelled
stubs, so the graph is inspectable offline.

Sample output:

```
Q Everyone is locked out after we turned on SAML. How do we get back in?
AUTO_ANSWER  If SSO (SAML) was misconfigured and locked everyone out, the
workspace owner should use the break-glass recovery link emailed at setup
time; it bypasses SSO for 60 minutes. [1]

decision layer:
  triage                  needs_kb=True, domain=auth, urgency=1, needs_human=False
  grade:auth-sso#3        answers_it=True,  relevance=3
  grade:auth-sso#0        answers_it=False, relevance=0
  grade:auth-mfa#1        answers_it=False, relevance=0
  verify                  grounded=True, on_question=True, completeness=2

jev 7 calls / 2105 ms / $0.000050    llm 1 calls / 708 ms / $0.000126
```

## How a gate is written

The whole point is that a gate is a schema, not a prompt:

```python
self._ask(
    "triage",
    state={"question": query, "account_tier": account_tier},
    questions={
        "needs_kb": Noul(
            instructions="Answering this requires looking up product "
                         "documentation, policy, or account specifics."
        ),
        "domain": Choice(
            instructions="Which documentation area covers this question",
            criteria={
                "billing": "Payments, invoices, refunds, seats, pricing",
                "auth":    "Login, SSO, SAML, MFA, account access",
                "api":     "API usage, rate limits, webhooks, integrations",
                "data":    "Data export, retention, deletion, privacy",
                "none":    "Not a product question at all",
            },
        ),
        "needs_human": Noul(
            instructions="This requires a human: a policy exception, a "
                         "contractual decision, or an irreversible action."
        ),
    },
)
```

`Noul` is a yes/no with a probability, `Choice` picks from options you defined,
`Score` ranks against ordered criteria. There is no parsing step, because there
is no prose to parse.

## Layout

```
app.py           Streamlit demo - surfaces every gate, not just the answer
src/
  gates.py       the three Jev gates + offline stub      ← the interesting file
  pipeline.py    orchestration and the escalation policy
  baseline.py    same graph with an LLM at every gate    ← the thing being beaten
  retriever.py   chunking + BM25, dependency-free
  generate.py    the single generative call (Groq)
  cli.py         interactive demo
  bench.py       the comparison harness
data/kb/         8 support documents
```

Retrieval is plain BM25 on purpose. It needs no embedding key, it is
deterministic, and it keeps vector-store noise out of a benchmark that is
measuring the decision layer.

## Things I would do next

- Run the gates against the live Jev API and replace the stubbed latencies.
- Re-run the benchmark repeatedly against a live key and report a distribution
  rather than one figure — neither the latency case nor the quality case is
  settled on a single stubbed run.
- Label a few hundred queries and measure gate *accuracy*, not just speed.
- Sweep the confidence floors to plot the coverage/precision curve.
- Cache triage decisions on a hash of the question; support traffic repeats.

## Licence

MIT
