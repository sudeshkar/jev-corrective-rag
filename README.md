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

| metric | Jev gates | LLM judge | delta |
|---|---:|---:|---:|
| p50 latency | 2,780 ms | 11,014 ms | **4.0× better** |
| mean latency | 2,219 ms | 9,929 ms | 4.5× better |
| **worst case** | **3,164 ms** | **19,636 ms** | **6.2× better** |
| LLM calls / query | 0.75 | 5.38 | **7.2× fewer** |
| cost / query | $0.00015 | $0.00038 | 2.6× better |

Reproduce with `python -m src.bench`.

Three things worth more than the headline speedup:

**The tail, not the median.** An LLM judge's latency compounds across N chunks
because you are waiting on N token streams. Typed gates fan out flat. 3.1s worst
case against 19.6s is the difference between a support widget and a support
ticket.

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
range, chosen so a stubbed run *under-sells* the latency case. The benchmark
prints `PARTIAL` and names the stub whenever this is what ran.

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
python -m src.cli                    # walk the sample queries
python -m src.cli "your question"    # ask one
python -m src.bench                  # Jev gates vs LLM judge
```

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
- Label a few hundred queries and measure gate *accuracy*, not just speed — the
  latency case is settled, the quality case is not.
- Sweep the confidence floors to plot the coverage/precision curve.
- Cache triage decisions on a hash of the question; support traffic repeats.

## Licence

MIT
