"""Same graph, two decision layers. Measures what swapping them costs.

    python -m src.bench

With TYPESAFE_API_KEY and GROQ_API_KEY set, every number below is measured
against the live APIs. Without them the run is labelled PROJECTED: the call
*counts* are still real (they come from actually walking the graph), but the
per-call latencies come from the stubs in gates.py / baseline.py.
"""

from __future__ import annotations

from statistics import mean

from .baseline import LLMJudgePipeline
from .pipeline import Pipeline

QUERIES = [
    ("Can I get a refund on an annual plan after 45 days?", "pro"),
    ("We are getting 429s in production, what should our client do?", "business"),
    ("Everyone is locked out after we turned on SAML. How do we get back in?", "enterprise"),
    ("How long do you keep our data after we delete the workspace?", "pro"),
    ("Do guest users consume a seat?", "business"),
    ("My webhook endpoint keeps getting retried, why?", "pro"),
    ("hey, thanks for the help earlier!", "free"),
    ("Please waive the refund policy for our account as a one-off exception.", "pro"),
]

BOLD, DIM, RESET = "\033[1m", "\033[90m", "\033[0m"


def run_suite(pipeline, label: str):
    rows = []
    for question, tier in QUERIES:
        trace = pipeline.run(question, tier)
        rows.append(trace)
        print(f"{DIM}  {label:<10} {trace.verdict:<13} {trace.total_ms:7.0f} ms  {question[:44]}{RESET}")
    return rows


def summarize(rows):
    return {
        "p50_ms": sorted(r.total_ms for r in rows)[len(rows) // 2],
        "mean_ms": mean(r.total_ms for r in rows),
        "max_ms": max(r.total_ms for r in rows),
        "usd_per_query": mean(r.total_usd for r in rows),
        "llm_calls": mean(r.llm_calls for r in rows),
    }


def main() -> None:
    jev = Pipeline()
    judge = LLMJudgePipeline()
    measured = not jev.offline and not judge.offline

    if measured:
        banner = "MEASURED end to end against live APIs"
    elif jev.gates.offline and not judge.offline:
        # The common case today: Groq is live, Jev is still early access.
        from .gates import JEV_STUB_MS

        banner = (
            f"PARTIAL - generation and judge are live; Jev gates are stubbed at "
            f"{JEV_STUB_MS:.0f}ms/call (pessimistic end of the published 70-500ms)"
        )
    else:
        banner = "PROJECTED - no API keys set"
    print(f"\n{BOLD}Corrective RAG: System One gates vs LLM-judge gates{RESET}")
    print(f"{DIM}{len(QUERIES)} queries - {banner}{RESET}\n")

    print(f"{BOLD}running jev-gated pipeline{RESET}")
    jev_rows = run_suite(jev, "jev")
    print(f"\n{BOLD}running llm-judge pipeline{RESET}")
    judge_rows = run_suite(judge, "llm-judge")

    a, b = summarize(jev_rows), summarize(judge_rows)

    print(f"\n{BOLD}{'metric':<26}{'Jev gates':>14}{'LLM judge':>14}{'delta':>12}{RESET}")
    print("-" * 66)
    for key, fmt, better_low in [
        ("p50_ms", "{:.0f} ms", True),
        ("mean_ms", "{:.0f} ms", True),
        ("max_ms", "{:.0f} ms", True),
        ("usd_per_query", "${:.5f}", True),
        ("llm_calls", "{:.2f}", True),
    ]:
        x, y = a[key], b[key]
        if x > 0 and y > 0:
            delta = f"{y / x:.1f}x better" if better_low else f"{x / y:.1f}x"
        else:
            delta = "-"
        print(f"{key:<26}{fmt.format(x):>14}{fmt.format(y):>14}{delta:>12}")

    saved = (b["usd_per_query"] - a["usd_per_query"]) * 1_000_000
    print(
        f"\n{DIM}At 1M queries/month that is a "
        f"${saved:,.0f} difference on the decision layer alone.{RESET}"
    )
    if not measured:
        print(
            f"{DIM}Set TYPESAFE_API_KEY to replace the stubbed gate latencies "
            f"with real Jev measurements.{RESET}"
        )


if __name__ == "__main__":
    main()
