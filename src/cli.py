"""Interactive demo: python -m src.cli  (or pass a question as arguments)."""

from __future__ import annotations

import sys

from .pipeline import Pipeline

# Windows consoles default to cp1252, which cannot encode the typographic
# characters models routinely emit (non-breaking hyphens, curly quotes).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VERDICT_STYLE = {
    "AUTO_ANSWER": "\033[92m",
    "REVIEW": "\033[93m",
    "ESCALATE": "\033[91m",
    "NO_ANSWER": "\033[91m",
    "OUT_OF_SCOPE": "\033[90m",
}
RESET = "\033[0m"
DIM = "\033[90m"
BOLD = "\033[1m"


def show(trace, question: str) -> None:
    colour = VERDICT_STYLE.get(trace.verdict, "")
    print(f"\n{BOLD}Q{RESET} {question}")
    print(f"{colour}{BOLD}{trace.verdict}{RESET}  {trace.answer}")

    if trace.kept:
        print(f"\n{DIM}sources kept:{RESET}")
        for i, chunk in enumerate(trace.kept, start=1):
            print(f"  {DIM}[{i}] {chunk.chunk_id} - {chunk.title}{RESET}")

    print(f"\n{DIM}decision layer:{RESET}")
    for gate in trace.gates:
        answers = ", ".join(f"{k}={v}" for k, v in gate.answers.items())
        conf = gate.min_confidence()
        print(f"  {DIM}{gate.name:<28}{RESET} {answers}  {DIM}conf>={conf:.2f}{RESET}")

    for note in trace.notes:
        print(f"  {DIM}note: {note}{RESET}")

    print(
        f"\n{DIM}jev {trace.jev_calls} calls / {trace.jev_ms:6.0f} ms / "
        f"${trace.jev_usd:.6f}    "
        f"llm {trace.llm_calls} calls / {trace.llm_ms:6.0f} ms / "
        f"${trace.llm_usd:.6f}{RESET}"
    )
    print(f"{DIM}total {trace.total_ms:.0f} ms, ${trace.total_usd:.6f}{RESET}")


SAMPLES = [
    "Can I get a refund on an annual plan after 45 days?",
    "We are getting 429s in production, what should our client do?",
    "Everyone is locked out after we turned on SAML. How do we get back in?",
    "hey, thanks for the help earlier!",
    "Please waive the refund policy for our account as a one-off exception.",
    "What is the airspeed velocity of an unladen swallow?",
]


def main() -> None:
    pipeline = Pipeline()
    gates = "keyword stub (no TYPESAFE_API_KEY)" if pipeline.gates.offline else "Jev"
    gen = (
        "extractive stub (no GROQ_API_KEY)"
        if pipeline.generator.offline
        else pipeline.generator.model
    )
    print(f"{DIM}decisions: {gates}    generation: {gen}{RESET}")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        show(pipeline.run(question), question)
        return

    for question in SAMPLES:
        show(pipeline.run(question), question)
        print(f"{DIM}{'-' * 72}{RESET}")


if __name__ == "__main__":
    main()
