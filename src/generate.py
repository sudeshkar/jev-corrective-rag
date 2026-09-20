"""The one generative call in the pipeline.

Jev handles every judgment; a generative LLM is used for the single thing a
System One model cannot do - writing prose. Keeping generation down to exactly
one call is the point of the architecture, so this module is deliberately thin.

Generation runs on Groq, whose LPU inference is about the fastest available for
open models. That is deliberate: if the thesis only held against slow providers
it would not be worth much.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False

MODEL = os.getenv("ANSWER_MODEL", "openai/gpt-oss-120b")

# Groq list prices, USD per million tokens. Verify against the current console
# pricing page before quoting these anywhere that matters.
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.29, 0.59),
    "groq/compound": (0.15, 0.75),
    "groq/compound-mini": (0.075, 0.30),
}
DEFAULT_PRICE = (0.15, 0.75)

SYSTEM = (
    "You are a support assistant. Answer using ONLY the numbered sources given. "
    "Cite the sources you use as [1], [2] inline. If the sources do not contain "
    "the answer, say so plainly instead of guessing. Be concise: three sentences "
    "or fewer unless the question needs steps."
)


@dataclass
class Generation:
    text: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = MODEL

    @property
    def usd(self) -> float:
        rate_in, rate_out = PRICING.get(self.model, DEFAULT_PRICE)
        return (
            self.input_tokens / 1_000_000 * rate_in
            + self.output_tokens / 1_000_000 * rate_out
        )


def format_sources(chunks: list) -> str:
    return "\n\n".join(
        f"[{i}] ({c.title}) {c.text}" for i, c in enumerate(chunks, start=1)
    )


class Generator:
    def __init__(self, *, force_offline: bool = False, model: str | None = None):
        self.model = model or MODEL
        self.offline = (
            force_offline or not _SDK_AVAILABLE or not os.getenv("GROQ_API_KEY")
        )
        self.client = None if self.offline else Groq()

    def answer(self, question: str, chunks: list) -> Generation:
        sources = format_sources(chunks)
        started = time.perf_counter()

        if self.offline:
            time.sleep(0.5)  # stands in for a real round trip
            text = _offline_answer(question, chunks)
            return Generation(
                text=text,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=len(sources) // 4,
                output_tokens=len(text) // 4,
                model=self.model,
            )

        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=512,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"Sources:\n{sources}\n\nQuestion: {question}",
                },
            ],
        )
        elapsed = (time.perf_counter() - started) * 1000
        usage = response.usage

        return Generation(
            text=(response.choices[0].message.content or "").strip(),
            latency_ms=elapsed,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=self.model,
        )


def _offline_answer(question: str, chunks: list) -> str:
    """Extractive stand-in so the pipeline runs end to end without a key."""
    if not chunks:
        return "The knowledge base does not cover this."
    lead = chunks[0].text.split(". ")
    body = ". ".join(lead[:2]).rstrip(".")
    return f"{body}. [1]  (OFFLINE extractive stand-in - no GROQ_API_KEY set.)"
