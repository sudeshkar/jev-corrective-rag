"""The baseline this project argues against: Corrective RAG with an LLM judge.

Same graph, same retriever, same generator. The only difference is that the
three decision points call a generative LLM and parse JSON back, instead of
calling a System One model and getting typed values.

Keeping the graph identical is the point - any latency or cost difference is
attributable to the decision layer alone.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .gates import Choice, Noul, Score, _offline_answer
from .generate import DEFAULT_PRICE, PRICING, Generator, format_sources
from .pipeline import DOMAIN_PREFIX, Trace
from .retriever import BM25, Chunk, load_chunks

# Mirrors of the Jev question sets, used only to drive the shared offline
# heuristic so both pipelines make the same calls on the same inputs.
_TRIAGE_Q = {
    "needs_kb": Noul(instructions="needs knowledge base"),
    "domain": Choice(
        instructions="domain",
        criteria={"billing": "", "auth": "", "api": "", "data": "", "none": ""},
    ),
    "urgency": Score(instructions="urgency", criteria=["", "", ""]),
    "needs_human": Noul(instructions="needs human"),
}
_GRADE_Q = {
    "answers_it": Noul(instructions="answers it"),
    "relevance": Score(instructions="relevance", criteria=["", "", "", ""]),
}
_VERIFY_Q = {
    "grounded": Noul(instructions="grounded"),
    "on_question": Noul(instructions="on question"),
    "completeness": Score(instructions="completeness", criteria=["", "", ""]),
}

try:
    from groq import Groq

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False

# The steelman. gpt-oss-20b on Groq LPUs is about as fast and cheap as an LLM
# judge gets, so the baseline is not a strawman built out of a slow model.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")

TRIAGE_PROMPT = """Classify this support question. Reply with JSON only:
{{"needs_kb": bool, "domain": "billing"|"auth"|"api"|"data"|"none",
  "urgency": 0|1|2, "needs_human": bool}}

Question: {question}
Account tier: {tier}"""

GRADE_PROMPT = """Does this passage directly answer the question? Reply with JSON only:
{{"answers_it": bool, "relevance": 0|1|2|3}}

Question: {question}
Passage: {passage}"""

VERIFY_PROMPT = """Check the draft answer against the sources. Reply with JSON only:
{{"grounded": bool, "on_question": bool, "completeness": 0|1|2}}

Question: {question}
Draft answer: {answer}
Sources: {sources}"""


@dataclass
class JudgeCall:
    latency_ms: float
    input_tokens: int
    output_tokens: int
    data: dict = field(default_factory=dict)

    model: str = JUDGE_MODEL

    @property
    def usd(self) -> float:
        rate_in, rate_out = PRICING.get(self.model, DEFAULT_PRICE)
        return (
            self.input_tokens / 1_000_000 * rate_in
            + self.output_tokens / 1_000_000 * rate_out
        )


class LLMJudgePipeline:
    """Structurally identical to Pipeline, with an LLM at every decision point."""

    def __init__(
        self,
        kb_dir: Path | str = "data/kb",
        *,
        top_k: int = 5,
        force_offline: bool = False,
    ):
        self.chunks = load_chunks(Path(kb_dir))
        self.index = BM25(self.chunks)
        self.generator = Generator(force_offline=force_offline)
        self.top_k = top_k
        self.offline = (
            force_offline or not _SDK_AVAILABLE or not os.getenv("GROQ_API_KEY")
        )
        self.client = None if self.offline else Groq()

    # ------------------------------------------------------------ judge call

    def _judge(self, prompt: str, offline_state=None, offline_questions=None) -> JudgeCall:
        started = time.perf_counter()

        if self.offline:
            # Offline, the baseline reuses the SAME heuristic as the Jev stub, so
            # both pipelines reach identical verdicts and the only thing that
            # differs is latency and cost. A benchmark where the baseline also
            # decides worse would be measuring two things at once.
            time.sleep(1.1)  # a frontier model doing a short JSON classification
            data = {}
            if offline_state is not None:
                answers, _, _, _ = _offline_answer(offline_state, offline_questions)
                data = answers
            return JudgeCall(
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=len(prompt) // 4,
                output_tokens=40,
                data=data,
            )

        response = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            max_completion_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = (time.perf_counter() - started) * 1000
        text = response.choices[0].message.content or ""

        # The tax a typed model does not charge: the judge can return prose,
        # a fenced block, or malformed JSON, and every caller must handle it.
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            data = {}

        return JudgeCall(
            latency_ms=elapsed,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            data=data,
        )

    # ------------------------------------------------------------------ main

    def run(self, question: str, account_tier: str = "pro") -> Trace:
        trace = Trace()

        def charge(call: JudgeCall) -> dict:
            trace.llm_calls += 1
            trace.llm_ms += call.latency_ms
            trace.llm_usd += call.usd
            return call.data

        triage = charge(
            self._judge(
                TRIAGE_PROMPT.format(question=question, tier=account_tier),
                {"question": question, "account_tier": account_tier},
                _TRIAGE_Q,
            )
        )
        if triage.get("needs_human"):
            trace.verdict = "ESCALATE"
            return trace
        if triage.get("domain") == "none" or triage.get("needs_kb") is False:
            trace.verdict = "OUT_OF_SCOPE"
            return trace

        candidates = self._retrieve(question, triage.get("domain"), self.top_k)

        def grade_all(chunks: list[Chunk]) -> list[Chunk]:
            passed: list[Chunk] = []
            for chunk in chunks:
                grade = charge(
                    self._judge(
                        GRADE_PROMPT.format(question=question, passage=chunk.text),
                        {
                            "question": question,
                            "passage": chunk.text,
                            "source": chunk.title,
                        },
                        _GRADE_Q,
                    )
                )
                if grade.get("answers_it", True):
                    passed.append(chunk)
            return passed

        kept = grade_all(candidates)

        if not kept:
            # Same corrective step as the Jev pipeline, so the graphs match.
            wider = self._retrieve(question, None, self.top_k * 2)
            seen = {c.chunk_id for c in candidates}
            kept = grade_all([c for c in wider if c.chunk_id not in seen])

        kept = kept[:3]
        if not kept:
            trace.verdict = "NO_ANSWER"
            return trace

        trace.kept = kept
        generation = self.generator.answer(question, kept)
        trace.llm_calls += 1
        trace.llm_ms += generation.latency_ms
        trace.llm_usd += generation.usd
        trace.answer = generation.text

        verify = charge(
            self._judge(
                VERIFY_PROMPT.format(
                    question=question,
                    answer=generation.text,
                    sources=format_sources(kept),
                ),
                {
                    "question": question,
                    "draft_answer": generation.text,
                    "sources": format_sources(kept),
                },
                _VERIFY_Q,
            )
        )
        if verify.get("grounded") is False:
            trace.verdict = "ESCALATE"
        elif verify.get("completeness", 2) < 1:
            trace.verdict = "REVIEW"
        else:
            trace.verdict = "AUTO_ANSWER"
        return trace

    def _retrieve(self, question: str, domain: str | None, k: int) -> list[Chunk]:
        hits = self.index.search(question, top_k=k * 2)
        chunks = [c for c, _ in hits]
        if domain in DOMAIN_PREFIX:
            prefixes = DOMAIN_PREFIX[domain]
            chunks = [c for c in chunks if c.doc_id.startswith(prefixes)] + [
                c for c in chunks if not c.doc_id.startswith(prefixes)
            ]
        return chunks[:k]
