"""Corrective RAG whose decision layer is a System One model.

  question
     |
     +-- [Jev] triage ............ needs KB? domain? urgent? human-only?
     |        `-- exits here for chitchat / policy exceptions. 0 LLM calls.
     |
     +-- BM25 retrieve
     |
     +-- [Jev] grade x N .......... does this passage ACTUALLY answer it?
     |        `-- all rejected -> widen once -> still none? 0 LLM calls.
     |
     +-- [Claude] generate ........ the only generative call in the system
     |
     +-- [Jev] verify ............. grounded? on-question? complete?
              `-- calibrated confidence -> AUTO / REVIEW / ESCALATE

The design rule: a generative model is never asked a question whose answer
space is already known. That is what makes the loop cheap enough to run
per-request in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gates import DecisionLayer, GateResult
from .generate import Generator, format_sources
from .retriever import BM25, Chunk, load_chunks

# Confidence floors. Jev returns calibrated probabilities, so these are real
# knobs rather than vibes: raise them to trade coverage for precision.
RELEVANCE_CONF_FLOOR = 0.50
VERIFY_CONF_FLOOR = 0.60

DOMAIN_PREFIX = {
    "billing": ("billing", "onboarding"),
    "auth": ("auth",),
    "api": ("api",),
    "data": ("data",),
}


@dataclass
class Trace:
    """Everything that happened, for the CLI and the benchmark to report on."""

    verdict: str = ""
    answer: str = ""
    gates: list[GateResult] = field(default_factory=list)
    llm_calls: int = 0
    jev_calls: int = 0
    jev_ms: float = 0.0
    llm_ms: float = 0.0
    jev_usd: float = 0.0
    llm_usd: float = 0.0
    kept: list[Chunk] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, *results: GateResult) -> None:
        for r in results:
            self.gates.append(r)
            self.jev_calls += 1
            self.jev_ms += r.latency_ms
            self.jev_usd += r.usd

    @property
    def total_ms(self) -> float:
        return self.jev_ms + self.llm_ms

    @property
    def total_usd(self) -> float:
        return self.jev_usd + self.llm_usd


class Pipeline:
    def __init__(
        self,
        kb_dir: Path | str = "data/kb",
        *,
        top_k: int = 5,
        force_offline: bool = False,
    ):
        self.chunks = load_chunks(Path(kb_dir))
        self.index = BM25(self.chunks)
        self.gates = DecisionLayer(force_offline=force_offline)
        self.generator = Generator(force_offline=force_offline)
        self.top_k = top_k

    @property
    def offline(self) -> bool:
        return self.gates.offline or self.generator.offline

    # ------------------------------------------------------------------ main

    def run(self, question: str, account_tier: str = "pro") -> Trace:
        trace = Trace()

        # --- Gate 1: triage, before any retrieval -------------------------
        triage = self.gates.triage(question, account_tier)
        trace.record(triage)

        if triage.answers.get("needs_human"):
            trace.verdict = "ESCALATE"
            trace.answer = (
                "This needs a human: it asks for a policy exception or a "
                "contractual decision."
            )
            trace.notes.append("triage.needs_human -> stopped before retrieval")
            return trace

        if not triage.answers.get("needs_kb") or triage.answers.get("domain") == "none":
            trace.verdict = "OUT_OF_SCOPE"
            trace.answer = "That is not something the knowledge base covers."
            trace.notes.append("triage.needs_kb=False -> 0 retrievals, 0 LLM calls")
            return trace

        # --- Retrieve, biased by the domain Jev just picked ---------------
        domain = triage.answers.get("domain")
        candidates = self._retrieve(question, domain, self.top_k)

        # --- Gate 2: relevance grading, fanned out ------------------------
        kept = self._grade_and_keep(question, candidates, trace)

        if not kept:
            # Corrective step: the first retrieval was junk. Widen once and
            # regrade rather than handing junk to the generator.
            trace.notes.append("all chunks rejected -> widening retrieval once")
            wider = self._retrieve(question, None, self.top_k * 2)
            fresh = [c for c in wider if c.chunk_id not in {x.chunk_id for x in candidates}]
            kept = self._grade_and_keep(question, fresh, trace)

        if not kept:
            trace.verdict = "NO_ANSWER"
            trace.answer = (
                "I could not find a passage that answers this. Routing to a human."
            )
            trace.notes.append("no passage passed grading -> 0 LLM calls")
            return trace

        trace.kept = kept

        # --- The one generative call --------------------------------------
        generation = self.generator.answer(question, kept)
        trace.llm_calls = 1
        trace.llm_ms += generation.latency_ms
        trace.llm_usd += generation.usd

        # --- Gate 3: groundedness ------------------------------------------
        verify = self.gates.verify(question, generation.text, format_sources(kept))
        trace.record(verify)

        trace.answer = generation.text
        trace.verdict = self._policy(verify, trace)
        return trace

    # ------------------------------------------------------------- internals

    def _retrieve(self, question: str, domain: str | None, k: int) -> list[Chunk]:
        hits = self.index.search(question, top_k=k * 2)
        chunks = [c for c, _ in hits]
        if domain and domain in DOMAIN_PREFIX:
            prefixes = DOMAIN_PREFIX[domain]
            preferred = [c for c in chunks if c.doc_id.startswith(prefixes)]
            others = [c for c in chunks if not c.doc_id.startswith(prefixes)]
            chunks = preferred + others
        return chunks[:k]

    def _grade_and_keep(
        self, question: str, candidates: list[Chunk], trace: Trace
    ) -> list[Chunk]:
        if not candidates:
            return []
        grades = self.gates.grade(question, candidates)
        trace.record(*grades)

        kept: list[tuple[Chunk, Any]] = []
        for chunk, grade in zip(candidates, grades):
            passes = grade.answers.get("answers_it") and grade.confidence.get(
                "answers_it", 0.0
            ) >= RELEVANCE_CONF_FLOOR
            if passes:
                kept.append((chunk, grade.answers.get("relevance", 0)))

        kept.sort(key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in kept[:3]]

    @staticmethod
    def _policy(verify: GateResult, trace: Trace) -> str:
        """Typed answers + calibrated confidence collapse into one of three lanes."""
        grounded = verify.answers.get("grounded")
        on_question = verify.answers.get("on_question")
        completeness = verify.answers.get("completeness", 0)
        confidence = verify.min_confidence()

        if not grounded:
            trace.notes.append("verify.grounded=False -> answer suppressed")
            return "ESCALATE"
        if not on_question or completeness < 1:
            trace.notes.append("answer incomplete or off-question")
            return "REVIEW"
        if confidence < VERIFY_CONF_FLOOR:
            trace.notes.append(f"confidence {confidence:.2f} below floor -> human review")
            return "REVIEW"
        return "AUTO_ANSWER"
