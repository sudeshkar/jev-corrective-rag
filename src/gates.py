"""The decision layer.

Every judgment this pipeline makes runs through Jev (TypeSafe AI's System One
model): typed questions in, typed answers + calibrated confidence out, one
parallel pass, no token generation.

Three gates:
  triage()  - before retrieval. Do we even need the KB? Which domain? Risky?
  grade()   - per retrieved chunk. Does this actually answer the question?
  verify()  - after generation. Is the answer grounded in the chunks?

Each returns a GateResult carrying the typed answers, per-answer confidence, and
wall-clock latency, so the benchmark can compare this against an LLM judge.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in offline demo mode
    _SDK_AVAILABLE = False

    class _Stub:
        def __init__(self, instructions: str, criteria: Any = None):
            self.instructions = instructions
            self.criteria = criteria

    class Choice(_Stub):
        pass

    class Noul(_Stub):
        pass

    class Score(_Stub):
        pass

    TypeSafeClient = None  # type: ignore[assignment]


MODEL = os.getenv("JEV_MODEL", "jev-latest")

# Jev pricing at launch: $0.042 per 1M input tokens. Output is effectively free.
JEV_USD_PER_INPUT_MTOK = 0.042

# What the offline stub sleeps per call, in ms. TypeSafe publishes 70-500ms
# end-to-end for Jev; this defaults to the pessimistic end of that range on
# purpose, so a stubbed run under-sells the latency case rather than over-sells
# it. Any headline number should come from a real key, not from this.
JEV_STUB_MS = float(os.getenv("JEV_STUB_MS", "300"))


@dataclass
class GateResult:
    """One Jev call: what it decided, how sure it was, how long it took."""

    name: str
    answers: dict[str, Any]
    confidence: dict[str, float]
    latency_ms: float
    input_tokens: int = 0
    probabilities: dict[str, Any] = field(default_factory=dict)

    @property
    def usd(self) -> float:
        return self.input_tokens / 1_000_000 * JEV_USD_PER_INPUT_MTOK

    def min_confidence(self) -> float:
        return min(self.confidence.values()) if self.confidence else 1.0


class DecisionLayer:
    """Wraps the Jev client. Falls back to a labelled offline stub without a key.

    The stub exists so the demo runs on a laptop with no credentials. It is
    keyword-based and clearly marked; it is NOT a simulation of Jev's quality.
    """

    def __init__(self, *, force_offline: bool = False):
        self.offline = (
            force_offline or not _SDK_AVAILABLE or not os.getenv("TYPESAFE_API_KEY")
        )
        self.client = None if self.offline else TypeSafeClient()

    # --------------------------------------------------------------- plumbing

    def _ask(self, name: str, state: Any, questions: dict[str, Any]) -> GateResult:
        started = time.perf_counter()

        if self.offline:
            answers, confidence, probabilities, tokens = _offline_answer(state, questions)
        else:
            response = self.client.system_one(state=state, questions=questions)
            answers, confidence, probabilities = {}, {}, {}
            for key, question in questions.items():
                ans = response.answers[key]
                if isinstance(question, Choice):
                    answers[key] = ans.choice
                elif isinstance(question, Score):
                    answers[key] = ans.score
                else:
                    answers[key] = bool(ans.noul)
                confidence[key] = float(getattr(ans, "confidence", 1.0))
                probabilities[key] = getattr(ans, "probabilities", None)
            tokens = _estimate_tokens(state, questions)

        return GateResult(
            name=name,
            answers=answers,
            confidence=confidence,
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=tokens,
            probabilities=probabilities,
        )

    # ------------------------------------------------------------------ gates

    def triage(self, query: str, account_tier: str) -> GateResult:
        """Gate 1 - runs BEFORE the vector store is touched.

        A chitchat or out-of-scope query exits here having spent a fraction of a
        cent and ~100ms, instead of a retrieval round-trip plus an LLM judge.
        """
        return self._ask(
            "triage",
            state={"question": query, "account_tier": account_tier},
            questions={
                "needs_kb": Noul(
                    instructions=(
                        "Answering this correctly requires looking up product "
                        "documentation, policy, or account specifics. Greetings, "
                        "thanks, and general chitchat do not."
                    )
                ),
                "domain": Choice(
                    instructions="Which documentation area covers this question",
                    criteria={
                        "billing": "Payments, invoices, refunds, seats, pricing",
                        "auth": "Login, SSO, SAML, MFA, account access",
                        "api": "API usage, rate limits, webhooks, integrations",
                        "data": "Data export, retention, deletion, privacy",
                        "none": "Not a product question at all",
                    },
                ),
                "urgency": Score(
                    instructions="How time-critical this is for the customer",
                    criteria=[
                        "General curiosity, no deadline",
                        "Blocked on normal work",
                        "Production is down or money is at risk",
                    ],
                ),
                "needs_human": Noul(
                    instructions=(
                        "This requires a human: it asks for an exception to policy, "
                        "a contractual decision, an irreversible destructive action, "
                        "or discloses another person's private data."
                    )
                ),
            },
        )

    def grade(self, query: str, chunks: list[Any]) -> list[GateResult]:
        """Gate 2 - relevance grading, one Jev call per chunk, fanned out.

        This is where an LLM judge hurts most: N calls of pure judgment that the
        user waits on. Here each is milliseconds and fractions of a cent.
        """

        def one(chunk: Any) -> GateResult:
            return self._ask(
                f"grade:{chunk.chunk_id}",
                state={"question": query, "passage": chunk.text, "source": chunk.title},
                questions={
                    "answers_it": Noul(
                        instructions=(
                            "This passage contains information that directly answers "
                            "the question. Being on the same topic is not enough."
                        )
                    ),
                    "relevance": Score(
                        instructions="How useful this passage is for answering",
                        criteria=[
                            "Unrelated to the question",
                            "Same topic but does not answer it",
                            "Contains part of the answer",
                            "Contains the complete answer",
                        ],
                    ),
                },
            )

        if not chunks:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as pool:
            return list(pool.map(one, chunks))

    def verify(self, query: str, answer: str, context: str) -> GateResult:
        """Gate 3 - groundedness check on the generated answer.

        Typed and calibrated, so the escalation policy is a threshold on a
        number rather than a regex over a judge model's prose.
        """
        return self._ask(
            "verify",
            state={"question": query, "draft_answer": answer, "sources": context},
            questions={
                "grounded": Noul(
                    instructions=(
                        "Every factual claim in the draft answer is supported by the "
                        "sources. An unsupported specific (a number, a deadline, a "
                        "menu path) makes this false."
                    )
                ),
                "on_question": Noul(
                    instructions="The draft answer actually addresses what was asked."
                ),
                "completeness": Score(
                    instructions="How completely the draft answers the question",
                    criteria=[
                        "Does not answer it",
                        "Partially answers it",
                        "Fully answers it",
                    ],
                ),
            },
        )


# ---------------------------------------------------------------- estimation

def _estimate_tokens(state: Any, questions: dict[str, Any]) -> int:
    """~4 chars/token. Used for the cost column only."""
    blob = str(state)
    for q in questions.values():
        blob += str(q.instructions) + str(getattr(q, "criteria", ""))
    return max(1, len(blob) // 4)


# -------------------------------------------------------------- offline stub

_DOMAIN_HINTS = {
    "billing": ("refund", "invoice", "bill", "charge", "seat", "price", "payment", "vat"),
    "auth": ("sso", "saml", "mfa", "login", "password", "2fa", "scim", "authenticator"),
    "api": ("api", "rate", "429", "webhook", "endpoint", "retry", "signature"),
    "data": ("export", "delete", "retention", "privacy", "gdpr", "purge", "backup"),
}


def _offline_answer(state: Any, questions: dict[str, Any]):
    """Keyword heuristic used when no TYPESAFE_API_KEY is present.

    Shape-compatible with the real thing so the pipeline code is identical, but
    the numbers it produces are not Jev's. Always labelled OFFLINE in output.
    """
    time.sleep(JEV_STUB_MS / 1000)  # see JEV_STUB_MS: deliberately pessimistic
    text = str(state).lower()
    answers: dict[str, Any] = {}
    confidence: dict[str, float] = {}

    for key, question in questions.items():
        if isinstance(question, Choice):
            criteria = question.criteria or {}
            hits = {
                opt: sum(w in text for w in _DOMAIN_HINTS.get(opt, ())) for opt in criteria
            }
            best = max(hits, key=lambda k: hits[k]) if hits else "none"
            answers[key] = best if hits.get(best, 0) else "none"
            confidence[key] = 0.55 + 0.1 * min(hits.get(best, 0), 4)
        elif isinstance(question, Score):
            span = len(question.criteria or [1, 2, 3])
            answers[key] = round(_overlap(state) * (span - 1))
            confidence[key] = 0.6
        else:
            answers[key] = _offline_noul(key, state)
            confidence[key] = 0.62

    return answers, confidence, {}, _estimate_tokens(state, questions)


def _overlap(state: Any) -> float:
    from .retriever import tokenize

    if not isinstance(state, dict):
        return 0.5
    question = str(state.get("question", ""))
    body = str(state.get("passage") or state.get("draft_answer") or "")
    if not question or not body:
        return 0.5
    q, b = set(tokenize(question)), set(tokenize(body))
    return len(q & b) / len(q) if q else 0.0


def _offline_noul(key: str, state: Any) -> bool:
    if key == "needs_kb":
        text = str(state.get("question", "")).lower() if isinstance(state, dict) else ""
        chitchat = ("hello", "hi ", "thanks", "thank you", "good morning", "how are you")
        return not any(c in text for c in chitchat) and len(text.split()) > 3
    if key == "needs_human":
        text = str(state).lower()
        return any(w in text for w in ("exception", "lawsuit", "legal", "waive", "sue"))
    if key == "answers_it":
        return _overlap(state) > 0.22
    if key in ("grounded", "on_question"):
        return _overlap(state) > 0.2
    return True
