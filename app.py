"""Streamlit demo: streamlit run app.py

Shows the decision layer, not just the answer. The point of the UI is that you
can watch each typed gate fire and see exactly where a query exited - which is
the part a screenshot of a chat window never conveys.
"""

from __future__ import annotations

import streamlit as st

from src.pipeline import RELEVANCE_CONF_FLOOR, VERIFY_CONF_FLOOR, Pipeline

st.set_page_config(page_title="Corrective RAG with typed gates", layout="wide")

VERDICT = {
    "AUTO_ANSWER": ("Answered automatically", "#16a34a"),
    "REVIEW": ("Flagged for human review", "#d97706"),
    "ESCALATE": ("Escalated to a human", "#dc2626"),
    "NO_ANSWER": ("No grounded answer found", "#dc2626"),
    "OUT_OF_SCOPE": ("Out of scope", "#64748b"),
}

SAMPLES = [
    "Can I get a refund on an annual plan after 45 days?",
    "We are getting 429s in production, what should our client do?",
    "Everyone is locked out after we turned on SAML. How do we get back in?",
    "How long do you keep our data after we delete the workspace?",
    "My webhook endpoint keeps getting retried, why?",
    "hey, thanks for the help earlier!",
    "Please waive the refund policy for our account as a one-off exception.",
]


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline()


pipeline = get_pipeline()

# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Runtime")
    gates_live = not pipeline.gates.offline
    gen_live = not pipeline.generator.offline

    st.markdown(
        f"**Decision layer**  \n"
        + ("Jev (live)" if gates_live else "keyword stub - no `TYPESAFE_API_KEY`")
    )
    st.markdown(
        f"**Generation**  \n"
        + (pipeline.generator.model if gen_live else "extractive stub - no `GROQ_API_KEY`")
    )

    if not gates_live:
        st.info(
            "Jev is early access. The gates fall back to a keyword stub that "
            "sleeps 300ms per call - the pessimistic end of the published "
            "70-500ms range, so this under-sells rather than over-sells.",
            icon=None,
        )

    st.divider()
    st.subheader("Policy")
    st.caption(
        "Jev returns calibrated probabilities, so escalation is a threshold "
        "rather than a judgement call."
    )
    st.code(
        f"RELEVANCE_CONF_FLOOR = {RELEVANCE_CONF_FLOOR}\n"
        f"VERIFY_CONF_FLOOR    = {VERIFY_CONF_FLOOR}",
        language="python",
    )
    tier = st.selectbox("Account tier", ["free", "pro", "business", "enterprise"], index=1)

# ---------------------------------------------------------------------- main

st.title("Corrective RAG with a System One decision layer")
st.caption(
    "Every decision gate is a typed model call. The generative model is used "
    "once, to write the sentence - and often not at all."
)

if "question" not in st.session_state:
    st.session_state.question = SAMPLES[0]

st.write("**Try one:**")
cols = st.columns(4)
for i, sample in enumerate(SAMPLES):
    if cols[i % 4].button(sample[:38] + ("..." if len(sample) > 38 else ""), key=f"s{i}"):
        st.session_state.question = sample

question = st.text_input("Question", value=st.session_state.question)

if st.button("Run", type="primary") or question:
    with st.spinner("Running gates..."):
        trace = pipeline.run(question, tier)

    label, colour = VERDICT.get(trace.verdict, (trace.verdict, "#64748b"))
    st.markdown(
        f"<div style='border-left:4px solid {colour};padding:0.6rem 1rem;"
        f"background:rgba(127,127,127,0.08);border-radius:4px'>"
        f"<div style='color:{colour};font-weight:700;font-size:0.8rem;"
        f"letter-spacing:0.05em'>{label.upper()}</div>"
        f"<div style='margin-top:0.4rem'>{trace.answer}</div></div>",
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("LLM calls", trace.llm_calls)
    b.metric("Typed gate calls", trace.jev_calls)
    c.metric("Total latency", f"{trace.total_ms:.0f} ms")
    d.metric("Cost", f"${trace.total_usd:.6f}")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Decision layer")
        for gate in trace.gates:
            answers = "  ".join(f"`{k}`={v}" for k, v in gate.answers.items())
            st.markdown(
                f"**{gate.name}** &nbsp; {answers} &nbsp; "
                f"<span style='color:#64748b'>conf&ge;{gate.min_confidence():.2f} "
                f"&middot; {gate.latency_ms:.0f}ms</span>",
                unsafe_allow_html=True,
            )
        for note in trace.notes:
            st.caption(f"-> {note}")

    with right:
        st.subheader("Sources kept")
        if trace.kept:
            for i, chunk in enumerate(trace.kept, start=1):
                with st.expander(f"[{i}] {chunk.title}"):
                    st.caption(chunk.chunk_id)
                    st.write(chunk.text)
        else:
            st.caption("None - the query never reached generation.")
