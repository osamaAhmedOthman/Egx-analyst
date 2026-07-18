"""
ui/pages/1_analyze.py — Stock analysis page.

Shows:
  - Prediction badge (UP/DOWN + confidence)
  - Key metrics strip
  - Conflict warning (if detected)
  - Full LLM analysis
  - News sources
  - Follow-up chat box scoped to this stock and analysis
"""
from __future__ import annotations

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import sys
from pathlib import Path

# Add project root to sys.path so agent/ and src/ are importable
# Path: ui/pages/1_analyze.py → up 3 levels → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()

st.set_page_config(
    page_title="EGX Analyst — Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Guard: redirect if no ticker selected ─────────────────────────────────────
if "selected_ticker" not in st.session_state or not st.session_state.selected_ticker:
    st.warning("No stock selected. Please go back to the home page.")
    if st.button("← Back to Home"):
        st.switch_page("app.py")
    st.stop()

ticker = st.session_state.selected_ticker

# ── Ticker display metadata ───────────────────────────────────────────────────
TICKER_META = {
    "COMI_CA": {"name": "Commercial International Bank", "short": "CIB",  "emoji": "🏦"},
    "HRHO_CA": {"name": "EFG Holding",                  "short": "EFG",  "emoji": "💼"},
    "TMGH_CA": {"name": "Talaat Moustafa Group",         "short": "TMG",  "emoji": "🏗️"},
    "SWDY_CA": {"name": "Elsewedy Electric",             "short": "SWDY", "emoji": "⚡"},
    "ORWE_CA": {"name": "Oriental Weavers",              "short": "ORWE", "emoji": "🏭"},
}
meta = TICKER_META.get(ticker, {"name": ticker, "short": ticker, "emoji": "📈"})

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .badge-up {
        display: inline-block;
        background: #d4edda; color: #155724;
        border: 1.5px solid #c3e6cb;
        border-radius: 8px; padding: 10px 24px;
        font-size: 2rem; font-weight: 700;
        letter-spacing: 2px;
    }
    .badge-down {
        display: inline-block;
        background: #f8d7da; color: #721c24;
        border: 1.5px solid #f5c6cb;
        border-radius: 8px; padding: 10px 24px;
        font-size: 2rem; font-weight: 700;
        letter-spacing: 2px;
    }
    .badge-unknown {
        display: inline-block;
        background: #e2e3e5; color: #383d41;
        border: 1.5px solid #d6d8db;
        border-radius: 8px; padding: 10px 24px;
        font-size: 2rem; font-weight: 700;
    }
    .conflict-box {
        background: #fff3cd; border: 1.5px solid #ffc107;
        border-radius: 10px; padding: 16px 20px;
        margin: 12px 0;
    }
    .no-conflict-box {
        background: #d4edda; border: 1.5px solid #28a745;
        border-radius: 10px; padding: 12px 20px;
        margin: 12px 0;
    }
    .section-header {
        font-size: 1.05rem; font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px; margin-bottom: 14px;
    }
    .news-card {
        background: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 8px; padding: 12px 16px;
        margin-bottom: 10px;
    }
    .chat-msg-user {
        background: #e8f4fd; border-radius: 10px;
        padding: 10px 14px; margin: 6px 0;
        text-align: right;
    }
    .chat-msg-assistant {
        background: #f1f3f4; border-radius: 10px;
        padding: 10px 14px; margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
col_back, col_title = st.columns([1, 8])
with col_back:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_title:
    st.markdown(
        f"### {meta['emoji']} {meta['name']}  "
        f"<span style='color:#888;font-size:0.9rem'>{ticker}</span>",
        unsafe_allow_html=True,
    )

st.divider()


# ── Run analysis (cached per ticker in session) ───────────────────────────────
def run_analysis_cached(ticker: str):
    """Run pipeline only if not already cached for this ticker."""
    if ticker not in st.session_state.analysis_cache:
        from agent.analyst import run_analysis
        report = run_analysis(ticker)
        st.session_state.analysis_cache[ticker] = report

        # Add to history if not already there
        report_dict = report.to_dict()
        existing_tickers = [h["ticker"] for h in st.session_state.analysis_history]
        if ticker not in existing_tickers:
            st.session_state.analysis_history.append(report_dict)

    return st.session_state.analysis_cache[ticker]


with st.spinner(
    f"Running analysis for {meta['short']}... "
    "Fetching prediction → searching news → synthesising with LLM"
):
    try:
        report = run_analysis_cached(ticker)
    except Exception as exc:
        err_text = str(exc).lower()
        if "429" in err_text or "rate limit" in err_text or "usage limit" in err_text:
            st.warning(
                "⚠️ We've hit the API rate limit (prediction service or news/LLM provider). "
                "Please wait a moment and try again."
            )
        else:
            st.error(f"Analysis failed: {exc}")
            st.info("If this persists, check that the API container is running and reachable.")
        st.stop()


# ── Section 1: Prediction badge + key metrics ─────────────────────────────────
st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

left, mid, right = st.columns([2, 3, 3], gap="large")

with left:
    direction = report.prediction
    if direction == "UP":
        st.markdown(
            '<div class="badge-up">▲ UP</div>',
            unsafe_allow_html=True,
        )
    elif direction == "DOWN":
        st.markdown(
            '<div class="badge-down">▼ DOWN</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-unknown">— UNKNOWN</div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"Next-day predicted direction  ·  {report.prediction_date}  "
        f"(based on data through {getattr(report, 'as_of_date', 'N/A')})"
    )

with mid:
    confidence_pct = round(report.confidence * 100, 1)
    label = report.confidence_label()
    label_color = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}.get(label, "⚪")
    st.metric("Confidence", f"{confidence_pct}%", delta=f"{label} {label_color}")
    st.progress(report.confidence, text="")

with right:
    st.metric("P(UP)", f"{round(report.up_probability * 100, 1)}%")
    st.caption(f"Model: **{report.model_name}**")
    st.caption(f"Generated: {report.generated_at[:16].replace('T', ' ')}")

st.markdown(" ")

# ── Section 2: Conflict detection ────────────────────────────────────────────
st.markdown('<div class="section-header">Signal vs News Alignment</div>',
            unsafe_allow_html=True)

if report.conflict_detected:
    st.markdown(
        f"""<div class="conflict-box">
        ⚠️ <strong>Conflict Detected</strong><br>
        The ML model prediction contradicts the current news sentiment.<br><br>
        <em>{report.conflict_explanation or 'See full analysis below for details.'}</em>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """<div class="no-conflict-box">
        ✓ <strong>No Conflict</strong> — Model signal and news sentiment are aligned.
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown(" ")

# ── Section 3: Full LLM analysis ─────────────────────────────────────────────
st.markdown('<div class="section-header">LLM Analysis</div>', unsafe_allow_html=True)

# Strip the structured headers from the raw LLM output before displaying
analysis_text = report.full_analysis
for prefix in [
    "CONFLICT_DETECTED:", "CONFLICT_EXPLANATION:", "None",
    "ANALYSIS:", "تحليل:"
]:
    # Don't remove Arabic section header — keep it for display
    if prefix not in ("ANALYSIS:", "تحليل:"):
        lines = analysis_text.splitlines()
        analysis_text = "\n".join(
            line for line in lines
            if not line.strip().startswith(prefix.rstrip(":"))
        )

with st.expander("View full analysis", expanded=True):
    st.markdown(analysis_text)

st.markdown(" ")

# ── Section 4: News sources ───────────────────────────────────────────────────
st.markdown('<div class="section-header">News Sources</div>', unsafe_allow_html=True)

if report.news_summary and report.news_summary != "No recent news articles found for this company.":
    with st.expander(f"View news articles used in this analysis", expanded=False):
        st.text(report.news_summary)
else:
    st.info("No recent news articles were found for this stock.")

st.divider()

# ── Section 5: Follow-up chat ─────────────────────────────────────────────────
st.markdown('<div class="section-header">Ask a follow-up question</div>',
            unsafe_allow_html=True)
st.caption(
    f"Chat is scoped to this analysis of {meta['name']} — "
    "ask anything about the prediction, the news, the model, or the risks."
)

# Initialise chat history for this ticker
if ticker not in st.session_state.chat_histories:
    st.session_state.chat_histories[ticker] = []

chat_history: list = st.session_state.chat_histories[ticker]


def build_chat_system_prompt(report) -> str:
    """
    Build a system prompt that scopes the chat strictly to
    the current analysis context so the LLM doesn't drift
    into generic financial advice.
    """
    return f"""You are EGX Analyst, a financial analyst specialising in the Egyptian Exchange.
You are answering follow-up questions about a specific stock analysis that was just completed.

ANALYSIS CONTEXT:
- Ticker        : {report.ticker}
- Direction     : {report.prediction}
- Confidence    : {round(report.confidence * 100, 1)}%
- P(UP)         : {round(report.up_probability * 100, 1)}%
- Model         : {report.model_name}
- Predicting for: {report.prediction_date}
- Data as of    : {getattr(report, 'as_of_date', 'N/A')}
- Conflict      : {'YES — ' + (report.conflict_explanation or '') if report.conflict_detected else 'NO'}

ANALYSIS SUMMARY:
{report.full_analysis[:1500]}

NEWS SUMMARY:
{report.news_summary[:800]}

RULES:
- Only answer questions related to this specific stock and today's analysis.
- If asked about other stocks or general markets, redirect politely to this stock.
- Be concise — 2 to 4 sentences unless the question needs more depth.
- Never give explicit buy/sell recommendations. Discuss signals and risks only.
- If you are uncertain, say so honestly.
"""


def get_chat_response(user_message: str, history: list, report) -> str:
    """Call Groq LLM with the full chat history and analysis context."""
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        return "GROQ_API_KEY is not set in your .env file."

    llm = ChatGroq(
        api_key=groq_api_key,
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.3,
        max_tokens=512,
    )

    # Build message list: system + full history + new user message
    messages = [SystemMessage(content=build_chat_system_prompt(report))]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        err_text = str(exc).lower()
        if "429" in err_text or "rate limit" in err_text or "usage limit" in err_text:
            return "⚠️ The chat assistant has hit its rate limit. Please wait a moment and try asking again."
        return f"Chat error: {exc}"


# ── Render chat history ───────────────────────────────────────────────────────
for msg in chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    f"Ask about {meta['short']}... e.g. 'What are the main risks?' or 'Why is confidence low?'"
)

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    chat_history.append({"role": "user", "content": user_input})

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_chat_response(user_input, chat_history[:-1], report)
        st.markdown(answer)
    chat_history.append({"role": "assistant", "content": answer})

    # Persist updated history
    st.session_state.chat_histories[ticker] = chat_history

# ── Clear chat button ─────────────────────────────────────────────────────────
if chat_history:
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_histories[ticker] = []
        st.rerun()

# ── Re-run analysis button ────────────────────────────────────────────────────
st.divider()
col_rerun, _ = st.columns([2, 6])
with col_rerun:
    if st.button("🔄 Re-run analysis", help="Clears cache and fetches fresh data"):
        if ticker in st.session_state.analysis_cache:
            del st.session_state.analysis_cache[ticker]
        if ticker in st.session_state.chat_histories:
            st.session_state.chat_histories[ticker] = []
        st.rerun()
