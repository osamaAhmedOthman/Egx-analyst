"""
ui/pages/2_history.py — Past analyses from the current session.

Shows a summary table of every stock analysed since the app started,
with the ability to jump back to any previous analysis.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

st.set_page_config(
    page_title="EGX Analyst — History",
    page_icon="🕘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px; margin-bottom: 14px;
    }
    .history-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_back, col_title = st.columns([1, 8])
with col_back:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_title:
    st.markdown("### 🕘 Analysis History")

st.caption("All analyses run in this session. Clears when you close the browser.")
st.divider()

# ── Guard: no history yet ─────────────────────────────────────────────────────
history: list = st.session_state.get("analysis_history", [])

if not history:
    st.info(
        "No analyses have been run yet in this session.\n\n"
        "Go back to the home page and click a stock to run your first analysis."
    )
    st.stop()

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Session Summary</div>',
            unsafe_allow_html=True)

TICKER_NAMES = {
    "COMI_CA": "Commercial International Bank",
    "HRHO_CA": "EFG Holding",
    "TMGH_CA": "Talaat Moustafa Group",
    "SWDY_CA": "Elsewedy Electric",
    "ORWE_CA": "Oriental Weavers",
}

rows = []
for h in history:
    ticker      = h.get("ticker", "—")
    direction   = h.get("prediction", "—")
    confidence  = h.get("confidence", 0)
    conflict    = h.get("conflict_detected", False)
    model       = h.get("model_name", "—")
    generated   = h.get("generated_at", "—")[:16].replace("T", " ")

    direction_icon = "▲ UP" if direction == "UP" else ("▼ DOWN" if direction == "DOWN" else "—")
    conflict_icon  = "⚠ YES" if conflict else "✓ NO"

    rows.append({
        "Ticker":    ticker,
        "Company":   TICKER_NAMES.get(ticker, ticker),
        "Direction": direction_icon,
        "Confidence":f"{round(confidence * 100, 1)}%",
        "Conflict":  conflict_icon,
        "Model":     model,
        "Run At":    generated,
    })

summary_df = pd.DataFrame(rows)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)

st.markdown(" ")

# ── Detailed cards ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Detailed View</div>',
            unsafe_allow_html=True)

for h in reversed(history):   # most recent first
    ticker    = h.get("ticker", "—")
    direction = h.get("prediction", "—")
    conf_pct  = round(h.get("confidence", 0) * 100, 1)
    conflict  = h.get("conflict_detected", False)
    analysis  = h.get("full_analysis", "")
    generated = h.get("generated_at", "")[:16].replace("T", " ")
    model     = h.get("model_name", "—")
    conf_label= h.get("confidence_label", "")

    dir_color = "#155724" if direction == "UP" else "#721c24"
    dir_bg    = "#d4edda" if direction == "UP" else "#f8d7da"
    dir_arrow = "▲" if direction == "UP" else "▼"

    with st.container():
        st.markdown(
            f"""<div class="history-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="font-size:1.1rem">{ticker}</strong>
                    &nbsp;&nbsp;
                    <span style="color:#888">{TICKER_NAMES.get(ticker, '')}</span>
                </div>
                <div style="font-size:0.8rem; color:#888">{generated}</div>
            </div>
            <div style="margin-top:10px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
                <span style="background:{dir_bg}; color:{dir_color};
                             padding:4px 14px; border-radius:6px;
                             font-weight:700; font-size:1.1rem;">
                    {dir_arrow} {direction}
                </span>
                <span style="color:#555; font-size:0.9rem;">
                    Confidence: <strong>{conf_pct}%</strong>
                </span>
                <span style="color:#555; font-size:0.9rem;">
                    Model: <strong>{model}</strong>
                </span>
                <span style="color:{'#856404' if conflict else '#155724'};
                             font-size:0.9rem; font-weight:500;">
                    {'⚠ Conflict detected' if conflict else '✓ No conflict'}
                </span>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Show truncated analysis snippet
        if analysis:
            snippet = analysis[:400].strip()
            if len(analysis) > 400:
                snippet += "..."
            with st.expander("View analysis snippet"):
                st.markdown(snippet)

        # Jump back to this analysis
        col_jump, _ = st.columns([2, 6])
        with col_jump:
            if st.button(f"View full analysis →", key=f"jump_{ticker}"):
                st.session_state.selected_ticker = ticker
                st.switch_page("pages/1_analyze.py")

        st.markdown(" ")

# ── Clear history ─────────────────────────────────────────────────────────────
st.divider()
col_clear, _ = st.columns([2, 6])
with col_clear:
    if st.button("🗑 Clear all history", type="secondary"):
        st.session_state.analysis_history = []
        st.session_state.analysis_cache   = {}
        st.session_state.chat_histories   = {}
        st.rerun()
