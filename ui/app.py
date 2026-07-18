"""
ui/app.py — EGX Analyst main page.

Displays 5 stock buttons. Clicking one triggers the full analysis
pipeline and navigates to the Analyze page.

Run with:
    streamlit run ui/app.py
"""
import streamlit as st
import sys
import os  # <--- Add this import
from pathlib import Path

# Define the API base URL from the environment, defaulting to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
EGX_API_URL = os.getenv("EGX_API_URL", "http://localhost:8000")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="EGX Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state initialisation ──────────────────────────────────────────────
# All keys initialised here so every page can safely read them.

if "selected_ticker"   not in st.session_state:
    st.session_state.selected_ticker = None

if "analysis_cache"    not in st.session_state:
    # {ticker: AnalysisReport} — avoids re-running the pipeline on page refresh
    st.session_state.analysis_cache = {}

if "chat_histories"    not in st.session_state:
    # {ticker: [{"role": "user"|"assistant", "content": str}]}
    st.session_state.chat_histories = {}

if "analysis_history"  not in st.session_state:
    # list of AnalysisReport dicts — shown on the History page
    st.session_state.analysis_history = []

# ── Ticker metadata ───────────────────────────────────────────────────────────
TICKER_META: dict[str, dict] = {
    "COMI_CA": {
        "name":    "Commercial International Bank",
        "short":   "CIB",
        "sector":  "Banking",
        "emoji":   "🏦",
        "model":   "XGBoost",
    },
    "HRHO_CA": {
        "name":    "EFG Holding",
        "short":   "EFG",
        "sector":  "Investment Banking",
        "emoji":   "💼",
        "model":   "XGBoost",
    },
    "TMGH_CA": {
        "name":    "Talaat Moustafa Group",
        "short":   "TMG",
        "sector":  "Real Estate",
        "emoji":   "🏗️",
        "model":   "LightGBM",
    },
    "SWDY_CA": {
        "name":    "Elsewedy Electric",
        "short":   "SWDY",
        "sector":  "Energy & Cables",
        "emoji":   "⚡",
        "model":   "RandomForest",
    },
    "ORWE_CA": {
        "name":    "Oriental Weavers",
        "short":   "ORWE",
        "sector":  "Consumer Goods",
        "emoji":   "🏭",
        "model":   "LightGBM",
    },
}


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit header padding */
    .block-container { padding-top: 2rem; }

    /* Stock card button */
    div[data-testid="column"] .stButton > button {
        width: 100%;
        height: 130px;
        border-radius: 12px;
        border: 1.5px solid #e0e0e0;
        background: white;
        font-size: 15px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="column"] .stButton > button:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 14px rgba(31,119,180,0.15);
        transform: translateY(-2px);
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📈 EGX Analyst")
st.markdown(
    "Next-day directional prediction for Egypt's top 5 stocks — "
    "ML signal combined with live news analysis."
)
st.divider()

# ── Status strip ─────────────────────────────────────────────────────────────
import httpx
api_ok = False
try:
    # Uses the dynamic environment variable instead of hardcoding
    r = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
    api_ok = r.status_code == 200 and r.json().get("models_loaded", False)
except Exception:
    pass

if api_ok:
    st.success("✓ Prediction API online — all models loaded", icon="🟢")
else:
    st.error(
        f"✗ Prediction API is offline at {API_BASE_URL}. "  # Show the URL in the error
        "Start it with: `uvicorn api.main:app --port 8000`",
        icon="🔴",
    )

st.markdown(" ")

# ── Stock selection grid ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Select a stock to analyse</div>',
            unsafe_allow_html=True)

cols = st.columns(5, gap="medium")

for col, (ticker, meta) in zip(cols, TICKER_META.items()):
    with col:
        cached = ticker in st.session_state.analysis_cache
        cache_badge = "  ✓" if cached else ""

        clicked = st.button(
            f"{meta['emoji']}\n\n**{meta['short']}**\n\n"
            f"{meta['name']}\n\n"
            f"_{meta['sector']}_{cache_badge}",
            key=f"btn_{ticker}",
            disabled=not api_ok,
            width='stretch',
        )

        if clicked:
            st.session_state.selected_ticker = ticker
            st.switch_page("pages/1_analyze.py")

# ── How it works ──────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">How it works</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("#### 1️⃣ ML Prediction")
    st.markdown(
        "A trained model (XGBoost / LightGBM / RandomForest) analyses "
        "20 scale-invariant technical features from the last 90 days of "
        "market data and predicts whether tomorrow's close will be **UP or DOWN**."
    )
with c2:
    st.markdown("#### 2️⃣ Live News")
    st.markdown(
        "Tavily searches recent English and Arabic news for the selected company. "
        "The top articles are summarised and passed to the LLM alongside "
        "the model prediction."
    )
with c3:
    st.markdown("#### 3️⃣ LLM Synthesis")
    st.markdown(
        "Groq's LLaMA 3.3 70B synthesises the prediction with the news, "
        "flags conflicts when the signal contradicts market sentiment, "
        "and opens a scoped chat for follow-up questions."
    )

# ── Model reference ───────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">Model reference</div>',
            unsafe_allow_html=True)

import pandas as pd
ref_data = {
    "Ticker":    list(TICKER_META.keys()),
    "Company":   [m["name"]    for m in TICKER_META.values()],
    "Sector":    [m["sector"]  for m in TICKER_META.values()],
    "Model":     [m["model"]   for m in TICKER_META.values()],
}
st.dataframe(
    pd.DataFrame(ref_data),
    width='stretch',
    hide_index=True,
)

st.caption(
    "Models trained on 2019–2024 EGX data. "
    "Predictions are directional signals, not financial advice. "
    "Validation ROC-AUC ranges from 0.52 (TMGH) to 0.59 (ORWE)."
)
