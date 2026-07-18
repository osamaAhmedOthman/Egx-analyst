"""
agent/graph.py — LangGraph graph definition.

Three nodes in sequence:
  data_fetcher   → calls FastAPI /predict, populates prediction fields
  news_searcher  → calls Tavily, populates news fields
  analyst        → calls Groq LLM, produces final analysis

The graph is compiled once at module level and reused across calls.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.prompts import (
    ANALYST_SYSTEM_PROMPT,
    ANALYST_USER_PROMPT,
    build_feature_highlights,
)
from agent.schemas import AnalystState
from agent.tools import (
    call_prediction_api,
    format_news_for_prompt,
    search_news,
    COMPANY_NAMES,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM setup ─────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = 2048
GROQ_TEMPERATURE = 0.2   # low temperature for consistent structured output


def _get_llm() -> ChatGroq:
    """Instantiate the Groq LLM. Called inside the analyst node."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here"
        )
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )


# ── Node 1: data_fetcher ──────────────────────────────────────────────────────

def data_fetcher_node(state: AnalystState) -> Dict[str, Any]:
    """
    Node 1 — Call the FastAPI prediction endpoint.

    Reads  : state.ticker
    Writes : prediction, confidence, up_probability, model_name,
             prediction_date, as_of_date, feature_snapshot, api_error
    """
    logger.info(f"[data_fetcher] Fetching prediction for {state.ticker}")

    result = call_prediction_api(state.ticker)

    if "error" in result:
        logger.error(f"[data_fetcher] API error for {state.ticker}: {result['error']}")
        return {"api_error": result["error"]}

    return {
        "prediction":      result.get("direction"),
        "confidence":      result.get("confidence"),
        "up_probability":  result.get("up_probability"),
        "model_name":      result.get("model_name"),
        "prediction_date": result.get("prediction_date"),
        "as_of_date":      result.get("as_of_date"),
        "feature_snapshot":result.get("feature_snapshot", {}),
        "api_error":       None,
    }


# ── Node 2: news_searcher ─────────────────────────────────────────────────────

def news_searcher_node(state: AnalystState) -> Dict[str, Any]:
    """
    Node 2 — Search for recent news via Tavily.

    Reads  : state.ticker
    Writes : news_results, news_summary, news_error
    """
    logger.info(f"[news_searcher] Searching news for {state.ticker}")

    news_results = search_news(state.ticker, max_results=5)

    if not news_results:
        logger.warning(f"[news_searcher] No news found for {state.ticker}")
        return {
            "news_results": [],
            "news_summary": "No recent news articles found for this company.",
            "news_error":   "No results returned by Tavily.",
        }

    news_summary = format_news_for_prompt(news_results)
    logger.info(f"[news_searcher] Found {len(news_results)} articles for {state.ticker}")

    return {
        "news_results": news_results,
        "news_summary": news_summary,
        "news_error":   None,
    }


# ── Node 3: analyst ───────────────────────────────────────────────────────────

def analyst_node(state: AnalystState) -> Dict[str, Any]:
    """
    Node 3 — Synthesise prediction + news using Groq LLM.

    Reads  : all fields populated by nodes 1 and 2
    Writes : conflict_detected, conflict_explanation,
             full_analysis, generated_at
    """
    logger.info(f"[analyst] Running LLM synthesis for {state.ticker}")

    # ── Handle upstream failures gracefully ───────────────────────────────────
    if state.api_error:
        error_analysis = (
            f"Unable to generate analysis for {state.ticker}. "
            f"Prediction API error: {state.api_error}"
        )
        return {
            "conflict_detected":    False,
            "conflict_explanation": None,
            "full_analysis":        error_analysis,
            "generated_at":         datetime.now().isoformat(),
        }

    # ── Build prompt ──────────────────────────────────────────────────────────
    company      = COMPANY_NAMES.get(state.ticker, {})
    company_name = company.get("en", state.ticker)

    confidence_pct    = round((state.confidence or 0) * 100, 1)
    up_prob_pct       = round((state.up_probability or 0) * 100, 1)
    confidence_label  = (
        "High"     if confidence_pct >= 65 else
        "Moderate" if confidence_pct >= 55 else
        "Low"
    )
    feature_highlights = build_feature_highlights(state.feature_snapshot)

    user_prompt = ANALYST_USER_PROMPT.format(
        ticker             = state.ticker,
        company_name       = company_name,
        direction          = state.prediction or "UNKNOWN",
        confidence_pct     = confidence_pct,
        up_probability_pct = up_prob_pct,
        model_name         = state.model_name or "Unknown",
        prediction_date    = state.prediction_date or "Unknown",
        confidence_label   = confidence_label,
        feature_highlights = feature_highlights,
        news_text          = state.news_summary or "No news available.",
    )

    # ── Call Groq LLM ─────────────────────────────────────────────────────────
    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response     = llm.invoke(messages)
        raw_analysis = response.content.strip()
        logger.info(f"[analyst] LLM response received for {state.ticker} ({len(raw_analysis)} chars)")

    except Exception as exc:
        logger.exception(f"[analyst] LLM call failed for {state.ticker}: {exc}")
        return {
            "conflict_detected":    False,
            "conflict_explanation": None,
            "full_analysis":        f"LLM analysis unavailable: {exc}",
            "generated_at":         datetime.now().isoformat(),
        }

    # ── Parse conflict detection from LLM response ────────────────────────────
    conflict_detected    = False
    conflict_explanation = None
    full_analysis        = raw_analysis

    lines = raw_analysis.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip().upper()

        # Detect conflict flag line
        if stripped.startswith("CONFLICT_DETECTED:"):
            value = line.split(":", 1)[-1].strip().upper()
            conflict_detected = value.startswith("YES")
            continue

        # Detect conflict explanation block
        if stripped.startswith("CONFLICT_EXPLANATION:"):
            explanation_lines = []
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                # Stop at the next section header
                if next_line.upper().startswith("ANALYSIS:") or next_line.upper().startswith("تحليل"):
                    break
                if next_line and next_line.lower() != "none":
                    explanation_lines.append(next_line)
            if explanation_lines:
                conflict_explanation = " ".join(explanation_lines).strip()
            break

    logger.info(
        f"[analyst] {state.ticker} — "
        f"conflict_detected={conflict_detected}, "
        f"analysis_length={len(full_analysis)} chars"
    )

    return {
        "conflict_detected":    conflict_detected,
        "conflict_explanation": conflict_explanation,
        "full_analysis":        full_analysis,
        "generated_at":         datetime.now().isoformat(),
    }


# ── Graph compilation ─────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Graph topology:
        START → data_fetcher → news_searcher → analyst → END

    The three nodes run sequentially — each node reads from the shared
    AnalystState and writes its outputs back into it. LangGraph merges
    the returned dicts into the state automatically.
    """
    builder = StateGraph(AnalystState)

    # Register nodes
    builder.add_node("data_fetcher",  data_fetcher_node)
    builder.add_node("news_searcher", news_searcher_node)
    builder.add_node("analyst",       analyst_node)

    # Wire edges
    builder.add_edge(START,          "data_fetcher")
    builder.add_edge("data_fetcher", "news_searcher")
    builder.add_edge("news_searcher","analyst")
    builder.add_edge("analyst",       END)

    return builder.compile()


# Compile once at import time — reused across all run_analysis() calls
graph = build_graph()
logger.info("LangGraph agent compiled successfully.")
