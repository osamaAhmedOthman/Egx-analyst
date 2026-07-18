"""
agent/schemas.py — Pydantic models for the LangGraph agent.

Two models:
  AnalystState   : the shared state object passed between all three graph nodes.
                   Each node reads from it and writes its outputs back into it.
  AnalysisReport : the final structured output returned to the caller (UI / API).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Graph state ───────────────────────────────────────────────────────────────

class AnalystState(BaseModel):
    """
    Shared mutable state passed through all three LangGraph nodes.

    Node responsibilities:
      data_fetcher   → populates: ticker, prediction, confidence,
                                  up_probability, model_name, prediction_date,
                                  feature_snapshot
      news_searcher  → populates: news_results, news_summary
      analyst        → populates: conflict_detected, conflict_explanation,
                                  full_analysis, generated_at
    """

    # ── Input (set before graph runs) ─────────────────────────────────────────
    ticker: str = Field(..., description="EGX ticker, e.g. 'ORWE_CA'")

    # ── Populated by data_fetcher node ────────────────────────────────────────
    prediction:       Optional[str]   = Field(default=None)   # "UP" or "DOWN"
    confidence:       Optional[float] = Field(default=None)   # 0.0–1.0
    up_probability:   Optional[float] = Field(default=None)
    model_name:       Optional[str]   = Field(default=None)
    prediction_date:  Optional[str]   = Field(default=None)
    as_of_date:       Optional[str]   = Field(default=None)
    feature_snapshot: Dict[str, Any]  = Field(default_factory=dict)
    api_error:        Optional[str]   = Field(default=None)   # set if API call fails

    # ── Populated by news_searcher node ───────────────────────────────────────
    news_results: List[Dict[str, str]] = Field(default_factory=list)
    news_summary: Optional[str]        = Field(default=None)
    news_error:   Optional[str]        = Field(default=None)

    # ── Populated by analyst node ─────────────────────────────────────────────
    conflict_detected:    bool         = Field(default=False)
    conflict_explanation: Optional[str]= Field(default=None)
    full_analysis:        Optional[str]= Field(default=None)
    generated_at:         Optional[str]= Field(default=None)

    class Config:
        arbitrary_types_allowed = True


# ── Final report ──────────────────────────────────────────────────────────────

class AnalysisReport(BaseModel):
    """
    Structured output returned by run_analysis(ticker) to the Streamlit UI.

    This is the final product of the entire pipeline:
      ML prediction + news context + LLM synthesis + conflict flag.
    """
    ticker:               str            = Field(..., description="EGX ticker")
    prediction:           str            = Field(..., description="UP or DOWN")
    confidence:           float          = Field(..., description="Model confidence 0–1")
    up_probability:       float          = Field(..., description="Raw P(UP)")
    model_name:           str            = Field(..., description="ML architecture used")
    prediction_date:      str            = Field(..., description="Trading date this prediction is FOR")
    as_of_date:           str            = Field(..., description="Trading date whose features were used as input")
    news_summary:         str            = Field(..., description="Summarised recent news")
    conflict_detected:    bool           = Field(..., description="True if news contradicts model")
    conflict_explanation: Optional[str]  = Field(default=None,
                                                  description="Why conflict was flagged")
    full_analysis:        str            = Field(..., description="Full LLM analysis paragraph")
    generated_at:         str            = Field(..., description="ISO timestamp")
    feature_snapshot:     Dict[str, Any] = Field(default_factory=dict)

    def confidence_label(self) -> str:
        """Human-readable confidence tier for the UI badge."""
        if self.confidence >= 0.65:
            return "High"
        if self.confidence >= 0.55:
            return "Moderate"
        return "Low"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
