"""
agent/analyst.py — Public entry point for the EGX analyst agent.

Single public function: run_analysis(ticker) → AnalysisReport

This is what the Streamlit UI and any external caller imports.
It hides all LangGraph internals behind a clean interface.

Usage
-----
    from agent.analyst import run_analysis
    report = run_analysis("ORWE_CA")
    print(report.direction, report.confidence, report.conflict_detected)

CLI
---
    python -m agent.analyst --ticker ORWE_CA
    python -m agent.analyst --all
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import List, Optional

from agent.graph import graph
from agent.schemas import AnalysisReport, AnalystState
from src.config import TICKERS

logger = logging.getLogger(__name__)


def run_analysis(ticker: str) -> AnalysisReport:
    """
    Run the full EGX analyst pipeline for one ticker.

    Pipeline:
      1. data_fetcher node  — fetch ML prediction from FastAPI
      2. news_searcher node — search recent news via Tavily
      3. analyst node       — synthesise with Groq LLM, detect conflicts

    Parameters
    ----------
    ticker : EGX ticker in project format, e.g. "ORWE_CA"

    Returns
    -------
    AnalysisReport with all fields populated.

    Raises
    ------
    ValueError : if ticker is not in the configured TICKERS list.
    """
    ticker = ticker.upper().strip()
    if ticker not in TICKERS:
        raise ValueError(
            f"Unknown ticker: '{ticker}'. "
            f"Supported tickers: {TICKERS}"
        )

    logger.info(f"Starting analysis pipeline for {ticker}")

    # ── Initialise state ──────────────────────────────────────────────────────
    initial_state = AnalystState(ticker=ticker)

    # ── Run the graph ─────────────────────────────────────────────────────────
    # graph.invoke() runs all three nodes sequentially and returns the
    # final merged state as a dict.
    try:
        final_state_dict = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception(f"Graph execution failed for {ticker}: {exc}")
        # Return a degraded report rather than crashing the UI
        return AnalysisReport(
            ticker               = ticker,
            prediction           = "UNKNOWN",
            confidence           = 0.0,
            up_probability       = 0.0,
            model_name           = "Unknown",
            prediction_date      = "Unknown",
            as_of_date           = "Unknown",
            news_summary         = "News search could not be completed.",
            conflict_detected    = False,
            conflict_explanation = None,
            full_analysis        = f"Analysis pipeline failed: {exc}",
            generated_at         = __import__("datetime").datetime.now().isoformat(),
        )

    # ── Extract final state ───────────────────────────────────────────────────
    # LangGraph returns either a dict or the state object depending on version
    if isinstance(final_state_dict, dict):
        state = AnalystState(**final_state_dict)
    else:
        state = final_state_dict

    logger.info(
        f"Analysis complete for {ticker} — "
        f"direction={state.prediction}, "
        f"conflict={state.conflict_detected}"
    )

    # ── Build report ──────────────────────────────────────────────────────────
    report = AnalysisReport(
        ticker               = ticker,
        prediction           = state.prediction or "UNKNOWN",
        confidence           = state.confidence or 0.0,
        up_probability       = state.up_probability or 0.0,
        model_name           = state.model_name or "Unknown",
        prediction_date      = state.prediction_date or "Unknown",
        as_of_date           = state.as_of_date or "Unknown",
        news_summary         = state.news_summary or "No news available.",
        conflict_detected    = state.conflict_detected,
        conflict_explanation = state.conflict_explanation,
        full_analysis        = state.full_analysis or "Analysis not available.",
        generated_at         = state.generated_at or __import__("datetime").datetime.now().isoformat(),
        feature_snapshot     = state.feature_snapshot,
    )

    return report


def run_analysis_all() -> List[AnalysisReport]:
    """
    Run analysis for all 5 tickers sequentially.
    Failures on one ticker do not block the others.
    """
    reports = []
    for ticker in TICKERS:
        try:
            logger.info(f"Running analysis for {ticker} ...")
            report = run_analysis(ticker)
            reports.append(report)
            print(
                f"  {ticker:<12} {report.prediction:<5} "
                f"confidence={report.confidence:.3f}  "
                f"conflict={'⚠ YES' if report.conflict_detected else 'NO'}"
            )
        except Exception as exc:
            logger.error(f"Analysis failed for {ticker}: {exc}")
            print(f"  {ticker:<12} ✗ {exc}")
    return reports


def _print_report(report: AnalysisReport) -> None:
    """Pretty-print an AnalysisReport to the terminal."""
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  EGX ANALYST REPORT — {report.ticker}")
    print(sep)
    print(f"  Prediction      : {report.prediction}")
    print(f"  Confidence      : {report.confidence*100:.1f}%  ({report.confidence_label()})")
    print(f"  P(UP)           : {report.up_probability*100:.1f}%")
    print(f"  Model           : {report.model_name}")
    print(f"  Prediction Date : {report.prediction_date}")
    print(f"  As-Of Date      : {report.as_of_date}")
    print(f"  Generated At    : {report.generated_at}")
    print(f"\n  Conflict Detected : {'⚠ YES' if report.conflict_detected else '✓ NO'}")
    if report.conflict_detected and report.conflict_explanation:
        print(f"  Conflict Reason   : {report.conflict_explanation}")
    print(f"\n{'─'*65}")
    print("  FULL ANALYSIS")
    print(f"{'─'*65}")
    print(report.full_analysis)
    print(f"\n{'─'*65}")
    print("  NEWS SUMMARY")
    print(f"{'─'*65}")
    print(report.news_summary)
    print(sep)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run EGX analyst pipeline for one or all tickers."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ticker",
        choices=TICKERS,
        help="Single ticker to analyse.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run analysis for all 5 tickers.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted report.",
    )
    args = parser.parse_args()

    if args.all:
        reports = run_analysis_all()
        if args.json:
            print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        report = run_analysis(args.ticker)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            _print_report(report)
