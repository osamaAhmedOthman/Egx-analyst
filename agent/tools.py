"""
agent/tools.py — Tool functions called by LangGraph nodes.

Two tools:
  call_prediction_api(ticker)  → hits the FastAPI /predict endpoint
  search_news(ticker)          → queries Tavily for recent news
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("EGX_API_URL", "http://localhost:8000")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
REQUEST_TIMEOUT: float = 30.0   # seconds

# Company names for news search — both English and Arabic for better coverage
COMPANY_NAMES: Dict[str, Dict[str, str]] = {
    "COMI_CA": {
        "en": "Commercial International Bank Egypt CIB",
        "ar": "البنك التجاري الدولي مصر",
        "short": "CIB Egypt",
    },
    "HRHO_CA": {
        "en": "EFG Holding Egypt investment bank",
        "ar": "EFG هيرميس القابضة مصر",
        "short": "EFG Holding",
    },
    "TMGH_CA": {
        "en": "Talaat Moustafa Group Egypt real estate",
        "ar": "مجموعة طلعت مصطفى العقارية",
        "short": "Talaat Moustafa",
    },
    "SWDY_CA": {
        "en": "Elsewedy Electric Egypt cables energy",
        "ar": "السويدي إليكتريك مصر",
        "short": "Elsewedy Electric",
    },
    "ORWE_CA": {
        "en": "Oriental Weavers Egypt carpets rugs",
        "ar": "الشرقية للسجاد مصر",         
        "short": "Oriental Weavers",
    },
}


# ── Tool 1: Prediction API ────────────────────────────────────────────────────

def call_prediction_api(ticker: str) -> Dict[str, Any]:
    """
    Call the FastAPI /predict endpoint for one ticker.

    Returns the full prediction dict on success, or a dict with
    an 'error' key on failure — so the graph node can handle
    failures gracefully without crashing the entire pipeline.

    Parameters
    ----------
    ticker : EGX ticker in project format, e.g. "COMI_CA"

    Returns
    -------
    dict with keys: ticker, direction, confidence, up_probability,
                    threshold, prediction_date, model_name, feature_snapshot
                    (or 'error' key on failure)
    """
    url = f"{API_BASE_URL}/predict"
    logger.info(f"Calling prediction API for {ticker}: {url}")

    try:
        response = httpx.post(
            url,
            json={"ticker": ticker},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        logger.info(
            f"API response for {ticker}: "
            f"direction={data.get('direction')} "
            f"confidence={data.get('confidence')}"
        )
        return data

    except httpx.ConnectError:
        msg = (
            f"Cannot connect to prediction API at {API_BASE_URL}. "
            "Ensure 'uvicorn api.main:app --port 8000' is running."
        )
        logger.error(msg)
        return {"error": msg, "ticker": ticker}

    except httpx.TimeoutException:
        msg = f"Prediction API timed out after {REQUEST_TIMEOUT}s for {ticker}."
        logger.error(msg)
        return {"error": msg, "ticker": ticker}

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            msg = f"Prediction API rate limit reached for {ticker}. Please try again in a moment."
            logger.warning(msg)
            return {"error": msg, "ticker": ticker, "rate_limited": True}

        msg = f"Prediction API returned HTTP {exc.response.status_code} for {ticker}: {exc.response.text}"
        logger.error(msg)
        return {"error": msg, "ticker": ticker}

    except Exception as exc:
        msg = f"Unexpected error calling prediction API for {ticker}: {exc}"
        logger.exception(msg)
        return {"error": msg, "ticker": ticker}


# ── Tool 2: News Search ───────────────────────────────────────────────────────

def search_news(ticker: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for recent news about a company using Tavily.

    Runs two queries — English and Arabic — to maximise coverage of
    EGX-relevant news, which appears in both languages.

    Parameters
    ----------
    ticker     : EGX ticker in project format, e.g. "COMI_CA"
    max_results: number of results per query (total up to 2x this)

    Returns
    -------
    List of dicts, each with keys: title, url, content, published_date
    Returns empty list on failure (never raises) so the graph continues.
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — skipping news search")
        return []

    company = COMPANY_NAMES.get(ticker, {})
    if not company:
        logger.warning(f"No company name mapping found for {ticker}")
        return []

    client   = TavilyClient(api_key=TAVILY_API_KEY)
    results  = []
    seen_urls = set()

    queries = [
        f"{company['en']} stock news 2026",
        f"{company['ar']} اخبار البورصة",
    ]

    rate_limited = False

    for query in queries:
        try:
            logger.info(f"Tavily search: '{query}'")
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
            )

            for result in response.get("results", []):
                url = result.get("url", "")
                if url in seen_urls:
                    continue        # deduplicate across queries
                seen_urls.add(url)
                results.append({
                    "title":          result.get("title", ""),
                    "url":            url,
                    "content":        result.get("content", "")[:500],  # truncate
                    "published_date": result.get("published_date", ""),
                })

        except Exception as exc:
            # Tavily's client raises plain exceptions without a stable type,
            # so detect rate limiting from the status code / message text.
            err_text = str(exc).lower()
            if "429" in err_text or "rate limit" in err_text or "usage limit" in err_text:
                logger.warning(f"Tavily rate limit reached on query '{query}': {exc}")
                rate_limited = True
                break   # no point retrying the second query if the quota's gone
            logger.warning(f"Tavily search failed for query '{query}': {exc}")
            continue    # try next query before giving up

    if rate_limited and not results:
        logger.warning(f"News search rate-limited for {ticker} — no articles retrieved")
        return [{"rate_limited": True}]

    logger.info(f"News search complete for {ticker}: {len(results)} articles found")
    return results


def format_news_for_prompt(news_results: List[Dict[str, str]]) -> str:
    """
    Format raw Tavily results into a clean text block for the LLM prompt.
    Truncates content and adds source attribution.
    """
    if not news_results:
        return "No recent news articles found for this company."

    if len(news_results) == 1 and news_results[0].get("rate_limited"):
        return (
            "News search is temporarily unavailable (rate limit reached). "
            "This analysis is based on the model prediction only, without recent news context."
        )

    lines = []
    for i, article in enumerate(news_results, 1):
        title   = article.get("title", "Untitled")
        content = article.get("content", "").strip()
        url     = article.get("url", "")
        date    = article.get("published_date", "")
        date_str = f" ({date})" if date else ""

        lines.append(f"[{i}] {title}{date_str}")
        if content:
            lines.append(f"    {content}")
        if url:
            lines.append(f"    Source: {url}")
        lines.append("")

    return "\n".join(lines).strip()
