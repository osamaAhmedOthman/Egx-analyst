"""
api/routers/predict.py — Prediction endpoints.

Routes
------
POST /predict          → single ticker prediction
GET  /predict/all      → all 5 tickers in one call
GET  /predict/{ticker} → single ticker via URL param (convenient for testing)
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from api.schemas import ErrorResponse, PredictionRequest, PredictionResponse
from src.config import TICKERS
from src.predict import PredictionResult, predict, predict_all

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Predictions"],
)


# ── Internal helper ───────────────────────────────────────────────────────────

def _result_to_response(result: PredictionResult) -> PredictionResponse:
    """Convert the src.predict.PredictionResult dataclass to a Pydantic response model."""
    return PredictionResponse(
        ticker           = result.ticker,
        direction        = result.direction,
        confidence       = round(result.confidence, 4),
        up_probability   = round(result.up_probability, 4),
        threshold        = round(result.threshold, 4),
        prediction_date  = result.prediction_date,
        as_of_date       = result.as_of_date,
        model_name       = result.model_name,
        feature_snapshot = {
            k: round(float(v), 6) if v is not None else None
            for k, v in result.feature_snapshot.items()
        },
    )


def _handle_predict_error(ticker: str, exc: Exception) -> HTTPException:
    """
    Map internal exceptions to appropriate HTTP status codes.

    FileNotFoundError → 503 Service Unavailable (model not trained yet)
    ValueError        → 422 Unprocessable Entity (bad ticker)
    RuntimeError      → 503 Service Unavailable (data fetch failed)
    Everything else   → 500 Internal Server Error
    """
    if isinstance(exc, FileNotFoundError):
        logger.error(f"Model artifact missing for {ticker}: {exc}")
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model for {ticker} is not available. Run 'python -m src.train' first.",
        )
    if isinstance(exc, ValueError):
        logger.warning(f"Invalid request for {ticker}: {exc}")
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    if isinstance(exc, RuntimeError):
        logger.error(f"Data fetch failed for {ticker}: {exc}")
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch market data for {ticker}. {exc}",
        )
    logger.exception(f"Unexpected error predicting {ticker}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Unexpected error generating prediction for {ticker}.",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict next-day direction for one ticker",
    description=(
        "Fetches live market data, runs the feature engineering pipeline, "
        "and returns a directional prediction (UP/DOWN) with confidence score "
        "for the specified EGX ticker."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid ticker"},
        503: {"model": ErrorResponse, "description": "Model not trained or data unavailable"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
    },
)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """
    **POST /predict**

    Request body:
    ```json
    { "ticker": "ORWE_CA" }
    ```

    Returns the full prediction including confidence, raw UP probability,
    the decision threshold applied, and the 20-feature snapshot used as input.
    """
    logger.info(f"Prediction requested for {request.ticker}")
    try:
        result = predict(request.ticker)
        logger.info(
            f"{request.ticker} → {result.direction} "
            f"(confidence={result.confidence:.3f}, P(UP)={result.up_probability:.3f})"
        )
        return _result_to_response(result)
    except Exception as exc:
        raise _handle_predict_error(request.ticker, exc)


@router.get(
    "/all",
    response_model=List[PredictionResponse],
    summary="Predict next-day direction for all 5 tickers",
    description=(
        "Runs predictions for all configured EGX tickers in sequence. "
        "Tickers that fail (e.g. data unavailable) are silently skipped — "
        "the response contains only successful predictions."
    ),
)
async def predict_all_tickers() -> List[PredictionResponse]:
    """
    **GET /predict/all**

    Convenience endpoint for the Streamlit UI home page and agent warm-up.
    Returns a list of PredictionResponse objects, one per successful ticker.
    """
    logger.info("Bulk prediction requested for all tickers")
    results = predict_all()   # failures are logged inside, not raised
    responses = []
    for result in results:
        try:
            responses.append(_result_to_response(result))
        except Exception as exc:
            logger.warning(f"Could not serialise result for {result.ticker}: {exc}")
    logger.info(f"Bulk prediction complete: {len(responses)}/{len(TICKERS)} succeeded")
    return responses


@router.get(
    "/{ticker}",
    response_model=PredictionResponse,
    summary="Predict next-day direction via URL parameter",
    description=(
        "Same as POST /predict but with the ticker in the URL path. "
        "Useful for quick browser testing and curl commands."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid ticker"},
        503: {"model": ErrorResponse, "description": "Model not trained or data unavailable"},
    },
)
async def predict_by_url(ticker: str) -> PredictionResponse:
    """
    **GET /predict/{ticker}**

    Example: `GET /predict/ORWE_CA`

    Identical output to POST /predict — just a more convenient
    URL format for browser testing and curl.
    """
    ticker = ticker.upper().strip()
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{ticker}' is not a supported ticker. Valid: {TICKERS}",
        )
    logger.info(f"URL-based prediction requested for {ticker}")
    try:
        result = predict(ticker)
        return _result_to_response(result)
    except Exception as exc:
        raise _handle_predict_error(ticker, exc)
