"""
api/schemas.py — Pydantic models for the EGX Analyst API.

Defines the exact shape of every request and response.
FastAPI uses these for automatic validation, serialisation,
and OpenAPI documentation generation.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from src.config import TICKERS


# ── Request ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Body of POST /predict."""

    ticker: str = Field(
        ...,
        description="EGX ticker in project format, e.g. 'COMI_CA'.",
        examples=["COMI_CA", "ORWE_CA"],
    )

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_valid(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in TICKERS:
            raise ValueError(
                f"'{v}' is not a supported ticker. "
                f"Valid options: {TICKERS}"
            )
        return v


# ── Response ──────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Full prediction response returned by POST /predict.

    Fields
    ------
    ticker          : EGX ticker string, e.g. "COMI_CA"
    direction       : "UP" or "DOWN" — predicted next-day price direction
    confidence      : probability of the predicted direction (0.0–1.0)
                      Always represents certainty about the stated direction:
                        - direction=UP   → confidence = P(UP)
                        - direction=DOWN → confidence = 1 - P(UP)
    up_probability  : raw model output — P(next day closes higher)
    threshold       : decision boundary applied for this ticker
    prediction_date : the trading date whose features were used as input
                      (last available market date, not necessarily today)
    model_name      : ML architecture that produced this prediction
    feature_snapshot: the 20 feature values fed to the model — useful for
                      the agent layer to reason about what drove the signal
    """

    ticker:           str                   = Field(..., description="EGX ticker")
    direction:        Literal["UP", "DOWN"] = Field(..., description="Predicted direction")
    confidence:       float                 = Field(..., ge=0.0, le=1.0,
                                                    description="Certainty about stated direction")
    up_probability:   float                 = Field(..., ge=0.0, le=1.0,
                                                    description="Raw P(UP) from model")
    threshold:        float                 = Field(..., description="Decision boundary used")
    prediction_date:  str                   = Field(..., description="Date of input features")
    model_name:       str                   = Field(..., description="Model architecture")
    feature_snapshot: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Feature values used for this prediction"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "ticker":         "ORWE_CA",
            "direction":      "UP",
            "confidence":     0.5712,
            "up_probability": 0.5712,
            "threshold":      0.4743,
            "prediction_date":"2026-07-09",
            "model_name":     "LightGBM",
            "feature_snapshot": {
                "RSI_14": 48.3,
                "MACD_Norm": 0.0012,
                "Return_Lag_1": 0.008,
            },
        }
    }}


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status:  str       = Field(default="ok")
    tickers: list[str] = Field(default_factory=list)
    models_loaded: bool = Field(default=False)


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standardised error body returned on 4xx / 5xx responses."""
    error:   str = Field(..., description="Short error type")
    detail:  str = Field(..., description="Human-readable explanation")
    ticker:  Optional[str] = Field(default=None)
