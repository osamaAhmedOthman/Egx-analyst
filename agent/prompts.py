"""
agent/prompts.py — LLM prompt templates for the analyst node.

One prompt: ANALYST_PROMPT
  Instructs the LLM to synthesise the ML prediction with live news,
  detect conflicts, and produce a structured bilingual analysis.
"""
from __future__ import annotations


ANALYST_SYSTEM_PROMPT = """You are EGX Analyst, a professional financial analyst specialising in the Egyptian Exchange (EGX). You combine quantitative ML model signals with qualitative news analysis to produce actionable, well-reasoned market insights.

Your analysis must always be:
- Grounded in the specific data provided (prediction + news)
- Honest about uncertainty — never overstate model confidence
- Aware of EGX-specific context: currency dynamics (EGP), macro regime shifts, Ramadan seasonality, political events
- Professional and concise — no filler, no generic disclaimers
"""


ANALYST_USER_PROMPT = """You have received the following inputs for {ticker} ({company_name}):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML MODEL PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ticker          : {ticker}
Direction       : {direction}
Confidence      : {confidence_pct}% (certainty about the stated direction)
P(UP)           : {up_probability_pct}%
Model           : {model_name}
Prediction Date : {prediction_date}
Confidence Tier : {confidence_label}

Key technical signals from the feature snapshot:
{feature_highlights}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECENT NEWS (from Tavily search)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{news_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce a structured analysis following this EXACT format:

CONFLICT_DETECTED: [YES or NO]

CONFLICT_EXPLANATION:
[If YES: explain specifically why the news contradicts the model signal. 
 Be precise — quote the specific news element that conflicts.
 If NO: write "None"]

ANALYSIS:
[3–4 paragraph professional analysis in English covering:
 1. What the model signal means in context (not just repeating the number)
 2. How the news aligns with or challenges the prediction
 3. Key risk factors specific to this stock and current EGX conditions
 4. A concise concluding view — should an analyst weight the model or the news more heavily today, and why?]

تحليل:
[1–2 paragraph Arabic summary of the key points above.
 Write naturally — do not mechanically translate the English.
 Focus on the most important insight for an Arabic-speaking investor.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFLICT DETECTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Flag CONFLICT_DETECTED: YES if ANY of these are true:
- Model predicts UP but news contains: earnings miss, regulatory action,
  management scandal, credit downgrade, sector headwinds, macro shock
- Model predicts DOWN but news contains: strong earnings beat, major contract win,
  dividend announcement, acquisition at premium, sector tailwind
- Confidence is LOW (< 55%) AND news sentiment strongly contradicts direction

Do NOT flag conflict for:
- Vague or neutral news with no clear directional implication
- Historical news older than 2 weeks with no current relevance
- News about unrelated EGX companies or the broader index
"""


def build_feature_highlights(feature_snapshot: dict) -> str:
    """
    Convert the feature snapshot dict into readable bullet points
    for the analyst prompt. Selects the most interpretable features.
    """
    if not feature_snapshot:
        return "  No feature snapshot available."

    def fmt(val) -> str:
        if val is None:
            return "N/A"
        try:
            return f"{float(val):.4f}"
        except (TypeError, ValueError):
            return str(val)

    highlights = []

    # RSI — overbought/oversold signal
    # oversold -> good time to buy
    # overbought -> good time to sell
    rsi = feature_snapshot.get("RSI_14")
    if rsi is not None:
        rsi_f = float(rsi)
        rsi_label = "overbought" if rsi_f > 70 else ("oversold" if rsi_f < 30 else "neutral")
        highlights.append(f"  RSI(14)             = {rsi_f:.1f}  [{rsi_label}]")

    # MACD histogram — momentum direction
    # Bullish momentum -> price likely to rise , advantage for who bought before at low price
    # Bearish momentum -> price likely to fall, advantage for who sold before at high price
    macd_hist = feature_snapshot.get("MACD_Hist_Norm")
    if macd_hist is not None:
        direction = "bullish momentum" if float(macd_hist) > 0 else "bearish momentum"
        highlights.append(f"  MACD Histogram      = {fmt(macd_hist)}  [{direction}]")

    # Bollinger position — where price sits in the band
    # Upper breakout -> price likely to continue rising, advantage for who bought before at low price
    # Lower band -> price likely to continue falling, advantage for who sold before at high price
    bb_pos = feature_snapshot.get("Bollinger_Position")
    if bb_pos is not None:
        bb_f = float(bb_pos)
        bb_label = "upper breakout" if bb_f > 1 else ("lower band" if bb_f < 0.2 else "mid-band")
        highlights.append(f"  Bollinger Position  = {bb_f:.3f}  [{bb_label}]")

    # 1-day return — yesterday's move
    r1 = feature_snapshot.get("Return_Lag_1")
    if r1 is not None:
        highlights.append(f"  Yesterday's Return  = {float(r1)*100:.2f}%")

    # 5-day return — weekly momentum
    r5 = feature_snapshot.get("Return_Lag_5")
    if r5 is not None:
        highlights.append(f"  5-Day Return        = {float(r5)*100:.2f}%")

    # Volume ratio — participation
    # high volume -> strong participation, price movment reliable
    # low volume -> weak participation, price movement may be noisy

    vol_ratio = feature_snapshot.get("Volume_Ratio")
    if vol_ratio is not None:
        vr = float(vol_ratio)
        vol_label = "high volume" if vr > 1.5 else ("low volume" if vr < 0.7 else "normal volume")
        highlights.append(f"  Volume Ratio        = {vr:.2f}x 21-day avg  [{vol_label}]")

    # MA ratio — trend extension
    # above MA21 -> price likely to continue rising, advantage for who bought before at low price
    # below MA21 -> price likely to continue falling, advantage for who sold before at high price
    ma21 = feature_snapshot.get("Close_MA21_Ratio")
    if ma21 is not None:
        ma_f = float(ma21)
        ma_label = "above 21-day MA" if ma_f > 1 else "below 21-day MA"
        highlights.append(f"  Price vs MA21       = {ma_f:.4f}  [{ma_label}]")

    return "\n".join(highlights) if highlights else "  Feature data not available."
