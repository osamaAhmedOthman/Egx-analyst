# EGX Analyst

A full-stack stock analysis application for the Egyptian Exchange (EGX) that combines
machine-learning price-direction predictions with LLM-powered news analysis.

The system predicts the next-day price direction (UP / DOWN) for five EGX tickers using
trained ML models, then uses an LLM agent to pull and analyze recent news for each stock,
cross-checking whether the news sentiment agrees or conflicts with the model's signal —
all surfaced through an interactive Streamlit UI.

## Covered Tickers

| Ticker    | Company                         |
|-----------|----------------------------------|
| COMI_CA   | Commercial International Bank    |
| HRHO_CA   | EFG Holding                      |
| TMGH_CA   | Talaat Moustafa Group            |
| SWDY_CA   | Elsewedy Electric                |
| ORWE_CA   | Oriental Weavers                 |

## How It Works

1. **Data pipeline** — historical OHLCV data is pulled via `yfinance` and turned into a
   set of 20 scale-invariant technical features (RSI, MACD, Bollinger Bands, ATR%, lagged
   returns, moving-average ratios, volume features, calendar effects including Ramadan).
2. **Model training** — a separate binary classifier (UP/DOWN) is trained per ticker,
   with the best-performing algorithm selected per stock via chronological
   train/validation split (test set sealed post-2024) and hyperparameter tuning with
   Optuna, tracked in MLflow.
3. **Prediction API** — a FastAPI service loads the trained models and serves next-day
   direction predictions with confidence scores.
4. **Analyst agent** — a LangGraph agent takes a model prediction, searches for and
   analyzes recent news (via Tavily) using an LLM (Groq), and produces a written analysis
   that flags whether the news supports or conflicts with the model's signal.
5. **UI** — a Streamlit app lets you run an analysis for any ticker, view the model
   signal alongside the LLM's news-based reasoning, browse sources, ask follow-up
   questions, and review session history.

## Model Selection Summary

| Ticker    | Best Model      | Notes                                      |
|-----------|-----------------|---------------------------------------------|
| COMI_CA   | XGBoost         | Threshold manually set to 0.50 (degenerate curve) |
| HRHO_CA   | XGBoost         | Required explicit `scale_pos_weight` fix     |
| TMGH_CA   | LightGBM        | Threshold manually set to 0.50 (degenerate curve) |
| SWDY_CA   | RandomForest    | Threshold manually set to 0.50 (degenerate curve) |
| ORWE_CA   | LightGBM        | Strongest model, ~0.59 validation ROC-AUC; genuinely calibrated threshold |

Confidence scores represent certainty about the *stated* direction (not the probability
of UP specifically).

## Project Structure

```
Egx-analyst/
├── agent/            # LangGraph agent: news analysis + signal-vs-news alignment
│   ├── analyst.py
│   ├── graph.py
│   ├── prompts.py
│   ├── schemas.py
│   └── tools.py
├── api/               # FastAPI prediction service
│   ├── main.py
│   ├── schemas.py
│   └── routers/predict.py
├── src/               # Core ML pipeline (shared by training + inference)
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── tune.py
│   ├── predict.py
│   └── evaluate.py
├── models/            # Trained model artifacts (.joblib) + threshold/config JSON
├── notebooks/         # EDA, feature engineering, model comparison, tuning, evaluation
├── ui/                # Streamlit UI
│   ├── app.py
│   └── pages/
│       ├── 1_analyze.py
│       └── 2_history.py
├── data/              # Raw + processed data (gitignored)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env               # API keys / secrets (gitignored, not committed)
└── requirements.txt
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Docker Compose)
- API keys for:
  - **Groq** (LLM inference for the agent)
  - **Tavily** (news search)

## Setup

1. Clone the repository and move into the project folder.

2. Create a `.env` file in the project root with your API keys:

   ```env
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

   This file is gitignored and never baked into the Docker image — it's injected at
   container runtime via `env_file` in `docker-compose.yml`.

3. Make sure trained models exist under `models/` (five `*_best_model.joblib` files plus
   `optimized_hyperparameters.json` and `winner_config.json`). These are required for the
   API to start serving predictions — see the Model Training section below if you need to
   regenerate them.

## Running the App

From the project root:

```bash
docker compose up --build
```

This builds a single shared image and starts two containers from it:

| Service | URL                          | Description                        |
|---------|-------------------------------|-------------------------------------|
| `api`   | http://localhost:8000         | FastAPI prediction service (docs at `/docs`) |
| `ui`    | http://localhost:8501         | Streamlit UI                        |

Open **http://localhost:8501** in your browser, pick a ticker on the Analyze page, and
run an analysis. The UI calls the API internally over the Docker network at
`http://api:8000` — it does not depend on anything running on your host machine outside
of Docker.

To stop:

```bash
docker compose down
```

### Running Individual Services

```bash
docker compose up --build api   # API only
docker compose up --build ui    # UI only (requires the api service to also be running)
```

## API Endpoints

| Method | Endpoint             | Description                          |
|--------|-----------------------|---------------------------------------|
| POST   | `/predict`             | Get a prediction for a single ticker (via request body) |
| GET    | `/predict/{ticker}`    | Get a prediction for a specific ticker |
| GET    | `/predict/all`         | Get predictions for all five tickers  |

Interactive Swagger docs are available at `http://localhost:8000/docs` once the API is
running.

## Environment Variables

| Variable          | Used by         | Purpose                                    |
|--------------------|-----------------|----------------------------------------------|
| `GROQ_API_KEY`      | `agent/`         | LLM calls for news analysis                  |
| `TAVILY_API_KEY`    | `agent/`         | News search                                  |
| `API_BASE_URL`      | `ui/app.py`      | Where the UI looks for the prediction API    |
| `EGX_API_URL`       | `agent/tools.py` | Where the agent looks for the prediction API |

`API_BASE_URL` and `EGX_API_URL` are both set to `http://api:8000` in
`docker-compose.yml` for the `ui` service, since the agent runs inside the Streamlit
process. They default to `http://localhost:8000` if unset, which is only correct when
running components outside Docker directly on the host.

## Model Training (Optional — Regenerating Models)

Model training happens outside the Docker workflow, via the notebooks in `notebooks/`
or the CLI:

```bash
python -m src.train --tickers COMI_CA HRHO_CA TMGH_CA SWDY_CA ORWE_CA
```

Trained models are written to `models/`, which is mounted as a volume into both
containers — so retraining doesn't require rebuilding the Docker image, just a
container restart.

## Key Engineering Notes

- **XGBoost `class_weight` silently does nothing.** `scale_pos_weight` must be set
  explicitly (`n_DOWN / n_UP`) — an earlier oversight caused a full F1 collapse on
  HRHO_CA before being caught.
- **Degenerate thresholds** on COMI_CA, TMGH_CA, and SWDY_CA (where
  `precision_recall_curve` produced a predict-all-UP threshold) were manually overridden
  to 0.50 in `optimized_hyperparameters.json`. HRHO_CA and ORWE_CA use genuinely
  calibrated thresholds.
- **Aggressive regularization** (shallow trees, high `min_child_weight`, strong L1/L2)
  is used throughout the Optuna search space to control overfitting on relatively thin
  EGX historical data.
- **Chronological splitting** is strictly enforced — the test set (post-2024) is sealed
  and never touched during experimentation or tuning.

## Known Limitations

- No automated test suite is currently included.
- Predictions reflect next-day direction only, not magnitude, and should not be treated
  as financial advice — see disclaimer below.

## Disclaimer

This project is for educational and research purposes only. Predictions and analysis
generated by this application do not constitute financial advice. Always do your own
research before making investment decisions.
