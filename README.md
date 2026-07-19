# EGX Analyst

A full-stack stock analysis application for the Egyptian Exchange (EGX) that combines
machine-learning price-direction predictions with LLM-powered news analysis.

The system predicts the next-day price direction (UP / DOWN) for five EGX tickers using
trained ML models, then uses an LLM agent to pull and analyze recent news for each stock,
cross-checking whether the news sentiment agrees or conflicts with the model's signal —
all surfaced through an interactive Streamlit UI with a scoped follow-up chat.

**Built as a portfolio / learning project** covering the full stack: data pipeline, ML
model training and selection, a FastAPI service, a LangGraph agent, a Streamlit UI,
Docker containerization, and a live cloud deployment.

## 🔗 Live Demo

| Service | Link |
|---|---|
| **UI** (start here) | `https://egx-analyst-ui-production.up.railway.app` |
| **API docs (Swagger)** | `https://egx-analyst-production.up.railway.app/docs` |

> Both services run on Railway's free tier. Cold starts and occasional slowness are
> expected on a free instance — this is a portfolio demo, not a production deployment.
> If a page seems stuck, give it a few seconds and refresh.

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
   direction predictions with confidence scores. Each prediction reports two distinct
   dates (see *Prediction Dates* below).
4. **Analyst agent** — a LangGraph agent takes a model prediction, searches for and
   analyzes recent news (via Tavily) using an LLM (Groq), and produces a written analysis
   in English and Arabic that flags whether the news supports or conflicts with the
   model's signal.
5. **UI** — a Streamlit app lets you run an analysis for any ticker, view the model
   signal alongside the LLM's news-based reasoning, browse sources, ask follow-up
   questions scoped to that analysis, and review session history.

## Prediction Dates

Each prediction reports two separate dates, since EGX's weekend is Friday–Saturday
(not Saturday–Sunday):

- **`as_of_date`** — the most recent trading day whose market data was used as input
  (the last available close before the prediction was made).
- **`prediction_date`** — the actual next EGX trading day being predicted *for*,
  computed by skipping forward past Friday/Saturday. This does **not** account for
  Egyptian national/religious holidays beyond the weekly weekend — a known, acceptable
  simplification for this project.

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
│   ├── analyst.py     # Public entry point — run_analysis(ticker)
│   ├── graph.py        # 3-node graph: data_fetcher → news_searcher → analyst
│   ├── prompts.py
│   ├── schemas.py      # AnalystState, AnalysisReport
│   └── tools.py        # call_prediction_api(), search_news() — Tavily/Groq rate-limit handling
├── api/               # FastAPI prediction service
│   ├── main.py
│   ├── schemas.py       # PredictionRequest/Response Pydantic models
│   └── routers/predict.py
├── src/               # Core ML pipeline (shared by training + inference)
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── tune.py
│   ├── predict.py       # Live inference — computes prediction_date / as_of_date
│   └── evaluate.py
├── models/            # Trained model artifacts (.joblib) + threshold/config JSON
├── notebooks/         # EDA, feature engineering, model comparison, tuning, evaluation
├── ui/                # Streamlit UI
│   ├── app.py
│   └── pages/
│       ├── 1_analyze.py   # Prediction + news + LLM analysis + follow-up chat
│       └── 2_history.py   # Session history of past analyses
├── data/              # Raw + processed data (gitignored)
├── Dockerfile         # Single shared image for both api and ui services
├── docker-compose.yml # Local orchestration (2 services from 1 Dockerfile)
├── .dockerignore
├── .env               # API keys / secrets (gitignored, not committed)
├── .env.example       # Template showing required env vars, no real secrets
└── requirements.txt
```

> **Note:** trained model files (`models/*.joblib`) are committed to this repo despite
> being binary artifacts, so the deployed API has something to serve predictions with.
> This is a deliberate simplification for a portfolio project — a production system
> would typically host models externally (S3, Hugging Face Hub, a model registry) and
> download them at startup instead.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Docker Compose)
- API keys for:
  - **Groq** (LLM inference for the agent) — [console.groq.com](https://console.groq.com)
  - **Tavily** (news search) — [tavily.com](https://tavily.com)

## Setup (Local)

1. Clone the repository and move into the project folder.

2. Copy `.env.example` to `.env` and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

   ```env
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here
   EGX_API_URL=http://localhost:8000
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

   This file is gitignored and never baked into the Docker image — it's injected at
   container runtime via `env_file` in `docker-compose.yml`.

3. Trained models already exist under `models/` (five `*_best_model.joblib` files plus
   `optimized_hyperparameters.json` and `winner_config.json`) — no training required to
   run the app. See *Model Training* below if you want to regenerate them.

## Running the App Locally

From the project root:

```bash
docker compose up --build
```

This builds a single shared image and starts two containers from it:

| Service | URL                          | Description                        |
|---------|-------------------------------|-------------------------------------|
| `api`   | http://localhost:8000         | FastAPI prediction service (docs at `/docs`) |
| `ui`    | http://localhost:8501         | Streamlit UI                        |

Open **http://localhost:8501**, pick a ticker on the Analyze page, and run an analysis.
The UI calls the API internally over the Docker network at `http://api:8000` — it does
not depend on anything running on your host machine outside of Docker.

To stop:

```bash
docker compose down
```

### Running Individual Services

```bash
docker compose up --build api   # API only
docker compose up --build ui    # UI only (requires the api service to also be running)
```

## Deployment (Railway)

The app is deployed as two separate Railway services from the same repository and the
same `Dockerfile`, each with a different start command:

| Service | Start Command |
|---|---|
| `egx-analyst-api` | Dockerfile default (`uvicorn api.main:app ...`) |
| `egx-analyst-ui`  | `sh -c "exec streamlit run ui/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"` |

Key things that had to be correct for this to work reliably on Railway's infrastructure:

- **Dynamic port binding** — Railway assigns a port via the `$PORT` env var, which isn't
  always the same value between deploys. The Dockerfile's `CMD` and the UI's custom
  start command both use `${PORT:-8000}` / `$PORT` with shell expansion, and the
  service's **Target Port** in Railway's Networking settings must match whatever `PORT`
  actually resolves to (this project pins `PORT=8000` explicitly as a Railway variable
  to keep it predictable).
- **Exec-form signal handling** — the Dockerfile's `CMD` uses
  `["sh", "-c", "exec uvicorn ..."]` rather than a bare shell string, so `exec` replaces
  the shell process with uvicorn as PID 1. Without `exec`, the shell stays PID 1 and
  doesn't forward OS signals correctly, which caused intermittent 502s under Railway's
  process supervision.
- **Uvicorn keep-alive timeout** — extended via `--timeout-keep-alive 75
  --proxy-headers` to stay compatible with Railway's reverse proxy connection reuse
  behavior (the default 5s timeout caused connections to drop between requests).
- **Environment variables** set per service in Railway's dashboard (not committed to
  the repo): `GROQ_API_KEY`, `TAVILY_API_KEY`, `GROQ_MODEL`, `API_BASE_URL`,
  `EGX_API_URL`, `PORT`.

## API Endpoints

| Method | Endpoint             | Description                          |
|--------|-----------------------|---------------------------------------|
| POST   | `/predict`             | Get a prediction for a single ticker (via request body) |
| GET    | `/predict/{ticker}`    | Get a prediction for a specific ticker |
| GET    | `/predict/all`         | Get predictions for all five tickers  |
| GET    | `/docs`                | Interactive Swagger UI                |

## Environment Variables

| Variable          | Used by         | Purpose                                    |
|--------------------|-----------------|----------------------------------------------|
| `GROQ_API_KEY`      | `agent/`, `ui/pages/1_analyze.py` | LLM calls for news analysis and follow-up chat |
| `GROQ_MODEL`        | `agent/`, `ui/pages/1_analyze.py` | Which Groq model to use (default `llama-3.3-70b-versatile`) |
| `TAVILY_API_KEY`    | `agent/tools.py` | News search                                  |
| `API_BASE_URL`      | `ui/app.py`      | Where the UI looks for the prediction API    |
| `EGX_API_URL`       | `agent/tools.py` | Where the agent looks for the prediction API |
| `PORT`              | Dockerfile `CMD`, UI start command | Port the container listens on — injected by the host platform (Railway) |

Locally (via Docker Compose), `API_BASE_URL` and `EGX_API_URL` are set to
`http://api:8000` for the `ui` service, since the agent runs inside the Streamlit
process. In production (Railway), both point to the API service's public HTTPS URL,
since each Railway service is a separate host reachable only over the public internet
(or Railway's private network, which this project doesn't currently use).

## Model Training (Optional — Regenerating Models)

Model training happens outside the Docker workflow, via the notebooks in `notebooks/`
or the CLI:

```bash
python -m src.train --tickers COMI_CA HRHO_CA TMGH_CA SWDY_CA ORWE_CA
```

Trained models are written to `models/`, which is mounted as a volume into both
containers locally — so retraining doesn't require rebuilding the Docker image, just a
container restart. In the deployed version, updated models require a new commit + push,
since Railway builds directly from the repository.

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
- **Rate-limit handling** — both the prediction API (`agent/tools.py`) and the Groq LLM
  calls (`ui/pages/1_analyze.py`) detect 429/rate-limit responses specifically and
  degrade gracefully with a clear message, rather than surfacing a raw stack trace to
  the user.
- **Two distinct dates per prediction** (`as_of_date` vs `prediction_date`) — see
  *Prediction Dates* above.

## Known Limitations

- No automated test suite is currently included.
- Predictions reflect next-day direction only, not magnitude, and should not be treated
  as financial advice — see disclaimer below.
- `prediction_date` accounts for EGX's weekly Friday–Saturday weekend but not Egyptian
  national/religious holidays.
- Occasional non-Arabic characters can appear in the Arabic analysis section — a known
  quirk of the underlying LLM's multilingual generation, not a bug in this codebase.
- Free-tier hosting (Railway) means the live demo may be slower under load or after
  periods of inactivity.

## Disclaimer

This project is for educational and research purposes only. Predictions and analysis
generated by this application do not constitute financial advice. Always do your own
research before making investment decisions.
