# ---------------------------------------------------------------
# EGX Analyst — shared image for FastAPI API + Streamlit UI
# The actual process each container runs is chosen in docker-compose.yml
# via the `command:` override, so this one image serves both services.
# ---------------------------------------------------------------
FROM python:3.11-slim

# libgomp1 is required by xgboost/lightgbm at runtime; build-essential
# covers any package that needs compiling during pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (models/, data/ get overridden by volumes
# at runtime, but are included here too so the image is runnable standalone)
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Default command runs the API; docker-compose overrides this for the ui service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
