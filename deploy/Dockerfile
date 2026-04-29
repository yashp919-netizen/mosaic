FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install deps first — cached layer invalidated only on lockfile change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source
COPY src/ src/
COPY ui/ ui/
COPY data/ data/

# Pre-generate synthetic data with default seed 42 so the image is self-contained.
# CSVs are reproducible — override by mounting a volume over /app/data/synth/.
RUN uv run python data/synth/generator.py --out-dir data/synth/

EXPOSE 8501

# MOSAIC_LLM_PROVIDER controls which LLM the UI defaults to.
# "none"   = heuristics only — no Ollama or API key required (default for demo image).
# "ollama" = requires the Ollama sidecar (see docker-compose.yml).
# "gemini" = requires GEMINI_API_KEY to be set as an environment variable.
ENV MOSAIC_LLM_PROVIDER=none

CMD ["uv", "run", "streamlit", "run", "ui/streamlit_app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
