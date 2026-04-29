FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY data/ data/

# Generate synthetic data (seed 42, deterministic)
RUN python data/synth/generator.py --out-dir data/synth/

# HuggingFace Docker Spaces use port 7860
EXPOSE 7860

ENV MOSAIC_LLM_PROVIDER=none
ENV PYTHONPATH=/app/src

COPY app.py ui/app.py

CMD ["streamlit", "run", "ui/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "7860", \
     "--server.headless", "true"]
