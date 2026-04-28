# Stub — implementation on Day 13
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["uvicorn", "src.mosaic.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
