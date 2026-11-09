FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env.example .env

ENV AWS_REGION=ap-south-1
ENV S3_BUCKET=customer-segmentation-2026

# ── FIX: Add /app/src to PYTHONPATH so imports work from any working directory
ENV PYTHONPATH=/app/src

EXPOSE 8501
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0"]
