FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "-m", "server.app"]
