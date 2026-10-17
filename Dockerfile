FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CPU-only torch for the oracle test; optional at runtime (test skips w/o it).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || true

COPY . .

# Default: full verification (tests + all experiments + report).
CMD ["sh", "-c", "python -m pytest tests/ -q && python experiments/run_all.py"]
