FROM python:3.13-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run as unbuffered
ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/run_ingestion.py", "--interval", "30"]
