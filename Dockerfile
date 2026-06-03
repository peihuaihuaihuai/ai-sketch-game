# Hugging Face Spaces — Docker SDK
# Deploys the Flask + PyTorch QuickDraw sketch recognition app as a Docker container.
#
# Push to HF Spaces and it auto-builds: huggingface.co/spaces/<your-username>/<space-name>

FROM python:3.10-slim

# -------------------------------------------------------------------
# System dependencies
# -------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Python dependencies (installed before app code for layer caching)
# -------------------------------------------------------------------

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------------
# Application code
# -------------------------------------------------------------------

COPY . .

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------

# HF Spaces reserves port 7860; also respects $PORT if set differently
EXPOSE 7860

# Single worker — model ~4MB, inference <5ms on CPU, memory ~300MB total
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 120 --access-logfile -"]
