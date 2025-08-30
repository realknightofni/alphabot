# syntax=docker/dockerfile:1.7

FROM python:3.9-slim AS builder

WORKDIR /app

# Install build deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Use BuildKit cache mounts to speed up apt and pip across builds
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install requirements into a temp location
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --prefix=/install -r requirements.txt


# --- Final runtime image ---
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Runtime libraries needed by OpenCV and others
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

CMD ["python", "-u", "./alphabot/simple_bot.py"]
