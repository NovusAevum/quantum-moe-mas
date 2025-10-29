# Multi-stage Dockerfile for Quantum MoE MAS
# Optimized for production deployment with security best practices

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Set work directory
WORKDIR /app

# Copy requirements
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/quantum/.local/bin:$PATH" \
    ENVIRONMENT=production \
    DEBUG=false

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r quantum && useradd -r -g quantum quantum

# Create directories
RUN mkdir -p /app /app/logs /app/data && \
    chown -R quantum:quantum /app

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=quantum:quantum src/ ./src/
COPY --chown=quantum:quantum config/ ./config/
COPY --chown=quantum:quantum pyproject.toml ./
COPY --chown=quantum:quantum README.md ./

# Install the package in development mode
USER quantum
RUN pip install --user -e .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["quantum-moe-mas", "start-api", "--host", "0.0.0.0", "--port", "8000"]