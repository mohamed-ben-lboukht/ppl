# Professional Multi-stage Dockerfile for Keystroke Analytics Application
# Optimized for security, performance, and minimal image size

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add labels for better organization
LABEL maintainer="Keystroke Analytics Team" \
      description="Professional keystroke timing analysis and user profiling system" \
      version=${VERSION} \
      build-date=${BUILD_DATE} \
      vcs-ref=${VCS_REF}

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directory
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app && chown appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    FLASK_APP=app.py \
    PORT=5000 \
    WORKERS=4

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user and directory
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app /app/logs /app/data /app/models && \
    chown -R appuser:appuser /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories and set permissions
RUN chmod +x /app/*.py && \
    chmod -R 755 /app/static && \
    chmod -R 755 /app/templates && \
    chmod -R 755 /app/models && \
    chmod -R 755 /app/data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Initialize database if needed\n\
python -c "from app import create_app; from models.database import db; app = create_app(); app.app_context().push(); db.create_all()"\n\
\n\
# Start the application\n\
if [ "$FLASK_ENV" = "production" ]; then\n\
    exec gunicorn --bind 0.0.0.0:$PORT --workers $WORKERS --worker-class eventlet --timeout 120 --max-requests 1000 --preload app:app\n\
else\n\
    exec python app.py\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# Development stage (optional)
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    debugpy

# Set development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=1

# Switch back to app user
USER appuser

# Override command for development
CMD ["python", "app.py"] 