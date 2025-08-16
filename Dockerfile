# Multi-stage Dockerfile for Email Engagement Prediction ML Pipeline
# Stage 1: Base Python environment with dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development environment
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/models data/logs data/quality data/performance data/cicd

# Set permissions
RUN chmod +x src/*.py

# Expose port for API service
EXPOSE 8000

# Default command for development
CMD ["python", "src/api_service.py", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production environment
FROM base as production

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/models data/logs data/quality data/performance data/cicd

# Set permissions
RUN chmod +x src/*.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app
USER mluser

# Expose port for API service
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "src/api_service.py", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Jupyter environment for analysis
FROM base as jupyter

# Install Jupyter and additional analysis tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    plotly \
    dash \
    streamlit

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/models data/logs data/quality data/performance data/cicd notebooks

# Set permissions
RUN chmod +x src/*.py

# Expose Jupyter port
EXPOSE 8888

# Default command for Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
