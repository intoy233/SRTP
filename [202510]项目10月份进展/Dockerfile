# Bridge VIV Risk Assessment System - Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Copy data (if needed for demo)
COPY bridge_dataset_fixed.csv ./data/bridge_dataset_fixed.csv

# Create necessary directories
RUN mkdir -p experiments results logs

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command - run help
CMD ["python", "src/train.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); import utils; print('OK')" || exit 1

# Labels
LABEL maintainer="Bridge VIV Research Team"
LABEL description="Bridge Vortex-Induced Vibration Risk Assessment System"
LABEL version="1.0.0"

# Expose port if needed for future web interface
EXPOSE 8080

# Alternative entry points
# Training: docker run --rm -v $(pwd)/data:/app/data bridge-viv python src/train.py --config config/config.yaml
# Prediction: docker run --rm -v $(pwd)/data:/app/data bridge-viv python src/predict.py --model experiments/best_model.pkl --input data/new_bridges.csv --output data/predictions.csv
# Experiments: docker run --rm -v $(pwd)/experiments:/app/experiments bridge-viv python src/experiments.py --config config/config.yaml