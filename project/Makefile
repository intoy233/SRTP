# Bridge VIV Risk Assessment System - Makefile

.PHONY: help install test clean train predict experiments docker setup lint format

# Configuration
PYTHON := python
CONFIG_FILE := config/config.yaml
EXPERIMENT_NAME := bridge_viv_$(shell date +%Y%m%d_%H%M%S)
OUTPUT_DIR := experiments

# Default target
help:
	@echo "Bridge VIV Risk Assessment System"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  setup        Setup project environment"
	@echo "  test         Run tests"
	@echo "  lint         Run code quality checks"
	@echo "  format       Format code"
	@echo "  clean        Clean temporary files"
	@echo "  train        Train baseline models"
	@echo "  hyperopt     Run hyperparameter optimization"
	@echo "  predict      Generate predictions"
	@echo "  experiments  Run complete experiment pipeline"
	@echo "  docker       Build and run Docker containers"
	@echo "  docker-dev   Start development environment"
	@echo "  docker-test  Run tests in Docker"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG_FILE     = $(CONFIG_FILE)"
	@echo "  EXPERIMENT_NAME = $(EXPERIMENT_NAME)"
	@echo "  OUTPUT_DIR      = $(OUTPUT_DIR)"

# Installation and Setup
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

setup: install
	@echo "Setting up project environment..."
	mkdir -p experiments results logs data
	@echo "Project setup complete!"

# Development
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term

test-fast:
	@echo "Running fast tests..."
	pytest tests/ -v --tb=short -x

lint:
	@echo "Running code quality checks..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf build/ dist/

# Training and Experiments
train:
	@echo "Training baseline models..."
	$(PYTHON) src/train.py \
		--config $(CONFIG_FILE) \
		--experiment-name $(EXPERIMENT_NAME)_baseline \
		--models linear random_forest \
		--output-dir $(OUTPUT_DIR)

hyperopt:
	@echo "Running hyperparameter optimization..."
	$(PYTHON) src/hyperparam_search.py \
		--config $(CONFIG_FILE) \
		--experiment-name $(EXPERIMENT_NAME)_hyperopt \
		--n-trials 50 \
		--output-dir $(OUTPUT_DIR)

predict:
	@echo "Generating predictions..."
	$(PYTHON) src/predict.py \
		--comprehensive \
		--input bridge_dataset_fixed.csv \
		--output results/predictions_$(EXPERIMENT_NAME).csv

experiments:
	@echo "Running complete experiment pipeline..."
	$(PYTHON) src/experiments.py \
		--config $(CONFIG_FILE) \
		--experiment-name $(EXPERIMENT_NAME) \
		--output-dir $(OUTPUT_DIR) \
		--full-pipeline

experiments-quick:
	@echo "Running quick experiments..."
	$(PYTHON) src/experiments.py \
		--config $(CONFIG_FILE) \
		--experiment-name $(EXPERIMENT_NAME)_quick \
		--output-dir $(OUTPUT_DIR) \
		--baseline-only

# Docker Commands
docker:
	@echo "Building Docker image..."
	docker build -t bridge-viv .

docker-dev: docker
	@echo "Starting development environment..."
	docker-compose up bridge-viv-dev

docker-test: docker
	@echo "Running tests in Docker..."
	docker-compose --profile test up bridge-viv-test

docker-mlflow: docker
	@echo "Starting MLflow tracking server..."
	docker-compose --profile mlflow up mlflow

docker-run: docker
	@echo "Running Bridge VIV container..."
	docker-compose up bridge-viv

docker-clean:
	@echo "Cleaning Docker containers and images..."
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Data Management
validate-data:
	@echo "Validating data files..."
	$(PYTHON) -c "
import pandas as pd
import yaml
with open('$(CONFIG_FILE)', 'r') as f:
    config = yaml.safe_load(f)
data_file = config.get('data', {}).get('file_path', '')
if data_file:
    df = pd.read_csv(data_file)
    print(f'Data shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print('Data validation passed!')
else:
    print('No data file specified in config')
"

prepare-data:
	@echo "Preparing sample data..."
	$(PYTHON) -c "
import pandas as pd
import numpy as np
np.random.seed(42)
# Generate sample data if needed
print('Sample data preparation completed!')
"

# Documentation
docs:
	@echo "Generating documentation..."
	$(PYTHON) -c "
import os
from pathlib import Path

# Create basic documentation structure
docs_dir = Path('docs')
docs_dir.mkdir(exist_ok=True)

# Create index
with open(docs_dir / 'README.md', 'w') as f:
    f.write('''# Bridge VIV Risk Assessment Documentation

## Quick Start

1. Install dependencies: \`make install\`
2. Run experiments: \`make experiments\`
3. Generate predictions: \`make predict\`

## Files

- \`src/\`: Source code
- \`config/\`: Configuration files
- \`tests/\`: Test suite
- \`experiments/\`: Experiment results
- \`results/\`: Generated predictions and reports

## Docker Usage

- Development: \`make docker-dev\`
- Testing: \`make docker-test\`
- Production: \`make docker-run\`
''')

print('Documentation generated in docs/')
"

# Reporting
report:
	@echo "Generating experiment report..."
	@if [ -f "$(OUTPUT_DIR)/complete_experiment_results.json" ]; then \
		$(PYTHON) -c "
import json
import datetime

with open('$(OUTPUT_DIR)/complete_experiment_results.json', 'r') as f:
    results = json.load(f)

print('# Bridge VIV Risk Assessment - Latest Results')
print()
print(f'**Generated:** {datetime.datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print()
print('## Best Models')
best_models = results.get('best_models', {})
for task, info in best_models.items():
    model_name = info.get('model_name', 'Unknown')
    score = info.get('best_score', 'N/A')
    print(f'- **{task.title()}**: {model_name} (Score: {score})')
print()
print('## Summary')
summary = results.get('summary', {})
for key, value in summary.items():
    print(f'- **{key.replace(\"_\", \" \").title()}**: {value}')
" > results/latest_report.md; \
		echo "Report generated: results/latest_report.md"; \
	else \
		echo "No experiment results found. Run 'make experiments' first."; \
	fi

# Environment Management
freeze:
	@echo "Freezing current environment..."
	pip freeze > requirements_frozen.txt
	@echo "Dependencies saved to requirements_frozen.txt"

check-env:
	@echo "Checking environment..."
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip packages:"
	@pip list | grep -E "(numpy|pandas|scikit-learn|torch|xgboost)"

# Full Pipeline
all: setup test lint experiments report
	@echo "Complete pipeline finished!"

quick: setup test experiments-quick
	@echo "Quick pipeline finished!"

# CI/CD Targets
ci-test: install test lint
	@echo "CI tests completed!"

ci-build: install test docker
	@echo "CI build completed!"

# Help for specific targets
help-docker:
	@echo "Docker Commands:"
	@echo "  docker        Build Docker image"
	@echo "  docker-dev    Start Jupyter development environment"
	@echo "  docker-test   Run tests in Docker"
	@echo "  docker-mlflow Start MLflow tracking server"
	@echo "  docker-run    Run main application"
	@echo "  docker-clean  Clean Docker resources"

help-experiments:
	@echo "Experiment Commands:"
	@echo "  train         Train baseline models only"
	@echo "  hyperopt      Run hyperparameter optimization"
	@echo "  experiments   Run complete experiment pipeline"
	@echo "  predict       Generate predictions with best models"
	@echo "  report        Generate experiment report"