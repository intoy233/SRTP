#!/bin/bash
# Bridge VIV Risk Assessment - Experiment Runner Script

set -e  # Exit on any error

# Configuration
CONFIG_FILE="${CONFIG_FILE:-config/config.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bridge_viv_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments}"
PYTHON_CMD="${PYTHON_CMD:-python}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."

    if ! command -v $PYTHON_CMD &> /dev/null; then
        error "Python not found. Please install Python 3.9+ or set PYTHON_CMD environment variable."
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
    log "Found Python $PYTHON_VERSION"

    # Check if required Python packages are available
    $PYTHON_CMD -c "import numpy, pandas, sklearn" 2>/dev/null || {
        error "Required Python packages not found. Please install requirements: pip install -r requirements.txt"
        exit 1
    }

    success "Dependencies check passed"
}

# Setup directories
setup_directories() {
    log "Setting up directories..."

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "results"
    mkdir -p "logs"

    success "Directories created"
}

# Run data validation
validate_data() {
    log "Validating data..."

    if [ ! -f "$CONFIG_FILE" ]; then
        error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi

    # Extract data file path from config
    DATA_FILE=$($PYTHON_CMD -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('data', {}).get('file_path', ''))
" 2>/dev/null || echo "")

    if [ -z "$DATA_FILE" ]; then
        error "Data file path not found in config"
        exit 1
    fi

    if [ ! -f "$DATA_FILE" ]; then
        error "Data file not found: $DATA_FILE"
        exit 1
    fi

    success "Data validation passed"
}

# Run baseline experiments
run_baseline() {
    log "Running baseline experiments..."

    $PYTHON_CMD src/train.py \
        --config "$CONFIG_FILE" \
        --experiment-name "${EXPERIMENT_NAME}_baseline" \
        --models linear random_forest \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "logs/${EXPERIMENT_NAME}_baseline.log"

    if [ $? -eq 0 ]; then
        success "Baseline experiments completed"
    else
        error "Baseline experiments failed"
        exit 1
    fi
}

# Run hyperparameter optimization
run_hyperopt() {
    log "Running hyperparameter optimization..."

    $PYTHON_CMD src/hyperparam_search.py \
        --config "$CONFIG_FILE" \
        --experiment-name "${EXPERIMENT_NAME}_hyperopt" \
        --n-trials 50 \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "logs/${EXPERIMENT_NAME}_hyperopt.log"

    if [ $? -eq 0 ]; then
        success "Hyperparameter optimization completed"
    else
        warning "Hyperparameter optimization failed, continuing with baseline results"
    fi
}

# Run complete experiment pipeline
run_complete_experiments() {
    log "Running complete experiment pipeline..."

    $PYTHON_CMD src/experiments.py \
        --config "$CONFIG_FILE" \
        --experiment-name "$EXPERIMENT_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --full-pipeline \
        2>&1 | tee "logs/${EXPERIMENT_NAME}_complete.log"

    if [ $? -eq 0 ]; then
        success "Complete experiments finished"
    else
        error "Complete experiments failed"
        exit 1
    fi
}

# Generate predictions on test data
generate_predictions() {
    log "Generating predictions..."

    # Find best model
    BEST_MODEL=$($PYTHON_CMD -c "
import json
import os
results_file = os.path.join('$OUTPUT_DIR', 'complete_experiment_results.json')
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    best_models = results.get('best_models', {})
    amplitude_model = best_models.get('amplitude', {}).get('model_path', '')
    print(amplitude_model)
else:
    print('')
" 2>/dev/null || echo "")

    if [ -n "$BEST_MODEL" ] && [ -f "$BEST_MODEL" ]; then
        DATA_FILE=$($PYTHON_CMD -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('data', {}).get('file_path', ''))
")

        $PYTHON_CMD src/predict.py \
            --model "$BEST_MODEL" \
            --input "$DATA_FILE" \
            --output "results/predictions_${EXPERIMENT_NAME}.csv" \
            --comprehensive \
            2>&1 | tee "logs/${EXPERIMENT_NAME}_predict.log"

        if [ $? -eq 0 ]; then
            success "Predictions generated: results/predictions_${EXPERIMENT_NAME}.csv"
        else
            warning "Prediction generation failed"
        fi
    else
        warning "Best model not found, skipping prediction generation"
    fi
}

# Run tests
run_tests() {
    log "Running tests..."

    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short 2>&1 | tee "logs/${EXPERIMENT_NAME}_tests.log"

        if [ $? -eq 0 ]; then
            success "All tests passed"
        else
            warning "Some tests failed, check logs for details"
        fi
    else
        warning "pytest not found, skipping tests"
    fi
}

# Generate experiment report
generate_report() {
    log "Generating experiment report..."

    REPORT_FILE="results/experiment_report_${EXPERIMENT_NAME}.md"

    cat > "$REPORT_FILE" << EOF
# Bridge VIV Risk Assessment - Experiment Report

**Experiment Name:** $EXPERIMENT_NAME
**Date:** $(date)
**Configuration:** $CONFIG_FILE

## Experiment Summary

This report summarizes the results of the Bridge VIV risk assessment experiment.

### Files Generated

- **Models:** $OUTPUT_DIR/
- **Predictions:** results/predictions_${EXPERIMENT_NAME}.csv
- **Logs:** logs/${EXPERIMENT_NAME}_*.log

### Key Results

EOF

    # Add results if available
    if [ -f "$OUTPUT_DIR/complete_experiment_results.json" ]; then
        $PYTHON_CMD -c "
import json
with open('$OUTPUT_DIR/complete_experiment_results.json', 'r') as f:
    results = json.load(f)

print('**Best Models:**')
best_models = results.get('best_models', {})
for task, info in best_models.items():
    model_name = info.get('model_name', 'Unknown')
    score = info.get('best_score', 'N/A')
    print(f'- {task.title()}: {model_name} (Score: {score})')

print()
print('**Performance Summary:**')
summary = results.get('summary', {})
for key, value in summary.items():
    print(f'- {key.replace(\"_\", \" \").title()}: {value}')
" >> "$REPORT_FILE"
    fi

    cat >> "$REPORT_FILE" << EOF

### Usage

To use the trained models for prediction:

\`\`\`bash
python src/predict.py \\
    --model $OUTPUT_DIR/best_model.pkl \\
    --input your_data.csv \\
    --output predictions.csv \\
    --comprehensive
\`\`\`

### Next Steps

1. Review model performance metrics
2. Analyze feature importance results
3. Consider additional feature engineering
4. Deploy best performing model

---
*Generated by Bridge VIV Risk Assessment System*
EOF

    success "Experiment report generated: $REPORT_FILE"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    # Add cleanup logic if needed
}

# Main execution
main() {
    log "Starting Bridge VIV Risk Assessment Experiments"
    log "Experiment Name: $EXPERIMENT_NAME"
    log "Configuration: $CONFIG_FILE"
    log "Output Directory: $OUTPUT_DIR"

    # Setup
    check_dependencies
    setup_directories
    validate_data

    # Run experiments based on arguments
    case "${1:-full}" in
        "baseline")
            run_baseline
            ;;
        "hyperopt")
            run_hyperopt
            ;;
        "predict")
            generate_predictions
            ;;
        "test")
            run_tests
            ;;
        "full")
            run_baseline
            run_hyperopt
            run_complete_experiments
            generate_predictions
            run_tests
            generate_report
            ;;
        *)
            error "Unknown command: $1"
            echo "Usage: $0 [baseline|hyperopt|predict|test|full]"
            exit 1
            ;;
    esac

    cleanup
    success "Experiment pipeline completed successfully!"
}

# Handle script termination
trap cleanup EXIT

# Run main function with all arguments
main "$@"