#!/bin/bash

# Function to display usage instructions
show_usage() {
    echo "Usage: ./pipeline.sh [options]"
    echo "Options:"
    echo "  --optimize        Enable hyperparameter optimization"
    echo "  --trials NUMBER   Number of optimization trials (default: 100)"
    echo "  --analysis       Enable exploratory analysis"
    echo "  --help           Show this help message"
}

# Initialize default values
OPTIMIZE=false
TRIALS=100
ANALYSIS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --optimize)
            OPTIMIZE=true
            shift
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --analysis)
            ANALYSIS=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Verify virtual environment
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python -m venv env
    source env/Scripts/activate
    pip install -r requirements.txt
else
    source env/Scripts/activate
fi

# Build the command string based on options
CMD="python src\\run_pipeline.py"
CMD+=" --random-state 42"
CMD+=" --n-estimators 100"
CMD+=" --output-dir model_results"
CMD+=" --test-size 0.2"
CMD+=" --valid-size 0.2"
CMD+=" --balance smote"
CMD+=" --dataset-type base"
CMD+=" --collect-new-data"

# Add optimization flag if enabled
if [ "$OPTIMIZE" = true ]; then
    echo "Hyperparameter optimization enabled with $TRIALS trials"
    CMD+=" --optimize-hyperparameters"
    CMD+=" --optimization-trials $TRIALS"
fi

# Add analysis flag if enabled
if [ "$ANALYSIS" = true ]; then
    echo "Exploratory analysis enabled"
    CMD+=" --exploratory-analysis"
fi

# Execute the pipeline
echo "Executing pipeline with command:"
echo "$CMD"
$CMD

deactivate