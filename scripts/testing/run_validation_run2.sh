#!/bin/bash

# Base Model Validation - Run 2
# Independent test with fresh gene sample to validate consistency

set -e

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Setup logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/validation_run2_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=============================================================================="
echo "BASE MODEL VALIDATION - RUN 2"
echo "=============================================================================="
echo ""
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "This test will:"
echo "  • Sample 30 new genes (20 protein-coding, 10 lncRNA)"
echo "  • Run base model predictions"
echo "  • Compare results with Run 1"
echo "  • Validate consistency and reproducibility"
echo ""
echo "Starting test in background..."
echo ""

# Run test in background with nohup
cd "$PROJECT_ROOT"

nohup conda run -n surveyor --no-capture-output python \
    scripts/testing/test_base_model_validation_run2.py \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "✅ Test started successfully"
echo ""
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check status:"
echo "  ps -p $PID"
echo ""
echo "=============================================================================="

