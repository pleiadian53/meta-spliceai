#!/bin/bash
# Run OpenSpliceAI Gene Category Test
# This script ensures the environment is activated and runs the test

set -e

echo "================================"
echo "OpenSpliceAI Test Runner"
echo "================================"
echo

# Activate environment
echo "Activating surveyor environment..."
source ~/.bash_profile
mamba activate surveyor

# Change to project directory
cd /Users/pleiadian53/work/meta-spliceai

# Create logs directory
mkdir -p logs

# Run test with logging
LOG_FILE="logs/openspliceai_test_$(date +%Y%m%d_%H%M%S).log"
echo "Running test (logging to: $LOG_FILE)..."
echo

python scripts/testing/test_openspliceai_gene_categories.py 2>&1 | tee "$LOG_FILE"

echo
echo "================================"
echo "Test complete!"
echo "Log file: $LOG_FILE"
echo "================================"

