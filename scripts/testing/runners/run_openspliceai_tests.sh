#!/bin/bash
# Run OpenSpliceAI Tests
# This script runs both the simple and full OpenSpliceAI tests

set -e

echo "================================================================================"
echo "OpenSpliceAI Testing Suite"
echo "================================================================================"
echo

# Ensure we're in the project directory
cd /Users/pleiadian53/work/meta-spliceai

# Create logs directory
mkdir -p logs

# Activate environment (using the method that works)
echo "Step 1: Activating surveyor environment..."
export PATH="/Users/pleiadian53/miniforge3-new/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /Users/pleiadian53/miniforge3-new/envs/surveyor

echo "✅ Environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo

# Test 1: Simple test (5 genes)
echo "================================================================================"
echo "TEST 1: Simple OpenSpliceAI Test (5 genes)"
echo "================================================================================"
echo

LOG_SIMPLE="logs/openspliceai_simple_$(date +%Y%m%d_%H%M%S).log"
echo "Running simple test (logging to: $LOG_SIMPLE)..."
echo

python scripts/testing/test_openspliceai_simple.py 2>&1 | tee "$LOG_SIMPLE"

SIMPLE_EXIT=$?
if [ $SIMPLE_EXIT -eq 0 ]; then
    echo
    echo "✅ Simple test completed successfully!"
    echo
else
    echo
    echo "❌ Simple test failed with exit code: $SIMPLE_EXIT"
    echo "Check log: $LOG_SIMPLE"
    exit $SIMPLE_EXIT
fi

# Test 2: Full test (30 genes)
echo "================================================================================"
echo "TEST 2: Full OpenSpliceAI Test (30 genes)"
echo "================================================================================"
echo

read -p "Proceed with full 30-gene test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    LOG_FULL="logs/openspliceai_full_$(date +%Y%m%d_%H%M%S).log"
    echo "Running full test (logging to: $LOG_FULL)..."
    echo
    
    python scripts/testing/test_openspliceai_gene_categories.py 2>&1 | tee "$LOG_FULL"
    
    FULL_EXIT=$?
    if [ $FULL_EXIT -eq 0 ]; then
        echo
        echo "✅ Full test completed successfully!"
        echo
    else
        echo
        echo "❌ Full test failed with exit code: $FULL_EXIT"
        echo "Check log: $LOG_FULL"
        exit $FULL_EXIT
    fi
else
    echo "Skipping full test."
fi

echo
echo "================================================================================"
echo "All tests complete!"
echo "================================================================================"
echo
echo "Logs:"
echo "  Simple test: $LOG_SIMPLE"
if [ ! -z "$LOG_FULL" ]; then
    echo "  Full test: $LOG_FULL"
fi
echo

