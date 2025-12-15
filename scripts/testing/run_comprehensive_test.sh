#!/bin/bash
# Run comprehensive base model test in background

cd /Users/pleiadian53/work/meta-spliceai

# Create logs directory
mkdir -p logs

# Run test in background with nohup using conda run
LOGFILE="logs/base_model_comprehensive_$(date +%Y%m%d_%H%M%S).log"

nohup conda run -n surveyor --no-capture-output python scripts/testing/test_base_model_comprehensive.py > "$LOGFILE" 2>&1 &

PID=$!

echo "============================================"
echo "Base Model Comprehensive Test Started"
echo "============================================"
echo "PID: $PID"
echo "Log file: $LOGFILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOGFILE"
echo ""
echo "To check status:"
echo "  ps aux | grep $PID"
echo "============================================"

