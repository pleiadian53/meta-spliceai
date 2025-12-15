#!/bin/bash
# Monitor meta-model training progress
# Usage: ./scripts/monitor_training.sh [PID]

set -e

# Get PID from argument or find running process
if [ -n "$1" ]; then
    PID=$1
else
    PID=$(ps aux | grep "run_gene_cv_sigmoid" | grep -v grep | awk '{print $2}' | head -1)
fi

if [ -z "$PID" ]; then
    echo "❌ No training process found"
    echo "Checking recent log files..."
    ls -lht logs/meta_training_*.log 2>/dev/null | head -3
    exit 1
fi

# Get latest log file
LATEST_LOG=$(ls -t logs/meta_training_*.log 2>/dev/null | head -1)

echo "=================================="
echo "Meta-Model Training Monitor"
echo "=================================="
echo ""

# Process status
echo "📊 Process Status:"
ps -p $PID -o pid,ppid,%cpu,%mem,etime,state,command 2>/dev/null || {
    echo "  ❌ Process $PID not found (may have completed)"
    echo ""
    echo "📄 Final Log Output:"
    if [ -n "$LATEST_LOG" ]; then
        tail -50 "$LATEST_LOG" | grep -E "Fold|F1=|AP=|completed|Success|Error"
    fi
    exit 0
}

echo ""

# Training progress
if [ -n "$LATEST_LOG" ]; then
    echo "📈 Training Progress:"
    echo "  Log file: $LATEST_LOG"
    echo ""
    
    # Extract fold information
    CURRENT_FOLD=$(tail -100 "$LATEST_LOG" | grep -E "🔀 Fold [0-9]+/[0-9]+" | tail -1)
    if [ -n "$CURRENT_FOLD" ]; then
        echo "  Current: $CURRENT_FOLD"
    fi
    
    # Extract latest metrics
    echo ""
    echo "  Recent Fold Results:"
    tail -200 "$LATEST_LOG" | grep -E "✅ Fold [0-9]+ results:" | tail -3 | sed 's/^/    /'
    
    echo ""
    echo "  Latest Activity (last 5 lines):"
    tail -5 "$LATEST_LOG" | sed 's/^/    /'
fi

echo ""
echo "=================================="
echo ""
echo "📋 Quick Commands:"
echo "  • View live log:  tail -f $LATEST_LOG"
echo "  • Check process:  ps -p $PID"
echo "  • Kill training:  kill $PID"
echo "  • Check output:   ls -lh results/meta_model_1000genes_3mers/"
echo ""

