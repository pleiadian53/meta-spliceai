#!/bin/bash
# Monitor OpenSpliceAI Test Progress

echo "Monitoring OpenSpliceAI Test..."
echo "Press Ctrl+C to stop monitoring"
echo

# Find the most recent log file
LOG_FILE=$(ls -t logs/openspliceai_test_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No log file found. Test may not have started yet."
    exit 1
fi

echo "Log file: $LOG_FILE"
echo "=" * 80
echo

# Tail the log file
tail -f "$LOG_FILE"

