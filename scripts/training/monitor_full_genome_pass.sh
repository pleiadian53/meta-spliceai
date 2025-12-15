#!/bin/bash
# Monitor Full Genome Base Model Pass

LOG_PATTERN="logs/full_genome_openspliceai_*.log"
LATEST_LOG=$(ls -t $LOG_PATTERN 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log file found matching pattern: $LOG_PATTERN"
    exit 1
fi

echo "================================================================================"
echo "MONITORING FULL GENOME BASE MODEL PASS"
echo "================================================================================"
echo "Log file: $LATEST_LOG"
echo ""

# Check if process is running
PROCESS=$(ps aux | grep -E "run_full_genome_base_model_pass" | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo "⚠️  Process is NOT running"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$LATEST_LOG"
    exit 1
else
    PID=$(echo "$PROCESS" | awk '{print $2}')
    echo "✅ Process is running (PID: $PID)"
    echo ""
fi

# Get log statistics
TOTAL_LINES=$(wc -l < "$LATEST_LOG" 2>/dev/null || echo "0")
FILE_SIZE=$(du -h "$LATEST_LOG" | cut -f1)
echo "Log Statistics:"
echo "  Lines: $TOTAL_LINES"
echo "  Size: $FILE_SIZE"
echo ""

# Check for errors
ERROR_COUNT=$(grep -i "error\|exception\|traceback\|failed" "$LATEST_LOG" | wc -l | tr -d ' ')
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Found $ERROR_COUNT potential errors in log"
    echo ""
    echo "Recent errors:"
    grep -i "error\|exception\|traceback\|failed" "$LATEST_LOG" | tail -5
    echo ""
else
    echo "✅ No errors found in log"
    echo ""
fi

# Check progress
echo "Recent Progress:"
echo "---"
tail -30 "$LATEST_LOG" | grep -E "(Processing chromosomes|genes to process|genes [0-9]+-[0-9]+|chromosome|STEP|COMPREHENSIVE|FINAL SUMMARY|✅|❌)" | tail -10
echo "---"
echo ""

# Check for completion
if grep -q "FINAL SUMMARY\|Ready for meta-learning\|✅ Full genome pass completed" "$LATEST_LOG"; then
    echo "🎉 PROCESS COMPLETED!"
    echo ""
    echo "Summary:"
    grep -A 20 "FINAL SUMMARY\|Ready for meta-learning" "$LATEST_LOG" | tail -20
else
    echo "⏳ Process still running..."
    echo ""
    echo "Current status (last 10 lines):"
    tail -10 "$LATEST_LOG"
fi

echo ""
echo "================================================================================"

