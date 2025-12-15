#!/bin/bash
# Launch full genome base model pass in background with monitoring
# This script uses nohup + tee so you can:
# - Monitor progress in real-time: tail -f logs/full_genome_cli.log
# - Run in background: process continues even if terminal closes

# Ensure logs directory exists
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_genome_cli_$TIMESTAMP.log"

echo "============================================================"
echo "Launching Full Genome Base Model Pass (Background)"
echo "============================================================"
echo "Log file: $LOG_FILE"
echo "Monitor with: tail -f $LOG_FILE"
echo "============================================================"
echo ""

# Launch in background with nohup + tee
nohup bash scripts/training/run_all_chromosomes_cli.sh 2>&1 | tee "$LOG_FILE" &

# Get process ID
PID=$!

echo "✅ Process started in background"
echo "   PID: $PID"
echo "   Log: $LOG_FILE"
echo ""
echo "To monitor:"
echo "   tail -f $LOG_FILE"
echo ""
echo "To check if running:"
echo "   ps aux | grep $PID"
echo ""
echo "To stop (if needed):"
echo "   kill $PID"
echo ""

