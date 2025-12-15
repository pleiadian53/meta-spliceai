#!/bin/bash
# Simple wrapper to start the full genome sequential run with nohup

cd /Users/pleiadian53/work/meta-spliceai

nohup bash scripts/training/run_chromosomes_sequential.sh > logs/full_genome_sequential.log 2>&1 &

PID=$!
echo "Started sequential chromosome processing"
echo "Process ID: ${PID}"
echo "Log file: logs/full_genome_sequential.log"
echo ""
echo "Monitor with:"
echo "  tail -f logs/full_genome_sequential.log"
echo ""
echo "Check status with:"
echo "  bash scripts/training/monitor_chromosomes.sh"

