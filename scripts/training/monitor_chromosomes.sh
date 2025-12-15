#!/bin/bash
################################################################################
# Monitor Chromosome Processing Progress
#
# This script monitors the progress of base model pass chromosome processing.
#
# Usage:
#   bash scripts/training/monitor_chromosomes.sh [chromosome]
#
# Examples:
#   bash scripts/training/monitor_chromosomes.sh       # Overall status
#   bash scripts/training/monitor_chromosomes.sh 21    # Monitor chr21
################################################################################

PROJECT_ROOT="/Users/pleiadian53/work/meta-spliceai"
LOG_DIR="${PROJECT_ROOT}/logs/chromosomes"
OUTPUT_DIR="${PROJECT_ROOT}/data/mane/GRCh38/openspliceai_eval/meta_models"

# Function to show overall status
show_overall_status() {
    echo "
================================================================================
Chromosome Processing Status
================================================================================
"
    
    # Check for running processes
    echo "Running Processes:"
    if ls "${LOG_DIR}"/*.pid 2>/dev/null | head -1 > /dev/null; then
        for pid_file in "${LOG_DIR}"/*.pid; do
            pid=$(cat "${pid_file}")
            if ps -p ${pid} > /dev/null 2>&1; then
                chr=$(basename "${pid_file}" | sed 's/chr\([^_]*\)_.*/\1/')
                echo "  ✅ Chromosome ${chr} (PID: ${pid})"
            fi
        done
    else
        echo "  (none)"
    fi
    
    echo ""
    
    # Check completed chromosomes
    if [ -f "${LOG_DIR}/completed_chromosomes.txt" ]; then
        echo "Completed Chromosomes:"
        cat "${LOG_DIR}/completed_chromosomes.txt" | sort -V | sed 's/^/  ✅ /'
        echo ""
        echo "Total Completed: $(wc -l < ${LOG_DIR}/completed_chromosomes.txt | tr -d ' ')"
    else
        echo "Completed Chromosomes: (none yet)"
    fi
    
    echo ""
    
    # Check failed chromosomes
    if [ -f "${LOG_DIR}/failed_chromosomes.txt" ]; then
        echo "Failed Chromosomes:"
        cat "${LOG_DIR}/failed_chromosomes.txt" | sort -V | sed 's/^/  ❌ /'
        echo ""
        echo "Total Failed: $(wc -l < ${LOG_DIR}/failed_chromosomes.txt | tr -d ' ')"
    fi
    
    echo ""
    
    # Output directory size
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "Output Directory:"
        echo "  Location: ${OUTPUT_DIR}"
        echo "  Size: $(du -sh ${OUTPUT_DIR} 2>/dev/null | cut -f1)"
        echo "  Files: $(find ${OUTPUT_DIR} -type f 2>/dev/null | wc -l | tr -d ' ')"
    fi
    
    echo ""
    
    # Latest log activity
    echo "Latest Log Activity:"
    if ls "${LOG_DIR}"/chr*.log 2>/dev/null | head -1 > /dev/null; then
        latest_log=$(ls -t "${LOG_DIR}"/chr*.log | head -1)
        chr=$(basename "${latest_log}" | sed 's/chr\([^_]*\)_.*/\1/')
        echo "  Chromosome: ${chr}"
        echo "  Log: ${latest_log}"
        echo ""
        echo "  Last 5 lines:"
        tail -5 "${latest_log}" | sed 's/^/    /'
    else
        echo "  (no logs yet)"
    fi
    
    echo "
================================================================================
"
}

# Function to monitor specific chromosome
monitor_chromosome() {
    local chr=$1
    
    # Find latest log for this chromosome
    local log_file=$(ls -t "${LOG_DIR}"/chr${chr}_*.log 2>/dev/null | head -1)
    
    if [ -z "${log_file}" ]; then
        echo "❌ No log file found for chromosome ${chr}"
        echo ""
        echo "Available chromosomes:"
        ls "${LOG_DIR}"/chr*.log 2>/dev/null | sed 's/.*chr\([^_]*\)_.*/\1/' | sort -u | sed 's/^/  - /'
        exit 1
    fi
    
    echo "
================================================================================
Monitoring Chromosome ${chr}
================================================================================

Log: ${log_file}

Press Ctrl+C to stop monitoring
================================================================================
"
    
    tail -f "${log_file}"
}

# Main
if [ $# -eq 0 ]; then
    show_overall_status
else
    monitor_chromosome "$1"
fi

