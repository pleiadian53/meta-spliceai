#!/bin/bash
################################################################################
# Run Full Genome Base Model Pass - Chromosome by Chromosome
#
# This script runs the base model pass sequentially for each chromosome to
# avoid memory issues on machines with limited RAM (16GB).
#
# Usage:
#   bash scripts/training/run_chromosomes_sequential.sh
#
# Features:
# - Runs one chromosome at a time
# - Uses nohup with tee for background execution and monitoring
# - Activates mamba environment
# - Saves logs for each chromosome
# - Tracks progress across chromosomes
################################################################################

set -e  # Exit on error

# Configuration
BASE_MODEL="openspliceai"
MODE="production"
COVERAGE="full_genome"
VERBOSITY=2

# Chromosomes to process (all autosomes + sex chromosomes)
CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y)

# Directories
PROJECT_ROOT="/Users/pleiadian53/work/meta-spliceai"
LOG_DIR="${PROJECT_ROOT}/logs/chromosomes"
OUTPUT_DIR="${PROJECT_ROOT}/data/mane/GRCh38/${BASE_MODEL}_eval/meta_models"

# Create log directory
mkdir -p "${LOG_DIR}"

# Function to run a single chromosome
run_chromosome() {
    local chr=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="${LOG_DIR}/chr${chr}_${timestamp}.log"
    
    echo "================================================================================
$(date): Starting chromosome ${chr}
================================================================================"
    
    # Activate environment and run
    (
        source ~/.bash_profile
        mamba activate metaspliceai
        
        cd "${PROJECT_ROOT}"
        
        python scripts/training/run_full_genome_base_model_pass.py \
            --base-model "${BASE_MODEL}" \
            --mode "${MODE}" \
            --coverage "${COVERAGE}" \
            --chromosomes "${chr}" \
            --verbosity "${VERBOSITY}"
    ) 2>&1 | tee "${log_file}"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ ${exit_code} -eq 0 ]; then
        echo "✅ Chromosome ${chr} completed successfully ($(date))"
        echo "${chr}" >> "${LOG_DIR}/completed_chromosomes.txt"
    else
        echo "❌ Chromosome ${chr} failed with exit code ${exit_code} ($(date))"
        echo "${chr}" >> "${LOG_DIR}/failed_chromosomes.txt"
        return ${exit_code}
    fi
    
    echo ""
}

# Main execution
main() {
    echo "
################################################################################
Full Genome Base Model Pass - Sequential Execution
################################################################################

Base Model: ${BASE_MODEL}
Mode: ${MODE}
Total Chromosomes: ${#CHROMOSOMES[@]}
Output Directory: ${OUTPUT_DIR}
Log Directory: ${LOG_DIR}

Started: $(date)
################################################################################
"
    
    # Clear previous progress files
    rm -f "${LOG_DIR}/completed_chromosomes.txt"
    rm -f "${LOG_DIR}/failed_chromosomes.txt"
    
    # Process each chromosome
    local total=${#CHROMOSOMES[@]}
    local current=0
    
    for chr in "${CHROMOSOMES[@]}"; do
        current=$((current + 1))
        echo "
[${current}/${total}] Processing chromosome ${chr}..."
        
        run_chromosome "${chr}"
        
        # Small delay between chromosomes
        sleep 5
    done
    
    # Summary
    echo "
################################################################################
Sequential Processing Complete
################################################################################

Completed: $(date)

Summary:
"
    
    if [ -f "${LOG_DIR}/completed_chromosomes.txt" ]; then
        local completed=$(wc -l < "${LOG_DIR}/completed_chromosomes.txt" | tr -d ' ')
        echo "  ✅ Completed: ${completed} chromosomes"
    else
        echo "  ✅ Completed: 0 chromosomes"
    fi
    
    if [ -f "${LOG_DIR}/failed_chromosomes.txt" ]; then
        local failed=$(wc -l < "${LOG_DIR}/failed_chromosomes.txt" | tr -d ' ')
        echo "  ❌ Failed: ${failed} chromosomes"
        echo ""
        echo "Failed chromosomes:"
        cat "${LOG_DIR}/failed_chromosomes.txt" | sed 's/^/    - /'
    else
        echo "  ❌ Failed: 0 chromosomes"
    fi
    
    echo "
Logs: ${LOG_DIR}
Output: ${OUTPUT_DIR}
################################################################################
"
}

# Run main
main

