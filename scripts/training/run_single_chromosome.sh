#!/bin/bash
################################################################################
# Run Base Model Pass for a Single Chromosome
#
# This script runs the base model pass for a single chromosome with nohup
# and tee for monitoring. Use this for manual chromosome-by-chromosome control.
#
# Usage:
#   bash scripts/training/run_single_chromosome.sh <chromosome>
#
# Example:
#   bash scripts/training/run_single_chromosome.sh 21
#   bash scripts/training/run_single_chromosome.sh X
################################################################################

# Check arguments
if [ $# -eq 0 ]; then
    echo "❌ Error: Chromosome argument required"
    echo ""
    echo "Usage:"
    echo "  bash scripts/training/run_single_chromosome.sh <chromosome>"
    echo ""
    echo "Examples:"
    echo "  bash scripts/training/run_single_chromosome.sh 21"
    echo "  bash scripts/training/run_single_chromosome.sh X"
    echo ""
    exit 1
fi

CHROMOSOME=$1

# Configuration
BASE_MODEL="openspliceai"
MODE="production"
COVERAGE="full_genome"
VERBOSITY=2

# Directories
PROJECT_ROOT="/Users/pleiadian53/work/meta-spliceai"
LOG_DIR="${PROJECT_ROOT}/logs/chromosomes"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/chr${CHROMOSOME}_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/chr${CHROMOSOME}_${TIMESTAMP}.pid"

# Create log directory
mkdir -p "${LOG_DIR}"

echo "================================================================================
Starting Chromosome ${CHROMOSOME} Base Model Pass
================================================================================

Configuration:
  Base Model:  ${BASE_MODEL}
  Mode:        ${MODE}
  Coverage:    ${COVERAGE}
  Chromosome:  ${CHROMOSOME}
  Verbosity:   ${VERBOSITY}

Output:
  Log File:    ${LOG_FILE}
  PID File:    ${PID_FILE}

Started:       $(date)
================================================================================
"

# Run with nohup and tee
nohup bash -c "
    source ~/.bash_profile
    mamba activate metaspliceai
    
    cd ${PROJECT_ROOT}
    
    python scripts/training/run_full_genome_base_model_pass.py \
        --base-model ${BASE_MODEL} \
        --mode ${MODE} \
        --coverage ${COVERAGE} \
        --chromosomes ${CHROMOSOME} \
        --verbosity ${VERBOSITY}
" > >(tee "${LOG_FILE}") 2>&1 &

# Save PID
echo $! > "${PID_FILE}"

echo "
✅ Background process started!

Process ID: $(cat ${PID_FILE})

Monitor progress:
  tail -f ${LOG_FILE}

Check status:
  ps -p \$(cat ${PID_FILE})

Kill process:
  kill \$(cat ${PID_FILE})

================================================================================
"

