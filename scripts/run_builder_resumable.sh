#!/bin/bash
#==============================================================================
# Resumable Incremental Training Dataset Builder
#==============================================================================
# This script runs the incremental builder in a way that can survive:
# - Laptop going to sleep
# - SSH disconnections
# - Terminal closure
#
# Methods provided:
# 1. nohup (simplest, no dependencies)
# 2. tmux (recommended, can reattach)
# 3. screen (alternative to tmux)
#
# The script also implements proper signal handling for graceful shutdowns

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT="/Users/pleiadian53/work/meta-spliceai"
OUTPUT_DIR="${PROJECT_ROOT}/data/train_pc_1000_3mers"
LOG_DIR="${PROJECT_ROOT}/logs"
ADDITIONAL_GENES="${PROJECT_ROOT}/additional_genes.tsv"

# Dataset parameters (customize as needed)
N_GENES=1000
SUBSET_POLICY="error_total"
BATCH_SIZE=100
BATCH_ROWS=20000
KMER_SIZES="3"

# =============================================================================
# Helper Functions
# =============================================================================

usage() {
    cat << EOF
Usage: $0 [METHOD]

Run incremental_builder in a resumable way using one of:
  nohup    - Run with nohup (simplest, process continues after logout)
  tmux     - Run in tmux session (recommended, can reattach)
  screen   - Run in screen session (alternative to tmux)
  direct   - Run directly (for testing, not resumable)

Examples:
  $0 nohup     # Run with nohup, logs to nohup.out
  $0 tmux      # Run in tmux session named 'builder'
  $0 screen    # Run in screen session named 'builder'

To reattach to a running session:
  tmux attach -t builder    # For tmux
  screen -r builder         # For screen

To check if process is running:
  ps aux | grep incremental_builder
  
To monitor progress:
  tail -f ${LOG_DIR}/incremental_builder_*.log
EOF
    exit 1
}

setup_environment() {
    echo "Setting up environment..."
    cd "${PROJECT_ROOT}"
    mkdir -p "${LOG_DIR}"
    
    # Create timestamp for this run
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/incremental_builder_${TIMESTAMP}.log"
    export LOG_FILE
    
    echo "Log file: ${LOG_FILE}"
}

check_conda_env() {
    # Check if surveyor environment exists
    if ! mamba env list | grep -q "surveyor"; then
        echo "ERROR: surveyor conda environment not found"
        echo "Create it first or modify the script to use your environment"
        exit 1
    fi
}

# The actual command to run
builder_command() {
    source ~/.zshrc
    mamba activate surveyor
    
    echo "Starting incremental builder at $(date)"
    echo "Parameters:"
    echo "  N_GENES: ${N_GENES}"
    echo "  SUBSET_POLICY: ${SUBSET_POLICY}"
    echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
    echo ""
    
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
        --n-genes "${N_GENES}" \
        --subset-policy "${SUBSET_POLICY}" \
        --gene-ids-file "${ADDITIONAL_GENES}" \
        --gene-col gene_id \
        --batch-size "${BATCH_SIZE}" \
        --batch-rows "${BATCH_ROWS}" \
        --kmer-sizes ${KMER_SIZES} \
        --output-dir "${OUTPUT_DIR}" \
        --overwrite \
        -vv \
        2>&1 | tee -a "${LOG_FILE}"
    
    EXIT_CODE=${PIPESTATUS[0]}
    echo ""
    echo "Finished at $(date) with exit code ${EXIT_CODE}"
    return ${EXIT_CODE}
}

# =============================================================================
# Execution Methods
# =============================================================================

run_with_nohup() {
    echo "Running with nohup..."
    echo "Process will continue even if you log out or close the terminal"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f ${LOG_FILE}"
    echo "  tail -f nohup.out"
    echo ""
    echo "To stop the process:"
    echo "  ps aux | grep incremental_builder"
    echo "  kill <PID>"
    echo ""
    
    # Export function so it's available to bash -c
    export -f builder_command
    export N_GENES SUBSET_POLICY BATCH_SIZE BATCH_ROWS KMER_SIZES OUTPUT_DIR ADDITIONAL_GENES LOG_FILE
    
    nohup bash -c "$(declare -f builder_command); builder_command" > nohup.out 2>&1 &
    PID=$!
    
    echo "✅ Process started with PID: ${PID}"
    echo "Log files:"
    echo "  - ${LOG_FILE}"
    echo "  - nohup.out"
    echo ""
    echo "Check status: ps -p ${PID}"
}

run_with_tmux() {
    if ! command -v tmux &> /dev/null; then
        echo "ERROR: tmux is not installed"
        echo "Install with: brew install tmux"
        exit 1
    fi
    
    SESSION_NAME="builder"
    
    # Kill existing session if it exists
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Killing existing tmux session: ${SESSION_NAME}"
        tmux kill-session -t "${SESSION_NAME}"
    fi
    
    echo "Starting tmux session: ${SESSION_NAME}"
    echo ""
    echo "To reattach: tmux attach -t ${SESSION_NAME}"
    echo "To detach: Press Ctrl+b then d"
    echo "To kill session: tmux kill-session -t ${SESSION_NAME}"
    echo ""
    
    # Export variables for tmux session
    export N_GENES SUBSET_POLICY BATCH_SIZE BATCH_ROWS KMER_SIZES OUTPUT_DIR ADDITIONAL_GENES LOG_FILE
    
    # Create tmux session and run command
    tmux new-session -d -s "${SESSION_NAME}" \
        "bash -c 'source ~/.zshrc; $(declare -f builder_command); builder_command; echo; echo Press ENTER to exit; read'"
    
    echo "✅ tmux session started"
    echo ""
    echo "Attaching to session (you can detach anytime with Ctrl+b d)..."
    sleep 1
    tmux attach -t "${SESSION_NAME}"
}

run_with_screen() {
    if ! command -v screen &> /dev/null; then
        echo "ERROR: screen is not installed"
        echo "Install with: brew install screen"
        exit 1
    fi
    
    SESSION_NAME="builder"
    
    # Kill existing session if it exists
    if screen -list | grep -q "${SESSION_NAME}"; then
        echo "Killing existing screen session: ${SESSION_NAME}"
        screen -S "${SESSION_NAME}" -X quit
    fi
    
    echo "Starting screen session: ${SESSION_NAME}"
    echo ""
    echo "To reattach: screen -r ${SESSION_NAME}"
    echo "To detach: Press Ctrl+a then d"
    echo "To kill session: screen -S ${SESSION_NAME} -X quit"
    echo ""
    
    # Export variables
    export N_GENES SUBSET_POLICY BATCH_SIZE BATCH_ROWS KMER_SIZES OUTPUT_DIR ADDITIONAL_GENES LOG_FILE
    
    # Create screen session and run command
    screen -dmS "${SESSION_NAME}" bash -c \
        "source ~/.zshrc; $(declare -f builder_command); builder_command; echo; echo 'Press ENTER to exit'; read"
    
    echo "✅ screen session started"
    echo ""
    echo "Attaching to session (you can detach anytime with Ctrl+a d)..."
    sleep 1
    screen -r "${SESSION_NAME}"
}

run_direct() {
    echo "Running directly (not resumable)..."
    echo "Press Ctrl+C to stop"
    echo ""
    builder_command
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
METHOD=${1:-}

if [[ -z "${METHOD}" ]]; then
    usage
fi

case "${METHOD}" in
    nohup)
        setup_environment
        check_conda_env
        run_with_nohup
        ;;
    tmux)
        setup_environment
        check_conda_env
        run_with_tmux
        ;;
    screen)
        setup_environment
        check_conda_env
        run_with_screen
        ;;
    direct)
        setup_environment
        check_conda_env
        run_direct
        ;;
    *)
        echo "ERROR: Unknown method: ${METHOD}"
        echo ""
        usage
        ;;
esac

