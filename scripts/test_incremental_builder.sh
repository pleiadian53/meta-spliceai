#!/bin/bash
#==============================================================================
# Test Script: Incremental Training Dataset Builder
#==============================================================================
# This script tests the incremental_builder.py with resumable execution support
#
# Features:
# - Small dataset for quick testing (--n-genes 100)
# - Tests all command-line argument logic
# - Includes ALS-related genes from additional_genes.tsv
# - Resumable via checkpointing
# - Proper logging

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT="/Users/pleiadian53/work/meta-spliceai"
OUTPUT_DIR="${PROJECT_ROOT}/data/train_test_100genes_3mers"
LOG_DIR="${PROJECT_ROOT}/logs"
ADDITIONAL_GENES="${PROJECT_ROOT}/additional_genes.tsv"

# Dataset parameters
N_GENES=100
SUBSET_POLICY="error_total"
BATCH_SIZE=50
BATCH_ROWS=20000
KMER_SIZES="3"

# Create log directory
mkdir -p "${LOG_DIR}"

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/incremental_builder_test_${TIMESTAMP}.log"

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_FILE}" >&2
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

log "Starting incremental builder test"
log "Project root: ${PROJECT_ROOT}"
log "Output directory: ${OUTPUT_DIR}"
log "Log file: ${LOG_FILE}"

# Check if additional genes file exists
if [[ ! -f "${ADDITIONAL_GENES}" ]]; then
    log_error "Additional genes file not found: ${ADDITIONAL_GENES}"
    exit 1
fi

log "✅ Found additional genes file"
log "   $(wc -l < "${ADDITIONAL_GENES}") genes listed"

# Activate conda environment
log "Activating surveyor environment..."
source ~/.zshrc
if ! mamba activate surveyor 2>&1 | tee -a "${LOG_FILE}"; then
    log_error "Failed to activate surveyor environment"
    exit 1
fi

log "✅ Environment activated"
log "   Python: $(which python)"
log "   Python version: $(python --version)"

# =============================================================================
# Run Incremental Builder
# =============================================================================

log ""
log "="*80
log "Running incremental_builder.py with parameters:"
log "  --n-genes: ${N_GENES}"
log "  --subset-policy: ${SUBSET_POLICY}"
log "  --batch-size: ${BATCH_SIZE}"
log "  --batch-rows: ${BATCH_ROWS}"
log "  --kmer-sizes: ${KMER_SIZES}"
log "  --gene-ids-file: ${ADDITIONAL_GENES}"
log "  --output-dir: ${OUTPUT_DIR}"
log "="*80
log ""

# Change to project root
cd "${PROJECT_ROOT}"

# Run the builder
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

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [[ ${EXIT_CODE} -eq 0 ]]; then
    log ""
    log "="*80
    log "✅ SUCCESS: Incremental builder completed"
    log "="*80
    log ""
    log "Output files in: ${OUTPUT_DIR}"
    
    # List output files
    if [[ -d "${OUTPUT_DIR}" ]]; then
        log "Output directory contents:"
        ls -lh "${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
    fi
    
else
    log ""
    log "="*80
    log_error "FAILED: Incremental builder exited with code ${EXIT_CODE}"
    log "="*80
    log ""
    log "Check log file for details: ${LOG_FILE}"
    exit ${EXIT_CODE}
fi

log ""
log "Test complete. Log saved to: ${LOG_FILE}"

