#!/bin/bash
#
# Sequential Chromosome Processing - OpenSpliceAI
#
# This script processes all chromosomes (1-22, X, Y) sequentially.
# Each chromosome runs in its own process to minimize memory usage.
#
# Expected total runtime: ~5 days (sequential)
# Memory usage per chromosome: ~2-4GB
#

set -e  # Exit on error

cd /Users/pleiadian53/work/meta-spliceai
source ~/.bash_profile
mamba activate surveyor

# Configuration
BASE_MODEL="openspliceai"
MODE="production"
COVERAGE="full_genome"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Summary file
SUMMARY_FILE="${LOG_DIR}/all_chromosomes_summary_${TIMESTAMP}.txt"

echo "=========================================" | tee "$SUMMARY_FILE"
echo "Sequential Chromosome Processing" | tee -a "$SUMMARY_FILE"
echo "=========================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Base Model: $BASE_MODEL" | tee -a "$SUMMARY_FILE"
echo "Mode: $MODE" | tee -a "$SUMMARY_FILE"
echo "Coverage: $COVERAGE" | tee -a "$SUMMARY_FILE"
echo "Start Time: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Process each chromosome
CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y)
TOTAL=${#CHROMOSOMES[@]}
SUCCESS_COUNT=0
FAILED_CHROMS=()

for i in "${!CHROMOSOMES[@]}"; do
    CHR="${CHROMOSOMES[$i]}"
    CHR_NUM=$((i + 1))
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "=========================================" | tee -a "$SUMMARY_FILE"
    echo "[$CHR_NUM/$TOTAL] Processing Chromosome $CHR" | tee -a "$SUMMARY_FILE"
    echo "=========================================" | tee -a "$SUMMARY_FILE"
    echo "Start: $(date)" | tee -a "$SUMMARY_FILE"
    
    CHR_START_TIME=$(date +%s)
    CHR_LOG="${LOG_DIR}/${BASE_MODEL}_chr${CHR}_${TIMESTAMP}.log"
    
    # Run chromosome
    if python scripts/training/run_full_genome_base_model_pass.py \
        --base-model "$BASE_MODEL" \
        --mode "$MODE" \
        --coverage "$COVERAGE" \
        --chromosomes "$CHR" \
        --verbosity 1 \
        2>&1 | tee "$CHR_LOG"; then
        
        CHR_END_TIME=$(date +%s)
        CHR_DURATION=$((CHR_END_TIME - CHR_START_TIME))
        CHR_DURATION_MIN=$((CHR_DURATION / 60))
        
        echo "✅ Chromosome $CHR completed successfully" | tee -a "$SUMMARY_FILE"
        echo "   Duration: ${CHR_DURATION_MIN} minutes" | tee -a "$SUMMARY_FILE"
        echo "   End: $(date)" | tee -a "$SUMMARY_FILE"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Chromosome $CHR FAILED" | tee -a "$SUMMARY_FILE"
        echo "   Check log: $CHR_LOG" | tee -a "$SUMMARY_FILE"
        FAILED_CHROMS+=("$CHR")
        
        # Ask user if they want to continue
        read -p "Continue with remaining chromosomes? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting..." | tee -a "$SUMMARY_FILE"
            exit 1
        fi
    fi
    
    # Show progress
    REMAINING=$((TOTAL - CHR_NUM))
    echo "" | tee -a "$SUMMARY_FILE"
    echo "Progress: $SUCCESS_COUNT/$TOTAL chromosomes completed successfully" | tee -a "$SUMMARY_FILE"
    echo "Remaining: $REMAINING chromosomes" | tee -a "$SUMMARY_FILE"
done

# Final summary
echo "" | tee -a "$SUMMARY_FILE"
echo "=========================================" | tee -a "$SUMMARY_FILE"
echo "FINAL SUMMARY" | tee -a "$SUMMARY_FILE"
echo "=========================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Total Chromosomes: $TOTAL" | tee -a "$SUMMARY_FILE"
echo "Successful: $SUCCESS_COUNT" | tee -a "$SUMMARY_FILE"
echo "Failed: ${#FAILED_CHROMS[@]}" | tee -a "$SUMMARY_FILE"

if [ ${#FAILED_CHROMS[@]} -gt 0 ]; then
    echo "Failed Chromosomes: ${FAILED_CHROMS[*]}" | tee -a "$SUMMARY_FILE"
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "End Time: $(date)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

if [ $SUCCESS_COUNT -eq $TOTAL ]; then
    echo "🎉 All chromosomes processed successfully!" | tee -a "$SUMMARY_FILE"
    exit 0
else
    echo "⚠️  Some chromosomes failed. Check logs for details." | tee -a "$SUMMARY_FILE"
    exit 1
fi

