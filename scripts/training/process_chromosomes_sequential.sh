#!/bin/bash
# Process chromosomes sequentially using run_base_model CLI
# Designed for memory-constrained systems (M1 MacBook Pro 16GB)
#
# Usage:
#   # Background with monitoring:
#   nohup bash scripts/training/process_chromosomes_sequential.sh 2>&1 | tee logs/full_genome_$(date +%Y%m%d_%H%M%S).log &
#
#   # Foreground:
#   bash scripts/training/process_chromosomes_sequential.sh

set -e  # Exit on error

# Activate environment
source ~/.bash_profile 2>/dev/null || true
mamba activate metaspliceai 2>/dev/null || conda activate metaspliceai || true

# Set tqdm to update less frequently for cleaner logs
# Updates every 5 minutes instead of constantly (good for long-running background processes)
export TQDM_MININTERVAL=300
export TQDM_MAXINTERVAL=600

# Configuration
CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 X Y)  # Skip 21 (already done)
TOTAL=${#CHROMOSOMES[@]}

echo "============================================================"
echo "Sequential Chromosome Processing (CLI)"
echo "============================================================"
echo "Base Model: OpenSpliceAI"
echo "Mode: Production"
echo "Chromosomes: ${TOTAL} (skipping chr21 - already complete)"
echo "Memory: Optimized for 16GB M1 MacBook Pro"
echo "Start: $(date)"
echo "============================================================"
echo ""

# Track progress
SUCCESS_COUNT=0
FAILED_CHROMS=()

# Process each chromosome
for i in "${!CHROMOSOMES[@]}"; do
    CHR="${CHROMOSOMES[$i]}"
    CHR_NUM=$((i + 1))
    
    echo ""
    echo "========================================"
    echo "[$CHR_NUM/$TOTAL] Chromosome $CHR"
    echo "========================================"
    echo "Start: $(date)"
    
    CHR_START=$(date +%s)
    
    # Run using CLI
    if run_base_model \
        --base-model openspliceai \
        --chromosomes "$CHR" \
        --mode production; then
        
        CHR_END=$(date +%s)
        DURATION=$(( (CHR_END - CHR_START) / 60 ))
        
        echo "✅ Chr $CHR complete (${DURATION} min)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Chr $CHR FAILED"
        FAILED_CHROMS+=("$CHR")
        
        # Ask if user wants to continue
        echo ""
        read -p "Continue with remaining chromosomes? [y/N] " -t 30 -n 1 -r || REPLY='y'
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting..."
            exit 1
        fi
    fi
    
    # Progress
    REMAINING=$((TOTAL - CHR_NUM))
    echo ""
    echo "Progress: $SUCCESS_COUNT/$TOTAL completed, $REMAINING remaining"
done

# Final summary
echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Successful: $SUCCESS_COUNT/$TOTAL"

if [ ${#FAILED_CHROMS[@]} -gt 0 ]; then
    echo "Failed: ${FAILED_CHROMS[*]}"
    exit 1
else
    echo "✅ All chromosomes processed successfully!"
    echo ""
    echo "Next step: Build meta-learning training dataset"
    echo "  python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \\"
    echo "    --n-genes 5000 \\"
    echo "    --output-dir train_dataset"
    exit 0
fi

