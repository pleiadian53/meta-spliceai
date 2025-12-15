#!/bin/bash
# Enhanced Sequential Chromosome Processing with Smart Checkpointing
# Automatically detects and skips chromosomes that have already been processed
#
# Usage:
#   # Background with monitoring:
#   caffeinate -i nohup bash scripts/training/process_chromosomes_sequential_smart.sh 2>&1 | tee logs/full_genome_$(date +%Y%m%d_%H%M%S).log &
#
#   # Foreground:
#   bash scripts/training/process_chromosomes_sequential_smart.sh
#
#   # Force reprocess specific chromosome (bypass checkpoint):
#   FORCE_CHROMOSOMES="2,3" bash scripts/training/process_chromosomes_sequential_smart.sh

set -e  # Exit on error

# Activate environment
source ~/.bash_profile 2>/dev/null || true
mamba activate metaspliceai 2>/dev/null || conda activate metaspliceai || true

# Change to project directory
cd "$(dirname "$0")/../.." || exit 1
PROJECT_ROOT=$(pwd)

# Configuration
BASE_MODEL="openspliceai"
MODE="production"
COVERAGE="full_genome"
ARTIFACT_DIR="${PROJECT_ROOT}/data/mane/GRCh38/openspliceai_eval/meta_models"

# Set tqdm to update less frequently for cleaner logs
# Updates every 5 minutes instead of constantly (good for long-running background processes)
export TQDM_MININTERVAL=300
export TQDM_MAXINTERVAL=600

# All chromosomes to process
ALL_CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y)

# Function to check if chromosome is complete
is_chromosome_complete() {
    local chr=$1
    local artifact_dir=$2
    
    # NOTE: We no longer do chromosome-level completion checking here.
    # The Python workflow now has chunk-level checkpointing that automatically
    # detects and skips completed chunks within a chromosome.
    # This allows partial chromosome resumption (e.g., Chr2 with 500/1216 genes done).
    
    # Always return "incomplete" so Python workflow can handle chunk-level detection
    return 1  # Let Python workflow handle checkpoint detection
}

# Function to get completion status
get_completion_details() {
    local chr=$1
    local artifact_dir=$2
    local pattern="${artifact_dir}/analysis_sequences_${chr}_*.tsv"
    
    local files=($(ls ${pattern} 2>/dev/null || true))
    local num_files=${#files[@]}
    
    if [ $num_files -eq 0 ]; then
        echo "Not started"
    else
        # Extract gene counts from filenames (e.g., analysis_sequences_1_chunk_1_500.tsv)
        local total_genes=0
        for file in "${files[@]}"; do
            # Extract the end gene number from filename
            if [[ $(basename "$file") =~ _([0-9]+)\.tsv$ ]]; then
                local end_gene="${BASH_REMATCH[1]}"
                if [ "$end_gene" -gt "$total_genes" ]; then
                    total_genes=$end_gene
                fi
            fi
        done
        
        if [ $total_genes -gt 0 ]; then
            echo "Complete (~${total_genes} genes in ${num_files} chunk(s))"
        else
            echo "Complete (${num_files} chunk(s))"
        fi
    fi
}

echo "============================================================"
echo "Smart Sequential Chromosome Processing"
echo "============================================================"
echo "Base Model: ${BASE_MODEL}"
echo "Mode: ${MODE}"
echo "Coverage: ${COVERAGE}"
echo "Artifact Directory: ${ARTIFACT_DIR}"
echo "Start: $(date)"
echo "============================================================"
echo ""

# Show status of existing artifacts (for informational purposes only)
echo "=========================================="
echo "Existing Artifacts Detection"
echo "=========================================="
echo "Note: Chunk-level checkpointing is active."
echo "      Already-processed chunks will be automatically skipped."
echo ""

for chr in "${ALL_CHROMOSOMES[@]}"; do
    details=$(get_completion_details "$chr" "$ARTIFACT_DIR")
    if [ "$details" != "Not started" ]; then
        echo "  Chr${chr}: ${details} (will skip completed chunks)"
    else
        echo "  Chr${chr}: Not started"
    fi
done

# Process all chromosomes (chunk-level checkpointing will skip completed chunks)
CHROMOSOMES_TO_PROCESS=("${ALL_CHROMOSOMES[@]}")

# Check if user wants to force specific chromosomes
if [ -n "${FORCE_CHROMOSOMES}" ]; then
    echo ""
    echo "FORCE MODE: Only processing chromosomes: ${FORCE_CHROMOSOMES}"
    IFS=',' read -ra CHROMOSOMES_TO_PROCESS <<< "$FORCE_CHROMOSOMES"
fi

echo ""
echo "Will process: ${CHROMOSOMES_TO_PROCESS[*]}"
echo "(Chunk-level checkpointing will automatically skip completed chunks)"
echo ""
read -t 10 -p "Press Enter to continue or Ctrl+C to cancel (auto-continuing in 10s)..." || echo ""

# Track progress
TOTAL=${#CHROMOSOMES_TO_PROCESS[@]}
SUCCESS_COUNT=0
FAILED_CHROMS=()

# Process each chromosome
for i in "${!CHROMOSOMES_TO_PROCESS[@]}"; do
    CHR="${CHROMOSOMES_TO_PROCESS[$i]}"
    CHR_NUM=$((i + 1))
    
    echo ""
    echo "========================================="
    echo "[$CHR_NUM/$TOTAL] Processing Chromosome $CHR"
    echo "========================================="
    echo "Start: $(date)"
    
    CHR_START=$(date +%s)
    
    # Run using CLI
    if run_base_model \
        --base-model "$BASE_MODEL" \
        --mode "$MODE" \
        --coverage "$COVERAGE" \
        --chromosomes "$CHR" \
        --verbosity 1; then
        
        CHR_END=$(date +%s)
        CHR_DURATION=$((CHR_END - CHR_START))
        CHR_DURATION_MIN=$((CHR_DURATION / 60))
        
        echo ""
        echo "✅ Chromosome $CHR completed successfully"
        echo "   Duration: ${CHR_DURATION_MIN} minutes"
        echo "   End: $(date)"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Verify artifacts were created
        if is_chromosome_complete "$CHR" "$ARTIFACT_DIR"; then
            details=$(get_completion_details "$CHR" "$ARTIFACT_DIR")
            echo "   Artifacts verified: ${details}"
        else
            echo "   ⚠️  Warning: Artifacts not found for chr${CHR}"
        fi
    else
        echo ""
        echo "❌ Chromosome $CHR FAILED"
        echo "   Check log for details."
        FAILED_CHROMS+=("$CHR")
        
        # Decide whether to continue
        if [[ -t 0 ]]; then
            read -p "Continue with remaining chromosomes? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Aborting..."
                break
            fi
        else
            echo "Non-interactive mode: Continuing with next chromosome after failure."
        fi
    fi
    
    # Progress summary
    REMAINING=$((TOTAL - CHR_NUM))
    echo ""
    echo "Progress: $SUCCESS_COUNT/$TOTAL completed, $REMAINING remaining"
done

# Final summary
echo ""
echo "========================================="
echo "FINAL SUMMARY"
echo "========================================="
echo ""
echo "Total chromosomes targeted: $TOTAL"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: ${#FAILED_CHROMS[@]}"
if [ ${#FAILED_CHROMS[@]} -gt 0 ]; then
    echo "  Failed chromosomes: ${FAILED_CHROMS[*]}"
fi
echo ""
echo "Note: Chunk-level checkpointing was active."
echo "      Completed chunks were automatically skipped."
echo ""
echo "End Time: $(date)"

if [ $SUCCESS_COUNT -eq $TOTAL ]; then
    echo ""
    echo "🎉 All targeted chromosomes processed successfully!"
    exit 0
else
    echo ""
    echo "⚠️  Some chromosomes failed. Check logs for details."
    exit 1
fi

