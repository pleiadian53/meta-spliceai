#!/bin/bash
# Run base model pass for all chromosomes using the CLI
# Skips chromosome 21 (already done)
# Usage: bash scripts/training/run_all_chromosomes_cli.sh

set -e  # Exit on error

# Activate environment
source ~/.bash_profile 2>/dev/null || true
mamba activate metaspliceai 2>/dev/null || conda activate metaspliceai || true

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define chromosomes (excluding chr21 which is already done)
CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 X Y)

echo "============================================================"
echo "Full Genome Base Model Pass (CLI Version)"
echo "============================================================"
echo "Base Model: OpenSpliceAI"
echo "Mode: Production"
echo "Chromosomes: ${#CHROMOSOMES[@]}"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Process each chromosome
for CHR in "${CHROMOSOMES[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing Chromosome $CHR"
    echo "========================================"
    
    # Run single chromosome script
    bash "$SCRIPT_DIR/run_chromosome_cli.sh" "$CHR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Chromosome $CHR completed successfully"
    else
        echo "❌ Chromosome $CHR failed"
        exit 1
    fi
    
    echo ""
done

echo ""
echo "============================================================"
echo "✅ ALL CHROMOSOMES COMPLETE!"
echo "============================================================"
echo "End time: $(date)"
echo "============================================================"

