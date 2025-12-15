#!/bin/bash
# Run base model pass for a single chromosome using the CLI
# Usage: bash scripts/training/run_chromosome_cli.sh <chromosome>

set -e  # Exit on error

# Activate environment
source ~/.bash_profile 2>/dev/null || true
mamba activate metaspliceai 2>/dev/null || conda activate metaspliceai || true

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <chromosome>"
    echo "Example: $0 21"
    exit 1
fi

CHR=$1

# Add chr prefix if not present
if [[ ! "$CHR" =~ ^chr ]]; then
    CHR_ARG="$CHR"
else
    CHR_ARG="${CHR#chr}"
fi

echo "============================================================"
echo "Starting base model pass for chromosome $CHR_ARG"
echo "============================================================"
echo "Command: run_base_model --base-model openspliceai --chromosomes $CHR_ARG --mode production"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Run the CLI command
run_base_model \
    --base-model openspliceai \
    --chromosomes "$CHR_ARG" \
    --mode production

echo ""
echo "============================================================"
echo "Chromosome $CHR_ARG complete!"
echo "End time: $(date)"
echo "============================================================"

