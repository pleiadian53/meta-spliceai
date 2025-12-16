#!/bin/bash

# Incremental Builder Tmux Runner
# Usage: ./run_builder.sh [session_name] [output_dir] [n_genes]

set -e

# Default values
DEFAULT_SESSION_NAME="builder_$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_DIR="train_pc_1000_plus_custom_3mers"
DEFAULT_N_GENES=1000
DEFAULT_GENE_FILE="additional_genes.tsv"

# Parse arguments
SESSION_NAME=${1:-$DEFAULT_SESSION_NAME}
OUTPUT_DIR=${2:-$DEFAULT_OUTPUT_DIR}
N_GENES=${3:-$DEFAULT_N_GENES}
GENE_FILE=${4:-$DEFAULT_GENE_FILE}

# Create log file name
LOG_FILE="builder_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "❌ Session '$SESSION_NAME' already exists!"
    echo "Choose a different name or kill the existing session:"
    echo "   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Check if conda environment exists
if ! conda env list | grep -q "surveyor"; then
    echo "❌ Conda environment 'surveyor' not found!"
    echo "Please create the surveyor environment first."
    exit 1
fi

# Check if gene file exists (if specified)
if [ "$GENE_FILE" != "additional_genes.tsv" ] || [ -f "$GENE_FILE" ]; then
    echo "✅ Gene file: $GENE_FILE"
else
    echo "⚠️  Gene file '$GENE_FILE' not found. Using default gene selection."
    GENE_FILE=""
fi

# Build command
BUILDER_CMD="conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder"
BUILDER_CMD="$BUILDER_CMD --n-genes $N_GENES"
BUILDER_CMD="$BUILDER_CMD --subset-policy error_total"

# Add gene file if specified
if [ -n "$GENE_FILE" ]; then
    BUILDER_CMD="$BUILDER_CMD --gene-ids-file $GENE_FILE --gene-col gene_id"
fi

BUILDER_CMD="$BUILDER_CMD --batch-size 250"
BUILDER_CMD="$BUILDER_CMD --batch-rows 20000"
BUILDER_CMD="$BUILDER_CMD --run-workflow"
BUILDER_CMD="$BUILDER_CMD --kmer-sizes 3"
BUILDER_CMD="$BUILDER_CMD --output-dir $OUTPUT_DIR"
BUILDER_CMD="$BUILDER_CMD --overwrite"
BUILDER_CMD="$BUILDER_CMD -v"
BUILDER_CMD="$BUILDER_CMD 2>&1 | tee $LOG_FILE"

echo "🚀 Starting Incremental Builder"
echo "================================"
echo "Session name: $SESSION_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Number of genes: $N_GENES"
echo "Gene file: ${GENE_FILE:-'(default selection)'}"
echo "Log file: $LOG_FILE"
echo ""

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -x $(tput cols) -y $(tput lines) "$BUILDER_CMD"

echo "✅ Session created successfully!"
echo ""
echo "📋 Useful Commands:"
echo "  • Attach to session:    tmux a -t $SESSION_NAME"
echo "  • Check status:         tmux ls"
echo "  • Monitor log:          tail -f $LOG_FILE"
echo "  • Monitor output:       ls -la $OUTPUT_DIR/master/"
echo "  • Kill session:         tmux kill-session -t $SESSION_NAME"
echo ""
echo "🔍 To monitor progress:"
echo "  • tmux a -t $SESSION_NAME"
echo "  • Press Ctrl+B then D to detach"
echo ""
echo "📁 Output will be saved to: $OUTPUT_DIR/"
echo "📄 Logs will be saved to: $LOG_FILE" 