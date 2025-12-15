#!/bin/bash

# Ultra-conservative memory settings for incremental builder
# Designed for systems with limited RAM or no swap

set -e  # Exit on any error

echo "=== Low Memory Build Strategy ==="
echo "System memory: $(free -h | grep Mem: | awk '{print $2}')"
echo "Available memory: $(free -h | grep Mem: | awk '{print $7}')"

# Configuration
GENES_FILE="additional_genes.tsv"
OUTPUT_DIR="train_pc_1000_3mers"   # Reuse existing directory
N_GENES=1000
BATCH_SIZE=50      # Very small batches
BATCH_ROWS=5000    # Very small row chunks
KMER_SIZE=3        # Memory-efficient k-mer size

echo "Configuration:"
echo "  Genes file: $GENES_FILE"
echo "  Output dir: $OUTPUT_DIR" 
echo "  Total genes: $N_GENES"
echo "  Batch size: $BATCH_SIZE"
echo "  Batch rows: $BATCH_ROWS"
echo "  K-mer size: $KMER_SIZE"

# Check if genes file exists
if [[ ! -f "$GENES_FILE" ]]; then
    echo "Error: Gene file $GENES_FILE not found"
    exit 1
fi

# Check existing dataset and provide options
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Found existing dataset: $OUTPUT_DIR"
    echo "Options:"
    echo "  1. Overwrite existing dataset (recommended for failed builds)"
    echo "  2. Resume from existing batches (if build was interrupted)"
    echo "  3. Create new directory with timestamp"
    
    read -p "Choose option (1/2/3) [default: 1]: " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            echo "Overwriting existing dataset..."
            rm -rf "$OUTPUT_DIR"
            ;;
        2)
            echo "Resuming from existing batches..."
            # Don't remove directory, let incremental builder handle resumption
            ;;
        3)
            timestamp=$(date +%Y%m%d_%H%M%S)
            OUTPUT_DIR="${OUTPUT_DIR}_${timestamp}"
            echo "Creating new directory: $OUTPUT_DIR"
            ;;
        *)
            echo "Invalid choice, overwriting existing dataset..."
    rm -rf "$OUTPUT_DIR"
            ;;
    esac
fi

# Set memory-friendly environment variables
export POLARS_MAX_THREADS=2           # Limit Polars parallelism
export ARROW_PRETTY_PRINT=0           # Reduce verbose arrow output
export PYTHONHASHSEED=0               # Consistent hashing

echo ""
echo "Starting incremental builder with conservative settings..."
echo "Command will be:"

CMD="python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes $N_GENES \
    --subset-policy error_total \
    --gene-ids-file $GENES_FILE \
    --gene-col gene_id \
    --batch-size $BATCH_SIZE \
    --batch-rows $BATCH_ROWS \
    --run-workflow \
    --kmer-sizes $KMER_SIZE \
    --output-dir $OUTPUT_DIR \
    --overwrite -v"

echo "$CMD"
echo ""

# Monitor memory usage during execution
echo "Starting memory monitoring..."
(
    while true; do
        if pgrep -f "incremental_builder" > /dev/null; then
            RSS=$(ps -o rss= -p $(pgrep -f "incremental_builder") 2>/dev/null | awk '{sum+=$1} END {print sum/1024}')
            AVAIL=$(free -m | grep Mem: | awk '{print $7}')
            echo "$(date): RSS=${RSS}MB, Available=${AVAIL}MB" >> memory_monitor.log
        fi
        sleep 10
    done
) &
MONITOR_PID=$!

# Trap to clean up monitor on exit
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# Run the actual command
eval $CMD

# Check success
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✓ Build completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
    echo "Memory usage log: memory_monitor.log"
    
    # Show final dataset size
    if [[ -d "$OUTPUT_DIR/master" ]]; then
        echo "Final dataset files:"
        ls -lah "$OUTPUT_DIR/master/"
        echo "Total size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
    fi
else
    echo ""
    echo "✗ Build failed!"
    echo "Check memory_monitor.log for memory usage patterns"
    exit 1
fi 