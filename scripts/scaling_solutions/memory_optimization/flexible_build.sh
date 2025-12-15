#!/bin/bash

# Flexible incremental builder that reuses directories
# Supports different dataset sizes with appropriate memory settings

set -e

# Default configuration
DEFAULT_GENES_FILE="additional_genes.tsv"
DEFAULT_OUTPUT_DIR="train_dataset"
DEFAULT_BATCH_SIZE=200
DEFAULT_BATCH_ROWS=15000
DEFAULT_KMER_SIZE=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --genes)
            GENES_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n-genes)
            N_GENES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --batch-rows)
            BATCH_ROWS="$2"
            shift 2
            ;;
        --kmer-size|--kmer-sizes)
            KMER_SIZE="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --auto)
            AUTO=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --genes FILE          Gene list file (optional, default: $DEFAULT_GENES_FILE)"
            echo "  --output-dir DIR      Output directory (default: $DEFAULT_OUTPUT_DIR)"
            echo "  --n-genes N           Number of genes to process"
            echo "  --batch-size N        Genes per batch (default: $DEFAULT_BATCH_SIZE)"
            echo "  --batch-rows N        Rows per batch (default: $DEFAULT_BATCH_ROWS)"
            echo "  --kmer-size N         K-mer size (default: $DEFAULT_KMER_SIZE)"
            echo "  --kmer-sizes N        K-mer size (alias for --kmer-size)"
            echo "  --strategy STRATEGY   conservative|moderate|aggressive"
            echo "  --overwrite           Force overwrite existing dataset"
            echo "  --resume              Resume from existing batches"
            echo "  --auto                Auto-detect settings based on gene count"
            echo ""
            echo "Examples:"
            echo "  $0 --n-genes 1000 --strategy conservative"
            echo "  $0 --n-genes 5000 --strategy moderate --overwrite"
            echo "  $0 --n-genes 10000 --strategy aggressive --auto"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults
GENES_FILE=${GENES_FILE:-$DEFAULT_GENES_FILE}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
BATCH_ROWS=${BATCH_ROWS:-$DEFAULT_BATCH_ROWS}
KMER_SIZE=${KMER_SIZE:-$DEFAULT_KMER_SIZE}

# Auto-detect settings if requested
if [[ "$AUTO" == "true" && -n "$N_GENES" ]]; then
    echo "Auto-detecting settings for $N_GENES genes..."
    
    if [[ $N_GENES -le 5000 ]]; then
        STRATEGY="conservative"
        BATCH_SIZE=100
        BATCH_ROWS=10000
    elif [[ $N_GENES -le 10000 ]]; then
        STRATEGY="moderate"
        BATCH_SIZE=200
        BATCH_ROWS=15000
    else
        STRATEGY="aggressive"
        BATCH_SIZE=500
        BATCH_ROWS=25000
    fi
    
    echo "Auto-selected strategy: $STRATEGY"
fi

# Apply strategy-specific settings
case "${STRATEGY:-}" in
    "conservative")
        BATCH_SIZE=50
        BATCH_ROWS=5000
        KMER_SIZE=3
        export POLARS_MAX_THREADS=2
        echo "Using conservative settings (low memory usage)"
        ;;
    "moderate")
        BATCH_SIZE=200
        BATCH_ROWS=15000
        KMER_SIZE=3
        export POLARS_MAX_THREADS=4
        echo "Using moderate settings (balanced performance)"
        ;;
    "aggressive")
        BATCH_SIZE=500
        BATCH_ROWS=25000
        KMER_SIZE=3
        export POLARS_MAX_THREADS=8
        echo "Using aggressive settings (high performance)"
        ;;
esac

# Check if genes file exists (only if specified)
if [[ -n "$GENES_FILE" && ! -f "$GENES_FILE" ]]; then
    echo "Error: Gene file $GENES_FILE not found"
    exit 1
fi

# Handle existing dataset
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Found existing dataset: $OUTPUT_DIR"
    
    if [[ "$OVERWRITE" == "true" ]]; then
        echo "Overwriting existing dataset..."
        rm -rf "$OUTPUT_DIR"
    elif [[ "$RESUME" == "true" ]]; then
        echo "Resuming from existing batches..."
        # Don't remove directory, let incremental builder handle resumption
    else
        echo "Options:"
        echo "  1. Overwrite existing dataset"
        echo "  2. Resume from existing batches"
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
fi

# Set memory-friendly environment variables
export ARROW_PRETTY_PRINT=0
export PYTHONHASHSEED=0

# Count additional genes if gene file exists
ADDITIONAL_GENE_COUNT=0
TOTAL_EXPECTED_GENES=$N_GENES

if [[ -n "$GENES_FILE" && -f "$GENES_FILE" ]]; then
    # Count lines in gene file (excluding header if present)
    if head -1 "$GENES_FILE" | grep -q "gene_id\|Gene\|GENE"; then
        # Has header, count lines minus 1
        ADDITIONAL_GENE_COUNT=$(( $(wc -l < "$GENES_FILE") - 1 ))
    else
        # No header, count all non-empty lines
        ADDITIONAL_GENE_COUNT=$(grep -c "^[[:space:]]*[^[:space:]]" "$GENES_FILE" 2>/dev/null || echo 0)
    fi
    
    if [[ $ADDITIONAL_GENE_COUNT -gt 0 ]]; then
        TOTAL_EXPECTED_GENES=$(( N_GENES + ADDITIONAL_GENE_COUNT ))
    fi
fi

# Display configuration
echo ""
echo "=== Build Configuration ==="
echo "Genes file: $GENES_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Base genes (via error_total policy): $N_GENES"
if [[ $ADDITIONAL_GENE_COUNT -gt 0 ]]; then
    echo "Additional genes (from file): $ADDITIONAL_GENE_COUNT"
    echo "Total expected genes: $TOTAL_EXPECTED_GENES"
else
    echo "Total expected genes: $N_GENES"
fi
echo "Batch size: $BATCH_SIZE"
echo "Batch rows: $BATCH_ROWS"
echo "K-mer size: $KMER_SIZE"
echo "Strategy: ${STRATEGY:-custom}"
echo "Polars threads: $POLARS_MAX_THREADS"
echo "=========================="
echo ""

# Check system resources
echo "=== System Resources ==="
RAM_GB=$(free -g | grep Mem | awk '{print $2}')
AVAIL_GB=$(free -g | grep Mem | awk '{print $7}')
echo "Total RAM: ${RAM_GB}GB"
echo "Available RAM: ${AVAIL_GB}GB"

# Estimate memory requirements based on total expected genes
ESTIMATED_PEAK=$((BATCH_SIZE * BATCH_ROWS * KMER_SIZE * 4 / 1000000))  # Rough estimate in GB
echo "Estimated peak memory: ~${ESTIMATED_PEAK}GB (based on batch size)"

if [[ $ESTIMATED_PEAK -gt $AVAIL_GB ]]; then
    echo "⚠️  Warning: Estimated memory usage may exceed available RAM"
    echo "   Consider using --strategy conservative or adding swap space"
fi

# Check swap
SWAP_GB=$(free -g | grep Swap | awk '{print $2}')
if [[ $SWAP_GB -eq 0 ]]; then
    echo "⚠️  No swap space detected. Consider adding swap for large datasets:"
    echo "   ./scripts/scaling_solutions/memory_optimization/swap_setup.sh"
fi

echo "========================"
echo ""

# Confirm before proceeding
if [[ "$AUTO" != "true" ]]; then
    read -p "Proceed with build? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Build cancelled."
        exit 0
    fi
fi

# Build command
CMD="python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes $N_GENES \
    --subset-policy error_total \
    --batch-size $BATCH_SIZE \
    --batch-rows $BATCH_ROWS \
    --run-workflow \
    --kmer-sizes $KMER_SIZE \
    --output-dir $OUTPUT_DIR \
    --overwrite -v"

# Add gene file parameters only if gene file is specified and exists
if [[ -n "$GENES_FILE" && -f "$GENES_FILE" ]]; then
    CMD="$CMD --gene-ids-file $GENES_FILE --gene-col gene_id"
fi

echo "Starting build..."
echo "Command: $CMD"
echo ""

# Execute the build
eval $CMD

# Check result
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Build completed successfully!"
    echo "Dataset location: $OUTPUT_DIR"
    
    # Show final dataset size
    if [[ -d "$OUTPUT_DIR/master" ]]; then
        echo "Final dataset size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
        echo "Number of batch files: $(ls -1 "$OUTPUT_DIR/master"/*.parquet 2>/dev/null | wc -l)"
    fi
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the output above for error details."
    echo "You can resume the build with:"
    echo "  $0 --n-genes $N_GENES --resume"
    exit 1
fi 