#!/usr/bin/env bash
################################################################################
# Incremental Dataset Size Testing
################################################################################
# 
# This script tests the incremental_builder with progressively larger datasets
# to validate scalability and ensure output matches expected structure from
# previous 5000-gene production datasets.
#
# Reference dataset structure:
#   meta_spliceai/splice_engine/case_studies/data_sources/datasets/train_pc_5000_3mers_diverse
#
# Test sequence:
#   1. 100 genes   (~5-10 min)  - Quick validation
#   2. 500 genes   (~20-30 min) - Medium scale
#   3. 1000 genes  (~40-60 min) - Production-like
#
# Usage:
#   ./scripts/testing/test_incremental_sizes.sh [stage]
#
# Examples:
#   ./scripts/testing/test_incremental_sizes.sh         # Run all stages
#   ./scripts/testing/test_incremental_sizes.sh 100     # Only 100 genes
#   ./scripts/testing/test_incremental_sizes.sh 500     # Only 500 genes
#   ./scripts/testing/test_incremental_sizes.sh 1000    # Only 1000 genes
#
################################################################################

set -e  # Exit on error

# Get project root (script is in scripts/testing/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 INCREMENTAL DATASET SIZE TESTING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 Project root: ${PROJECT_ROOT}"
echo ""

# Configuration matching previous 5000-gene dataset
GENE_TYPES="protein_coding lncRNA"  # Diverse gene types
KMER_SIZES="3"                      # 3-mers (matching train_pc_5000_3mers_diverse)
BATCH_SIZE=100
BATCH_ROWS=500000
SUBSET_POLICY="random"              # For diversity

# Additional genes (ALS-related)
ADDITIONAL_GENES="${PROJECT_ROOT}/additional_genes.tsv"

# Timestamp for this test run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs/incremental_test"
mkdir -p "${LOG_DIR}"

# Determine which stages to run
STAGE_ARG="${1:-all}"

# Define test stages (using simple approach for compatibility)
get_stage_description() {
    case "$1" in
        100)  echo "100 genes - Quick validation (~5-10 min)" ;;
        500)  echo "500 genes - Medium scale (~20-30 min)" ;;
        1000) echo "1000 genes - Production-like (~40-60 min)" ;;
        *)    echo "Unknown stage" ;;
    esac
}

# Check conda environment (surveyor should already be active)
echo "🔧 Checking conda environment..."
if [ -n "${CONDA_DEFAULT_ENV}" ]; then
    echo "✅ Conda environment active: ${CONDA_DEFAULT_ENV}"
else
    echo "⚠️  No conda environment detected, attempting to activate surveyor..."
    if command -v mamba &> /dev/null; then
        eval "$(mamba shell.bash hook)" 2>/dev/null || true
        mamba activate surveyor 2>/dev/null || echo "⚠️  Could not activate via mamba"
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)" 2>/dev/null || true
        conda activate surveyor 2>/dev/null || echo "⚠️  Could not activate via conda"
    fi
    
    # Check if Python is available anyway (might be in PATH)
    if command -v python &> /dev/null; then
        echo "✅ Python found in PATH, proceeding..."
    else
        echo "❌ ERROR: Python not found! Please activate surveyor environment manually:"
        echo "   mamba activate surveyor"
        echo "   # Then run: $0 $@"
        exit 1
    fi
fi
echo ""

# Function to run a single test stage
run_test_stage() {
    local n_genes=$1
    local description=$2
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 STAGE: ${n_genes} GENES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Description: ${description}"
    echo "⏰ Start time: $(date)"
    echo ""
    
    # Output directory matching previous naming convention
    DATASET_DIR="data/train_pc_${n_genes}_3mers_diverse"
    LOG_FILE="${LOG_DIR}/test_${n_genes}genes_${TIMESTAMP}.log"
    
    echo "📋 Configuration:"
    echo "  Genes:          ${n_genes}"
    echo "  Gene types:     ${GENE_TYPES}"
    echo "  K-mer sizes:    ${KMER_SIZES}"
    echo "  Subset policy:  ${SUBSET_POLICY}"
    echo "  Output dir:     ${DATASET_DIR}"
    echo "  Log file:       ${LOG_FILE}"
    echo ""
    
    # Run incremental builder
    echo "🏗️  Running incremental_builder.py..."
    echo ""
    
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
        --n-genes "${n_genes}" \
        --subset-policy "${SUBSET_POLICY}" \
        --gene-types ${GENE_TYPES} \
        --gene-ids-file "${ADDITIONAL_GENES}" \
        --gene-col gene_id \
        --batch-size "${BATCH_SIZE}" \
        --batch-rows "${BATCH_ROWS}" \
        --run-workflow \
        --kmer-sizes ${KMER_SIZES} \
        --output-dir "${DATASET_DIR}" \
        --overwrite \
        --verbose 2>&1 | tee -a "${LOG_FILE}"
    
    BUILDER_EXIT_CODE=$?
    
    if [ ${BUILDER_EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "❌ ERROR: Dataset generation failed for ${n_genes} genes (exit code ${BUILDER_EXIT_CODE})"
        echo "📋 Check log file: ${LOG_FILE}"
        return 1
    fi
    
    echo ""
    echo "✅ Dataset generation completed for ${n_genes} genes!"
    echo ""
    
    # Verify and compare with expected structure
    verify_dataset_structure "${DATASET_DIR}" "${n_genes}" "${LOG_FILE}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ STAGE COMPLETE: ${n_genes} genes"
    echo "⏰ Completion time: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Function to verify dataset structure against expected format
verify_dataset_structure() {
    local dataset_dir=$1
    local n_genes=$2
    local log_file=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 VERIFYING DATASET STRUCTURE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    MASTER_DIR="${dataset_dir}/master"
    
    # Check master directory
    if [ ! -d "${MASTER_DIR}" ]; then
        echo "❌ ERROR: Master directory not found: ${MASTER_DIR}"
        return 1
    fi
    
    echo "✅ Master directory exists: ${MASTER_DIR}"
    
    # Count parquet files
    PARQUET_COUNT=$(ls -1 "${MASTER_DIR}"/*.parquet 2>/dev/null | wc -l)
    echo "✅ Parquet files: ${PARQUET_COUNT}"
    
    if [ ${PARQUET_COUNT} -eq 0 ]; then
        echo "❌ ERROR: No parquet files found in master directory!"
        return 1
    fi
    
    # Check for gene manifest
    if [ -f "${dataset_dir}/gene_manifest.csv" ]; then
        GENE_COUNT=$(tail -n +2 "${dataset_dir}/gene_manifest.csv" | wc -l)
        echo "✅ Gene manifest: ${GENE_COUNT} unique genes"
    else
        echo "⚠️  Gene manifest not found (may be normal for downsampled datasets)"
    fi
    
    # Detailed dataset inspection
    echo ""
    echo "📊 Dataset Analysis:"
    python << EOF 2>&1 | tee -a "${log_file}"
import polars as pl
import pyarrow.dataset as ds
from pathlib import Path
import json

dataset_dir = Path("${dataset_dir}")
master_dir = dataset_dir / "master"

print()
print("="*70)
print(f"Dataset: {dataset_dir.name}")
print("="*70)

# Load dataset
try:
    dataset = ds.dataset(master_dir, format="parquet")
    total_rows = dataset.count_rows()
    print(f"✅ Total rows: {total_rows:,}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Sample schema from first file
try:
    sample_file = list(master_dir.glob("*.parquet"))[0]
    df_sample = pl.read_parquet(sample_file, n_rows=10000)
    
    print()
    print(f"📝 Schema (from {sample_file.name}):")
    print(f"   Total columns: {len(df_sample.columns)}")
    
    # Key column groups
    key_cols = ['position', 'chrom', 'splice_type', 'donor_score', 'acceptor_score', 'neither_score']
    existing_keys = [col for col in key_cols if col in df_sample.columns]
    print(f"   Key columns: {', '.join(existing_keys)}")
    
    # K-mer features
    kmer_cols = [col for col in df_sample.columns if 'mer_' in col.lower()]
    print(f"   K-mer features: {len(kmer_cols)}")
    if kmer_cols:
        print(f"   K-mer examples: {', '.join(kmer_cols[:5])}")
    
    # Probability features
    prob_cols = [col for col in df_sample.columns if any(x in col.lower() for x in ['probability', 'ratio', 'diff', 'surge', 'peak'])]
    print(f"   Probability features: {len(prob_cols)}")
    if prob_cols:
        print(f"   Probability examples: {', '.join(prob_cols[:5])}")
    
    # Context features
    context_cols = [col for col in df_sample.columns if 'context' in col.lower()]
    print(f"   Context features: {len(context_cols)}")
    if context_cols:
        print(f"   Context examples: {', '.join(context_cols[:3])}")
    
    print()
    print("🏷️  Label Distribution (sample):")
    if 'splice_type' in df_sample.columns:
        label_dist = df_sample['splice_type'].value_counts()
        for row in label_dist.iter_rows():
            label, count = row
            pct = (count / len(df_sample)) * 100
            print(f"   {label}: {count:,} ({pct:.1f}%)")
    
    # Gene type distribution if available
    if 'gene_type' in df_sample.columns:
        print()
        print("🧬 Gene Type Distribution (sample):")
        gene_type_dist = df_sample['gene_type'].value_counts().head(5)
        for row in gene_type_dist.iter_rows():
            gene_type, count = row
            pct = (count / len(df_sample)) * 100
            print(f"   {gene_type}: {count:,} ({pct:.1f}%)")
    
    print()
    print("="*70)
    print(f"✅ Dataset structure matches expected format!")
    print("="*70)
    
except Exception as e:
    print(f"❌ Error analyzing dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF
    
    INSPECT_EXIT_CODE=$?
    
    if [ ${INSPECT_EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "❌ ERROR: Dataset inspection failed"
        return 1
    fi
    
    echo ""
    echo "✅ Dataset structure verification passed!"
}

# Main execution
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 TEST PLAN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "${STAGE_ARG}" == "all" ]; then
    echo "Running all stages:"
    for stage in 100 500 1000; do
        echo "  • ${stage} genes: $(get_stage_description ${stage})"
    done
    echo ""
    
    # Run all stages
    for stage in 100 500 1000; do
        run_test_stage "${stage}" "$(get_stage_description ${stage})"
    done
else
    # Run single stage
    stage_desc=$(get_stage_description "${STAGE_ARG}")
    if [ "$stage_desc" == "Unknown stage" ]; then
        echo "❌ ERROR: Unknown stage '${STAGE_ARG}'"
        echo "Valid stages: 100, 500, 1000, all"
        exit 1
    fi
    
    echo "Running single stage: ${STAGE_ARG} genes"
    echo "  ${stage_desc}"
    echo ""
    
    run_test_stage "${STAGE_ARG}" "${stage_desc}"
fi

# Final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 INCREMENTAL SIZE TESTING COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 Generated datasets:"
if [ "${STAGE_ARG}" == "all" ]; then
    for stage in "100" "500" "1000"; do
        dataset_dir="data/train_pc_${stage}_3mers_diverse"
        if [ -d "${dataset_dir}" ]; then
            total_size=$(du -sh "${dataset_dir}" | cut -f1)
            echo "  ✅ ${dataset_dir} (${total_size})"
        fi
    done
else
    dataset_dir="data/train_pc_${STAGE_ARG}_3mers_diverse"
    if [ -d "${dataset_dir}" ]; then
        total_size=$(du -sh "${dataset_dir}" | cut -f1)
        echo "  ✅ ${dataset_dir} (${total_size})"
    fi
fi

echo ""
echo "📋 Logs:"
ls -lht "${LOG_DIR}"/*.log | head -5

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📖 NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Compare dataset structures:"
echo "   ls -lh data/train_pc_*/master/"
echo ""
echo "2. Inspect gene manifests:"
echo "   head data/train_pc_*/gene_manifest.csv"
echo ""
echo "3. Train meta-models on each dataset:"
echo "   for size in 100 500 1000; do"
echo "     python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \\"
echo "       --dataset data/train_pc_\${size}_3mers_diverse/master \\"
echo "       --out-dir data/model_\${size}genes \\"
echo "       --n-folds 5 \\"
echo "       --n-estimators 800"
echo "   done"
echo ""
echo "4. Compare model performance across dataset sizes"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

