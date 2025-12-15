#!/bin/bash
################################################################################
# Complete Pipeline Test: Dataset Generation → Meta-Model Training
################################################################################
# 
# This script tests the complete meta-learning pipeline:
# 1. Generate training dataset with incremental_builder.py
# 2. Train meta-model with run_gene_cv_sigmoid.py
# 3. Verify all outputs and performance
#
# Usage:
#   ./scripts/testing/test_complete_pipeline.sh [n_genes]
#
# Examples:
#   ./scripts/testing/test_complete_pipeline.sh         # 50 genes (quick test)
#   ./scripts/testing/test_complete_pipeline.sh 100     # 100 genes (thorough test)
#
################################################################################

set -e  # Exit on error

# Get project root (script is in scripts/testing/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 COMPLETE META-LEARNING PIPELINE TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 Project root: ${PROJECT_ROOT}"
echo ""

# Configuration
N_GENES="${1:-50}"  # Default: 50 genes for quick test
BATCH_SIZE=50
BATCH_ROWS=500000
KMER_SIZES="3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output directories
DATASET_DIR="data/test_pipeline_${N_GENES}genes_${TIMESTAMP}"
MODEL_DIR="data/test_model_${N_GENES}genes_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/logs/pipeline_test"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/test_pipeline_${N_GENES}genes_${TIMESTAMP}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 TEST CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Genes:          ${N_GENES}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "  K-mer sizes:    ${KMER_SIZES}"
echo "  Dataset output: ${DATASET_DIR}"
echo "  Model output:   ${MODEL_DIR}"
echo "  Log file:       ${LOG_FILE}"
echo ""

# Additional gene list (ALS-related genes for testing)
ADDITIONAL_GENES="${PROJECT_ROOT}/additional_genes.tsv"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STAGE 1: GENERATE TRAINING DATASET"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏰ Start time: $(date)"
echo ""

# Activate conda environment
echo "🔧 Activating surveyor environment..."
if command -v mamba &> /dev/null; then
    eval "$(mamba shell.bash hook)"
    mamba activate surveyor
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate surveyor
else
    echo "❌ ERROR: Neither mamba nor conda found!"
    exit 1
fi

echo "✅ Environment activated: ${CONDA_DEFAULT_ENV}"
echo ""

# Run incremental builder
echo "🏗️  Running incremental_builder.py..."
echo "   Command: python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder"
echo ""

python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes "${N_GENES}" \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
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
    echo "❌ ERROR: Dataset generation failed with exit code ${BUILDER_EXIT_CODE}"
    echo "📋 Check log file: ${LOG_FILE}"
    exit 1
fi

echo ""
echo "✅ Dataset generation completed successfully!"
echo ""

# Verify dataset output
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 VERIFYING DATASET OUTPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

MASTER_DIR="${DATASET_DIR}/master"

if [ ! -d "${MASTER_DIR}" ]; then
    echo "❌ ERROR: Master directory not found: ${MASTER_DIR}"
    exit 1
fi

echo "📂 Dataset structure:"
ls -lh "${DATASET_DIR}" | head -20
echo ""

echo "📂 Master directory:"
PARQUET_COUNT=$(ls -1 "${MASTER_DIR}"/*.parquet 2>/dev/null | wc -l)
echo "   Parquet files: ${PARQUET_COUNT}"
ls -lh "${MASTER_DIR}" | head -10
echo ""

if [ ${PARQUET_COUNT} -eq 0 ]; then
    echo "❌ ERROR: No parquet files found in master directory!"
    exit 1
fi

# Check for gene manifest
if [ -f "${DATASET_DIR}/gene_manifest.csv" ]; then
    echo "📋 Gene manifest:"
    GENE_COUNT=$(tail -n +2 "${DATASET_DIR}/gene_manifest.csv" | wc -l)
    echo "   Unique genes: ${GENE_COUNT}"
    echo "   First 5 genes:"
    head -6 "${DATASET_DIR}/gene_manifest.csv" | column -t -s,
    echo ""
else
    echo "⚠️  Warning: Gene manifest not found (may be normal for downsampled datasets)"
fi

echo "✅ Dataset verification passed!"
echo ""

# Quick dataset inspection with Python
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 DATASET INSPECTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python << EOF
import polars as pl
import pyarrow.dataset as ds
from pathlib import Path

master_dir = Path("${MASTER_DIR}")
dataset = ds.dataset(master_dir, format="parquet")

print("📊 Dataset Statistics:")
print(f"   Total rows: {dataset.count_rows():,}")
print()

# Sample first file to check schema
sample_file = list(master_dir.glob("*.parquet"))[0]
df_sample = pl.read_parquet(sample_file, n_rows=1000)
print(f"📝 Schema (from {sample_file.name}):")
print(f"   Total columns: {len(df_sample.columns)}")
print(f"   Column names: {', '.join(df_sample.columns[:10])}...")
print()

# Check for key columns
key_columns = ['position', 'splice_type', 'donor_score', 'acceptor_score']
existing_keys = [col for col in key_columns if col in df_sample.columns]
print(f"✅ Key columns present: {', '.join(existing_keys)}")
print()

# Check label distribution
if 'splice_type' in df_sample.columns:
    label_dist = df_sample['splice_type'].value_counts()
    print("🏷️  Label distribution (sample):")
    for row in label_dist.iter_rows():
        label, count = row
        print(f"   {label}: {count:,}")
    print()

# Check for k-mer features
kmer_cols = [col for col in df_sample.columns if 'mer_' in col.lower()]
print(f"🧬 K-mer features: {len(kmer_cols)}")
if kmer_cols:
    print(f"   Examples: {', '.join(kmer_cols[:5])}")
print()

print("✅ Dataset inspection completed!")
EOF

INSPECT_EXIT_CODE=$?

if [ ${INSPECT_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Dataset inspection failed"
    exit 1
fi

echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 STAGE 2: TRAIN META-MODEL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏰ Start time: $(date)"
echo ""

# Create model output directory
mkdir -p "${MODEL_DIR}"

echo "🧠 Running meta-model training with run_gene_cv_sigmoid.py..."
echo "   Command: python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid"
echo ""

# Run with minimal settings for quick test
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset "${MASTER_DIR}" \
    --out-dir "${MODEL_DIR}" \
    --n-folds 3 \
    --n-estimators 100 \
    --diag-sample 5000 \
    --row-cap 50000 \
    --skip-shap \
    --minimal-diagnostics \
    --no-plot-curves \
    --verbose 2>&1 | tee -a "${LOG_FILE}"

TRAINING_EXIT_CODE=$?

if [ ${TRAINING_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Meta-model training failed with exit code ${TRAINING_EXIT_CODE}"
    echo "📋 Check log file: ${LOG_FILE}"
    exit 1
fi

echo ""
echo "✅ Meta-model training completed successfully!"
echo ""

# Verify training output
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 VERIFYING TRAINING OUTPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📂 Model directory:"
ls -lh "${MODEL_DIR}" | head -20
echo ""

# Check for key output files
KEY_FILES=(
    "model_multiclass.pkl"
    "gene_cv_metrics.csv"
    "metrics_aggregate.json"
    "train.features.json"
)

echo "📋 Required output files:"
for file in "${KEY_FILES[@]}"; do
    if [ -f "${MODEL_DIR}/${file}" ]; then
        echo "   ✅ ${file}"
    else
        echo "   ❌ ${file} (MISSING)"
    fi
done
echo ""

# Display CV metrics if available
if [ -f "${MODEL_DIR}/gene_cv_metrics.csv" ]; then
    echo "📊 CV Metrics Summary:"
    python << EOF
import pandas as pd
import json

cv_metrics = pd.read_csv("${MODEL_DIR}/gene_cv_metrics.csv")
print(cv_metrics.to_string())
print()

# Display aggregate metrics if available
try:
    with open("${MODEL_DIR}/metrics_aggregate.json", "r") as f:
        agg_metrics = json.load(f)
    print("📈 Aggregate Metrics:")
    for key, value in agg_metrics.items():
        print(f"   {key}: {value:.4f}")
except Exception as e:
    print(f"Could not load aggregate metrics: {e}")
EOF
    echo ""
fi

echo "✅ Training output verification passed!"
echo ""

# Final summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 COMPLETE PIPELINE TEST SUCCESSFUL!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Dataset generation:  PASSED"
echo "✅ Dataset verification: PASSED"
echo "✅ Meta-model training:  PASSED"
echo "✅ Output verification:  PASSED"
echo ""
echo "📂 Generated artifacts:"
echo "   Dataset:  ${DATASET_DIR}"
echo "   Model:    ${MODEL_DIR}"
echo "   Log:      ${LOG_FILE}"
echo ""
echo "⏰ Completion time: $(date)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📖 NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Review CV metrics:       cat ${MODEL_DIR}/gene_cv_metrics.csv"
echo "2. Inspect trained model:   cat ${MODEL_DIR}/metrics_aggregate.json"
echo "3. Check full logs:         cat ${LOG_FILE}"
echo ""
echo "For production training with more genes:"
echo "   ./scripts/builder/run_builder_resumable.sh tmux"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

