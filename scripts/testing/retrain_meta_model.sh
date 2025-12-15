#!/bin/bash

# Retrain meta-model with current package name and full feature set
# This replaces the old meta_spliceai model with a fresh meta_spliceai model

set -e  # Exit on error

# Create logs directory
mkdir -p logs

echo "════════════════════════════════════════════════════════════════"
echo "🎯 RETRAINING META-MODEL WITH FULL FEATURE SET"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Dataset: data/train_pc_1000_3mers/master"
echo "Features: Base scores + k-mers + enriched features"
echo "Output: results/meta_model_1000genes_3mers_fresh"
echo ""
echo "This will create a fresh model with:"
echo "  ✅ Package name: meta_spliceai (not meta_spliceai)"
echo "  ✅ Full feature set including k-mers"
echo "  ✅ Per-class calibration"
echo "  ✅ Automatic leakage detection"
echo "  ✅ Overfitting monitoring"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Activate environment
echo "📦 Activating surveyor environment..."
source ~/miniforge3-new/etc/profile.d/conda.sh
conda activate surveyor

# Verify dataset exists
if [ ! -d "data/train_pc_1000_3mers/master" ]; then
    echo "❌ ERROR: Dataset not found: data/train_pc_1000_3mers/master"
    echo "   Please ensure the training dataset exists"
    exit 1
fi

# Run training with comprehensive feature set
echo "🚀 Starting meta-model training..."
echo ""

python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_pc_1000_3mers/master \
    --out-dir results/meta_model_1000genes_3mers_fresh \
    --n-estimators 800 \
    --n-folds 5 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --monitor-overfitting \
    --early-stopping-patience 30 \
    --verbose 2>&1 | tee logs/meta_training_1000genes_fresh.log

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ META-MODEL TRAINING COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Output directory: results/meta_model_1000genes_3mers_fresh"
echo "📋 Training log: logs/meta_training_1000genes_fresh.log"
echo ""
echo "Next steps:"
echo "  1. Verify model: results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl"
echo "  2. Check metrics: results/meta_model_1000genes_3mers_fresh/gene_cv_metrics.csv"
echo "  3. Update inference workflow to use new model path"
echo ""
echo "════════════════════════════════════════════════════════════════"

