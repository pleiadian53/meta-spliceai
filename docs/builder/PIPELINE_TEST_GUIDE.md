# Complete Pipeline Testing Guide

## Overview

This guide explains how to test the complete meta-learning pipeline from training dataset generation to meta-model training and evaluation.

## Pipeline Stages

The complete meta-learning pipeline consists of two main stages:

### Stage 1: Training Dataset Generation
- **Tool**: `incremental_builder.py`
- **Input**: GTF + FASTA (via genomic resources), base model predictions
- **Output**: Training dataset with features (Parquet format)
- **Duration**: ~5-15 minutes for 50-100 genes

### Stage 2: Meta-Model Training
- **Tool**: `run_gene_cv_sigmoid.py`
- **Input**: Training dataset from Stage 1
- **Output**: Trained meta-model + evaluation metrics
- **Duration**: ~5-10 minutes for 50-100 genes

## Quick Test (Recommended First Step)

```bash
# Quick test with 50 genes (~15-20 minutes total)
./scripts/testing/test_complete_pipeline.sh

# View results
cat logs/pipeline_test/test_pipeline_50genes_*.log
```

## Thorough Test

```bash
# More thorough test with 100 genes (~25-30 minutes total)
./scripts/testing/test_complete_pipeline.sh 100
```

## What the Test Does

### 1. Dataset Generation Phase
```bash
# Generates training dataset with:
- N genes (50 by default)
- Random gene selection + ALS genes
- Protein-coding and lncRNA genes
- 3-mer k-mer features
- --run-workflow: Base model pass (SpliceAI inference)
```

**Expected Outputs**:
- `data/test_pipeline_*genes_*/master/*.parquet` - Training data files
- `data/test_pipeline_*genes_*/gene_manifest.csv` - Gene metadata
- Batch processing artifacts

### 2. Meta-Model Training Phase
```bash
# Trains meta-model with:
- 3-fold cross-validation (quick test)
- 100 trees per model (minimal for speed)
- Minimal diagnostics (skip SHAP, etc.)
- 5000 diagnostic samples
```

**Expected Outputs**:
- `model_multiclass.pkl` - Trained meta-model
- `gene_cv_metrics.csv` - Per-fold performance
- `metrics_aggregate.json` - Overall performance
- `train.features.json` - Feature manifest

## Verification Steps

The test script automatically verifies:

### Dataset Verification
- ✅ Master directory exists with Parquet files
- ✅ Gene manifest generated (if applicable)
- ✅ Schema contains key columns (position, splice_type, donor_score, acceptor_score)
- ✅ Label distribution looks reasonable
- ✅ K-mer features present

### Training Verification
- ✅ Model file generated (`model_multiclass.pkl`)
- ✅ CV metrics computed (`gene_cv_metrics.csv`)
- ✅ Aggregate metrics available (`metrics_aggregate.json`)
- ✅ Feature manifest saved (`train.features.json`)

## Expected Performance

### Dataset Generation (50 genes)
- **Time**: 5-15 minutes
- **Output size**: ~10-50 MB
- **Rows**: ~50K-200K training examples

### Meta-Model Training (50 genes)
- **Time**: 5-10 minutes
- **F1 Score**: 0.85-0.95 (depends on data quality)
- **Top-k Accuracy**: 0.80-0.95

## Interpreting Results

### CV Metrics (`gene_cv_metrics.csv`)

Key columns to check:
- `test_macro_f1`: Overall F1 score (higher is better, >0.85 is good)
- `donor_f1`, `acceptor_f1`: Per-class F1 scores
- `top_k_accuracy`: Gene-level top-k accuracy (>0.80 is good)
- `auc_meta`: ROC AUC for meta-model (>0.90 is good)

Example output:
```
fold  test_macro_f1  donor_f1  acceptor_f1  top_k_accuracy  auc_meta
0     0.887          0.892     0.881        0.854           0.947
1     0.891          0.895     0.887        0.862           0.951
2     0.884          0.888     0.880        0.847           0.945
```

### Aggregate Metrics (`metrics_aggregate.json`)

```json
{
  "test_macro_f1": 0.8873,
  "donor_f1": 0.8917,
  "acceptor_f1": 0.8829,
  "top_k_accuracy": 0.8543,
  "auc_meta": 0.9477
}
```

## Troubleshooting

### Issue: "No parquet files found"
**Cause**: Dataset generation failed or genes have no training data
**Solution**: 
- Check if selected genes have splice sites
- Try with `--gene-types protein_coding` only
- Increase `--n-genes` to ensure some succeed

### Issue: "Schema mismatch" during training
**Cause**: Dataset schema doesn't match training expectations
**Solution**:
- Regenerate dataset with `--overwrite`
- Ensure `--run-workflow` was used (includes probability features)

### Issue: Training is very slow
**Cause**: Too many genes or features for quick test
**Solution**:
- Use default 50 genes for testing
- Increase `--row-cap` in training phase
- Use `--minimal-diagnostics` flag

### Issue: Low F1 scores (<0.70)
**Cause**: Insufficient training data or data quality issues
**Solution**:
- Increase number of genes (try 100-200)
- Check data quality in gene_manifest.csv
- Ensure base model predictions are available

## Advanced Testing

### Test with Specific Genes

```bash
# Create custom gene list
cat > my_test_genes.txt << EOF
ENSG00000007314  # UNC13A
ENSG00000184557  # STMN2
ENSG00000142611  # PRDX5
EOF

# Run builder with custom genes
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --subset-policy custom \
    --gene-ids-file my_test_genes.txt \
    --output-dir data/test_custom_genes \
    --run-workflow \
    --overwrite
```

### Test with Different K-mer Sizes

```bash
# Test with multiple k-mer sizes
./scripts/testing/test_complete_pipeline.sh 50  # Edit script to change KMER_SIZES="3,5"
```

### Memory-Constrained Testing

```bash
# For laptops with limited memory
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/test_pipeline_50genes_*/master \
    --out-dir data/test_model_minimal \
    --n-folds 3 \
    --n-estimators 50 \
    --row-cap 25000 \
    --diag-sample 2000 \
    --skip-shap \
    --minimal-diagnostics \
    --memory-optimize
```

## Production Workflow

After successful testing, proceed to production:

### 1. Generate Production Dataset (1000-5000 genes)

```bash
# Using resumable execution
./scripts/builder/run_builder_resumable.sh tmux
```

### 2. Train Production Model

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_1000genes_3mers/master \
    --out-dir data/model_1000genes \
    --n-folds 5 \
    --n-estimators 800 \
    --verbose
```

## Expected Timeline

### Quick Test (50 genes)
- Dataset generation: ~5-10 min
- Model training: ~5-10 min
- **Total**: ~10-20 minutes

### Thorough Test (100 genes)
- Dataset generation: ~10-20 min
- Model training: ~10-15 min
- **Total**: ~20-35 minutes

### Production (1000 genes)
- Dataset generation: ~40-60 min
- Model training: ~30-60 min
- **Total**: ~1-2 hours

### Large Scale (5000 genes)
- Dataset generation: ~3-4 hours
- Model training: ~2-3 hours
- **Total**: ~5-7 hours

## Monitoring Progress

### During Dataset Generation
```bash
# Watch progress
tail -f logs/pipeline_test/test_pipeline_*.log | grep -E '\[batch_|genes|rows\]'
```

### During Training
```bash
# Watch fold progress
tail -f logs/pipeline_test/test_pipeline_*.log | grep -E 'Fold|F1='
```

## Output Locations

All test outputs use timestamped directories:
- **Datasets**: `data/test_pipeline_<N>genes_<timestamp>/`
- **Models**: `data/test_model_<N>genes_<timestamp>/`
- **Logs**: `logs/pipeline_test/test_pipeline_<N>genes_<timestamp>.log`

## Cleanup

After testing, you can clean up test artifacts:

```bash
# Remove test datasets (keep logs)
rm -rf data/test_pipeline_*
rm -rf data/test_model_*

# Or keep everything for reference
# (recommended for first few tests)
```

## Success Criteria

A successful pipeline test should show:

✅ **Dataset Generation**
- All batch files created
- Gene manifest generated
- Row counts match expectations

✅ **Model Training**
- All CV folds complete
- F1 score > 0.80
- Top-k accuracy > 0.75
- No errors in logs

✅ **Performance**
- Meta-model outperforms base model
- Consistent metrics across folds
- Reasonable training time

## Next Steps

After successful pipeline testing:

1. **Review metrics** to understand baseline performance
2. **Generate larger dataset** (1000-5000 genes) for production
3. **Train production model** with full hyperparameters
4. **Evaluate on held-out genes** for final validation
5. **Deploy model** for variant analysis

## Related Documentation

- [Builder Usage Guide](USAGE_GUIDE.md) - Detailed incremental_builder options
- [Command Examples](COMMAND_EXAMPLES.md) - More command examples
- [Base Model Pass Workflow](../base_models/BASE_MODEL_PASS_WORKFLOW.md) - Understanding --run-workflow
- [Training Guide](../../meta_spliceai/splice_engine/meta_models/training/docs/) - Meta-model training details

---

**Last Updated**: October 17, 2025

