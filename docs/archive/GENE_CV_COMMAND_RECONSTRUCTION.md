# Gene CV Command Reconstruction - Run 4 Analysis

## Summary

Based on analysis of the successful model directory `results/gene_cv_pc_1000_3mers_run_4/`, I've reconstructed the command that generated these excellent results and updated the documentation accordingly.

## Reconstructed Command

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_pc_1000_3mers_run_4 \
    --n-folds 5 --n-estimators 800 \
    --calibrate-per-class --calib-method platt \
    --plot-curves --plot-format pdf \
    --check-leakage --leakage-threshold 0.95 --auto-exclude-leaky \
    --monitor-overfitting --overfitting-threshold 0.05 \
    --early-stopping-patience 30 --convergence-improvement 0.001 \
    --diag-sample 25000 --neigh-sample 1000 \
    --transcript-topk \
    --splice-sites-path data/ensembl/splice_sites.tsv \
    --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
    --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv \
    --calibration-analysis --quick-overconfidence-check \
    --verbose
```

## Key Evidence from Analysis

### Dataset Characteristics
- **Dataset**: `train_pc_1000_3mers/master` (1000 protein-coding genes)
- **Features**: 124 total features including 3-mer k-mers (`3mer_AAA`, `3mer_AAC`, etc.)
- **Excluded**: 11 features automatically excluded (transcript/gene info features)

### Performance Results
- **F1 Improvement**: 47.2% (0.648 → 0.954)
- **Error Reduction**: 47,687 total errors reduced
  - False Positive Reduction: 59.9% (8,289 errors)
  - False Negative Reduction: 78.3% (39,398 errors)
- **Top-k Accuracy**: 96.8% gene-level accuracy
- **ROC AUC**: 0.999+ for meta vs 0.989 for base

### Training Configuration Evidence
- **Overfitting Analysis**: 15 binary models trained (5 folds × 3 classes)
- **Early Stopping**: All models early stopped at ~312 iterations on average
- **Recommended n_estimators**: 416 (but 800 was used with early stopping)
- **Calibration**: Per-class Platt calibration enabled
- **Memory Management**: 25,000 diagnostic samples, overfitting monitoring

### Generated Artifacts
- Comprehensive CV metrics visualization (7 plots)
- SHAP analysis with feature importance
- Leakage analysis with correlation detection
- Overfitting analysis with learning curves
- Multi-class ROC/PR curves
- Per-nucleotide meta-scores for inference

## Documentation Updates

Updated `meta_spliceai/splice_engine/meta_models/training/docs/gene_cv_sigmoid.md` with:

1. **Production-Ready Command**: Based on successful Run 4 configuration
2. **Updated Examples**: All examples now use `train_pc_1000_3mers` dataset
3. **Performance Expectations**: Updated with actual Run 4 results
4. **Memory Management**: Proper `--memory-optimize` usage
5. **Troubleshooting**: Added FAQ based on successful configuration

## Key Parameters for Success

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--n-estimators` | 800 | High enough for early stopping to find optimal |
| `--calibrate-per-class` | enabled | Essential for multi-class probability calibration |
| `--monitor-overfitting` | enabled | Prevents overfitting with early stopping |
| `--diag-sample` | 25000 | Balanced sample size for comprehensive diagnostics |
| `--auto-exclude-leaky` | enabled | Automatically removes problematic features |
| `--early-stopping-patience` | 30 | Allows sufficient convergence time |
| `--calibration-analysis` | enabled | Detects overconfidence issues |

## Next Steps

To replicate these results:

1. Use the "Production-Ready Command" from the updated documentation
2. Ensure you have the `train_pc_1000_3mers` dataset available
3. Expect 2-4 hours runtime with 8-16 GB RAM usage
4. Results will be saved to `results/gene_cv_pc_1000_3mers_run_5/`

The updated documentation now provides clear guidance for achieving similar excellent results with proper memory management and performance evaluation.






