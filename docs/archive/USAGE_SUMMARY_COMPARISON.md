# Usage Summary Comparison: Your Previous vs Run 4 Success

## Dataset Analysis
- **Total Dataset Size**: 1,329,518 rows (you mentioned 816,900 - this was likely an older dataset)
- **Run 4 Test Rows**: 999,425 total across 5 folds (75.2% of dataset)
- **Conclusion**: Run 4 likely used `--row-cap 0` successfully

## Parameter Comparison

| Parameter | Your Previous | Run 4 Success | Recommendation |
|-----------|---------------|---------------|----------------|
| `--n-estimators` | 500 | 800 | **Update to 800** |
| `--calibrate` | `--calibrate` | `--calibrate-per-class` | **Update to per-class** |
| `--calib-method` | platt | platt | ✅ Keep |
| `--diag-sample` | 15000 | 25000 | **Update to 25000** |
| `--neigh-sample` | 5000 | 1000 | **Reduce to 1000** |
| `--leakage-probe` | enabled | not used | Optional |
| `--transcript-topk` | missing | enabled | **Add this** |
| `--calibration-analysis` | missing | enabled | **Add this** |
| `--quick-overconfidence-check` | missing | enabled | **Add this** |

## Updated Optimized Command

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_pc_1000_3mers_run_6 \
    --n-folds 5 \
    --n-estimators 800 \
    --row-cap 0 \
    --diag-sample 25000 \
    --plot-curves --plot-format pdf \
    --check-leakage --leakage-threshold 0.95 --auto-exclude-leaky \
    --calibrate-per-class --calib-method platt \
    --neigh-sample 1000 --neigh-window 10 \
    --monitor-overfitting \
    --overfitting-threshold 0.05 \
    --early-stopping-patience 30 \
    --convergence-improvement 0.001 \
    --transcript-topk \
    --splice-sites-path data/ensembl/splice_sites.tsv \
    --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
    --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv \
    --calibration-analysis --quick-overconfidence-check \
    --verbose --seed 42
```

## Key Changes Explained

### 1. **Calibration Enhancement**
- **Old**: `--calibrate` (binary splice/non-splice)
- **New**: `--calibrate-per-class` (separate calibration for neither/donor/acceptor)
- **Impact**: Better probability estimates for each splice site class

### 2. **Increased n_estimators**
- **Old**: 500
- **New**: 800 with early stopping
- **Impact**: Better model performance with overfitting protection

### 3. **Optimized Sample Sizes**
- **diag-sample**: 15000 → 25000 (more comprehensive diagnostics)
- **neigh-sample**: 5000 → 1000 (reduced for efficiency)

### 4. **Added Transcript-Level Analysis**
- **New**: `--transcript-topk` with annotation paths
- **Impact**: More detailed performance evaluation

### 5. **Enhanced Calibration Analysis**
- **New**: `--calibration-analysis --quick-overconfidence-check`
- **Impact**: Detects and handles overconfidence issues

## Performance Expectations with Updated Command

Based on Run 4 results with similar configuration:
- **F1 Improvement**: 47.2% (0.648 → 0.954)
- **Error Reduction**: ~47,000 total errors reduced
- **Top-k Accuracy**: 96.8% gene-level accuracy
- **Runtime**: 2-4 hours with full analysis
- **Memory**: 8-16 GB RAM

## Backward Compatibility

Your original command will still work, but the updated version provides:
- Better calibration for multi-class problems
- More comprehensive analysis
- Better memory management
- Enhanced diagnostic capabilities






