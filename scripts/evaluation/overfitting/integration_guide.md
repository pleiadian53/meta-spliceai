# Overfitting Monitoring Integration Guide

This guide shows how to integrate comprehensive overfitting detection into your existing CV scripts.

## ✅ Integration Complete!

The overfitting monitoring system has been successfully integrated into both CV scripts:
- **Gene-aware CV**: `run_gene_cv_sigmoid.py` 
- **Chromosome-aware CV**: `run_loco_cv_multiclass_scalable.py`

## New Command Line Arguments

Both scripts now support these additional arguments for overfitting monitoring and memory optimization:

### Overfitting Monitoring
- `--monitor-overfitting`: Enable comprehensive overfitting monitoring and analysis
- `--overfitting-threshold FLOAT`: Performance gap threshold for overfitting detection (default: 0.05)
- `--early-stopping-patience INT`: Patience for early stopping detection (default: 20)  
- `--convergence-improvement FLOAT`: Minimum improvement threshold for convergence detection (default: 0.001)

### Memory Optimization (Gene CV only)
- `--memory-optimize`: Enable memory optimization for low-memory systems
- `--max-diag-sample INT`: Maximum diagnostic sample size for memory optimization (default: 25000)

## Quick Integration Steps

### 1. Gene CV Script Integration (`run_gene_cv_sigmoid.py`)

Add these imports at the top:
```python
from meta_spliceai.splice_engine.meta_models.evaluation.overfitting_monitor import (
    OverfittingMonitor, enhanced_model_training
)
```

Initialize the monitor in the main function:
```python
def main(argv: List[str] | None = None) -> None:
    # ... existing code ...
    
    # Initialize overfitting monitor
    monitor = OverfittingMonitor(
        primary_metric="logloss",
        gap_threshold=0.05,  # Adjust based on your tolerance
        patience=20,
        min_improvement=0.001
    )
    
    # ... existing CV loop ...
```

Modify the model training section:
```python
# Replace the existing _train_binary_model calls with:
for cls in (0, 1, 2):
    y_train_bin = (y[train_idx] == cls).astype(int)
    y_val_bin = (y[valid_idx] == cls).astype(int)
    
    # Enhanced training with monitoring
    model_c = enhanced_model_training(
        X[train_idx], y_train_bin,
        X[valid_idx], y_val_bin,
        args, monitor, f"{fold_idx}_{cls}"
    )
    models_cls.append(model_c)
```

Add overfitting analysis after the CV loop:
```python
# After all folds complete
if args.monitor_overfitting:  # Add this flag to args
    print("\n📋 Generating overfitting analysis...")
    overfitting_report = monitor.generate_overfitting_report(out_dir)
    monitor.plot_learning_curves(out_dir, plot_format=args.plot_format)
    
    # Print summary
    summary = overfitting_report['summary']
    print(f"Folds with overfitting: {summary['folds_with_overfitting']}")
    print(f"Recommended n_estimators: {summary['recommended_n_estimators']}")
```

### 2. LOCO CV Script Integration (`run_loco_cv_multiclass_scalable.py`)

Similar integration pattern:

```python
# Initialize monitor
monitor = OverfittingMonitor(
    primary_metric="mlogloss",  # Use multiclass log loss
    gap_threshold=0.05,
    patience=20,
    min_improvement=0.001
)

# In the fold training loop, replace model.fit() with:
# Create enhanced model with monitoring
model = create_xgb_classifier(args)

# Enhanced fit with evaluation tracking
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_names=['train', 'eval'],
    verbose=args.verbose > 0
)

# Extract evaluation results and add to monitor
if hasattr(model, 'evals_result'):
    evals_result = model.evals_result()
    monitor.add_fold_metrics(evals_result, fold_idx)
```

## Command Line Arguments

Add these new arguments to both scripts:

```python
def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    # ... existing arguments ...
    
    # Overfitting monitoring arguments
    parser.add_argument(
        "--monitor-overfitting", 
        action="store_true",
        help="Enable comprehensive overfitting monitoring and analysis"
    )
    parser.add_argument(
        "--overfitting-threshold", 
        type=float, 
        default=0.05,
        help="Performance gap threshold for overfitting detection (default: 0.05)"
    )
    parser.add_argument(
        "--early-stopping-patience", 
        type=int, 
        default=20,
        help="Patience for early stopping detection (default: 20)"
    )
    parser.add_argument(
        "--convergence-improvement", 
        type=float, 
        default=0.001,
        help="Minimum improvement threshold for convergence detection (default: 0.001)"
    )
```

## Updated Command Examples

### Gene CV with Overfitting Monitoring:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir results_gene_cv_1000_monitored \
    --n-folds 5 \
    --n-estimators 500 \
    --monitor-overfitting \
    --overfitting-threshold 0.05 \
    --early-stopping-patience 30 \
    --convergence-improvement 0.001 \
    --plot-curves \
    --verbose \
    --seed 42
```

### LOCO CV with Overfitting Monitoring:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir results_loco_cv_1000_monitored \
    --n-estimators 500 \
    --monitor-overfitting \
    --overfitting-threshold 0.05 \
    --early-stopping-patience 30 \
    --convergence-improvement 0.001 \
    --plot-curves \
    --verbose 2 \
    --seed 42
```

### Gene CV with Memory Optimization (for low-memory systems):
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir results_gene_cv_1000_memory_opt \
    --n-folds 5 \
    --n-estimators 500 \
    --monitor-overfitting \
    --memory-optimize \
    --max-diag-sample 10000 \
    --diag-sample 10000 \
    --neigh-sample 500 \
    --plot-curves \
    --seed 42
```

## Enhanced Output Files

With overfitting monitoring enabled, you'll get these additional outputs organized in a subdirectory:

```
results_cv_monitored/
├── gene_cv_metrics.csv                     # Standard CV metrics
├── model_multiclass.pkl                    # Trained model
├── feature_manifest.csv                    # Feature list
├── overfitting_analysis/                   # Overfitting analysis subdirectory
│   ├── overfitting_analysis.json          # Comprehensive overfitting metrics
│   ├── overfitting_summary.txt            # Human-readable summary report
│   ├── learning_curves_by_fold.pdf        # Individual fold learning curves
│   ├── aggregated_learning_curves.pdf     # Mean curves with confidence bands
│   └── overfitting_summary.pdf            # Overfitting detection visualizations
├── shap_importance_incremental.csv         # SHAP analysis results
├── cv_metrics_visualization/               # Standard CV visualizations
└── ... (other standard CV outputs)
```

## Key Benefits for Presentations

### 1. **Real-time Overfitting Detection**
- Immediate feedback during CV: "🚨 OVERFITTING DETECTED in Fold 3!"
- Performance gap visualization across folds
- Early warning system for model tuning

### 2. **Publication-Ready Visualizations**
- Learning curves showing training vs validation performance
- Confidence bands showing variance across folds
- Professional dashboard summarizing all metrics

### 3. **Data-Driven Recommendations**
- Optimal n_estimators based on actual convergence patterns
- Early stopping effectiveness analysis
- Hyperparameter tuning guidance

### 4. **Presentation-Friendly Metrics**
- Clear overfitting scores and thresholds
- Fold-by-fold performance consistency
- Executive summary with actionable insights

## Example Output for Presentations

```
===============================================================
OVERFITTING ANALYSIS SUMMARY
===============================================================
Total folds: 5
Folds with overfitting: 2
Early stopped folds: 3
Mean performance gap: 0.0342 ± 0.0156
Mean overfitting score: 0.0298
Recommended n_estimators: 387
===============================================================

📋 ACTIONABLE RECOMMENDATIONS:
1. 🚨 OVERFITTING DETECTED: 2/5 folds show overfitting. 
   Consider reducing n_estimators from 500 to ~387.
2. ✅ EFFECTIVE EARLY STOPPING: 3 folds benefited from early stopping. 
   Continue using early stopping to optimize training efficiency.
3. ✅ GOOD GENERALIZATION: Performance gap is within acceptable range.
```

## Visual Analysis Examples

The monitoring provides several types of visualizations perfect for presentations:

1. **Learning Curves**: Show how training and validation performance evolve
2. **Performance Gaps**: Highlight overfitting across folds
3. **Convergence Analysis**: Demonstrate model stability
4. **Executive Dashboard**: Single-page summary for stakeholders

## Integration Testing

Test the integration with a small dataset first:

```bash
# Test with sample genes
python scripts/enhanced_cv_example.py
```

This will create a demo output directory showing all the visualizations and reports you'll get with the full integration.

## Best Practices

1. **Set Appropriate Thresholds**: Start with 0.05 gap threshold, adjust based on your domain
2. **Monitor Multiple Metrics**: Use logloss as primary, but track F1 and AUC too
3. **Cross-Validate Thresholds**: Different datasets may need different overfitting thresholds
4. **Document Findings**: The generated reports are perfect for method sections in papers

## Troubleshooting

### Common Issues:
1. **No evaluation results**: Ensure `eval_set` is properly configured in XGBoost
2. **Memory issues**: Use diagnostic sampling for large datasets
3. **Visualization errors**: Check matplotlib backend settings

### Solutions:
- Add verbose logging to track evaluation progress
- Use memory-efficient batch processing for large folds
- Save intermediate results for debugging

This integration transforms your CV workflow from simple performance tracking to comprehensive overfitting analysis with publication-ready visualizations!

## Quick Testing

To test the integration, you can run the provided test script:

```bash
# From the project root directory
python scripts/test_overfitting_integration.py
```

This script will:
1. Run both CV approaches with small datasets for quick testing
2. Verify that overfitting analysis files are generated correctly  
3. Compare output structures between the two approaches
4. Generate a summary report with integration status

The test uses small parameters for quick execution:
- Sample genes: 10 (for gene CV)
- Row cap: 5,000 (for LOCO CV)  
- N estimators: 50
- Folds: 3 (for gene CV)

After successful testing, you can run the full commands with your production parameters.

## Troubleshooting Common Issues

### 1. **Out of Memory (OOM) Errors**
```bash
# If you encounter "Killed" or OOM errors, use memory optimization:
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --memory-optimize \
    --max-diag-sample 10000 \
    --diag-sample 10000 \
    --neigh-sample 500 \
    ... other args
```

### 2. **Feature Importance Analysis Failures**
```
Error: 'CalibratedSigmoidEnsemble' object has no attribute 'get_booster'
```
**Solution**: This is now handled automatically. The integration detects ensemble models and uses fallback methods.

### 3. **String to Float Conversion Errors**
```
Error: could not convert string to float: 'X'
```
**Solution**: This occurs with chromosome encoding and is handled gracefully. The analysis continues with other diagnostic methods.

### 4. **Missing Function Errors**
```
Error: cannot import name 'evaluate_predictions'
```
**Solution**: This is now handled with appropriate fallbacks. Base vs meta comparison will be skipped if the required functions are unavailable.

### 5. **Neighbor Window Diagnostics Errors**
```
Error: unexpected keyword argument 'sample'
```
**Solution**: Fixed in the integration. Now uses correct parameter name `n_sample`.

### Memory Usage Tips

- Start with small diagnostic samples: `--diag-sample 5000`
- Use `--memory-optimize` on low-memory systems
- Reduce neighbor analysis: `--neigh-sample 500` or `--neigh-sample 0` to disable
- Consider using fewer CV folds: `--n-folds 3` instead of 5 