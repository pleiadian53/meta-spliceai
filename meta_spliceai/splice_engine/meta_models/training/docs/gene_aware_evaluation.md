# Gene-Aware Evaluation Strategy

Meta-models in MetaSpliceAI are trained to **correct errors made by a base splice-site predictor (e.g. SpliceAI)**.  Because splice sites that belong to the same gene are highly correlated, random row-level splits overestimate generalisation performance.  To obtain an *honest* estimate we evaluate **per-gene**:

* All splice-site rows belonging to one `gene_id` are forced into the **same fold**.
* Metrics are reported on multiple levels:
  1. **Site-level** (classic AUC-ROC/AUC-PR, accuracy) – for backwards compatibility.
  2. **Gene-level** (*recommended*) – confusion counts and F1 aggregated per gene.
  3. **Transcript-level** (*optional*) – top-k accuracy at the transcript level, requiring transcript annotations.

---

## Implementation Details

| Component | How it works |
|-----------|--------------|
| **Group splits** | `datasets.train_valid_test_split(..., groups=gene_id)` uses `GroupShuffleSplit` to create train/valid/test.  `datasets.cv_splits(..., groups=gene_id)` wraps `GroupKFold`. |
| **Trainer** | `Trainer(group_col="gene_id")` (default) automatically passes the column through and stores test-set gene IDs. |
| **Metrics** | After fitting, `Trainer` calls `_gene_level_metrics()` which computes TP/FP/FN/TN, precision, recall and F1 for every gene and exposes them via `trainer.gene_metrics_df`. |
| **Overall score** | `gene_avg_f1` – simple arithmetic mean of per-gene F1.
| **Baseline comparison** | `Trainer.compare_baseline(full_splice_performance.tsv)` joins the meta-model's metrics with the baseline SpliceAI metrics and adds a `delta` column (positive = improvement).

> Requirements: **scikit-learn ≥0.22** (for `GroupKFold`).

---

## Enhanced Gene-Aware CV Scripts

### Modern Implementation: `run_gene_cv_sigmoid.py`

The current implementation provides comprehensive analysis capabilities including:

#### Core Features
- **Sigmoid Ensemble Architecture**: Three independent binary XGBoost classifiers for better class separation
- **Per-Class Calibration**: Individual calibration for each splice site class
- **Gene-Level Top-K Accuracy**: Evaluation at the gene level with detailed metrics
- **Transcript-Level Analysis**: Optional transcript-level top-k accuracy evaluation

#### Enhanced Analysis Pipeline
1. **CV Metrics Visualization**: 7 comprehensive plots analyzing performance across folds
2. **Enhanced ROC/PR Curves**: Both binary and multiclass curve analysis
3. **Comprehensive Feature Importance**: 4 different analytical methods (XGBoost internal, statistical tests, effect sizes, mutual information)
4. **Enhanced SHAP Analysis**: Memory-efficient SHAP with comprehensive visualization reports
5. **Feature Leakage Detection**: Correlation analysis to identify potentially problematic features

#### Output Structure
```
<out_dir>/
├── gene_cv_metrics.csv                     # Main CV results
├── metrics_aggregate.json                  # Summary statistics  
├── metrics_fold{N}.json                    # Per-fold metrics
├── metrics_folds.tsv                       # Fold summary table
├── model_multiclass.pkl                    # Final trained model
├── feature_manifest.csv                    # Feature list
├── cv_metrics_visualization/               # 7 visualization plots
│   ├── f1_comparison.pdf
│   ├── auc_comparison.pdf
│   ├── average_precision_comparison.pdf
│   ├── error_reduction_analysis.pdf
│   ├── performance_overview.pdf
│   ├── improvement_summary.pdf
│   └── top_k_analysis.pdf
├── roc_pr_curves_base.pdf                  # Base model ROC/PR
├── roc_pr_curves_meta.pdf                  # Meta model ROC/PR
├── multiclass_roc_curves.pdf               # Multi-class ROC
├── multiclass_pr_curves.pdf                # Multi-class PR
├── shap_analysis/                          # SHAP reports
│   ├── shap_summary.pdf
│   ├── shap_dependence_plots.pdf
│   └── shap_feature_importance.csv
├── feature_importance_analysis/            # Multi-method analysis
│   ├── feature_importance_comprehensive.xlsx
│   ├── xgboost_importance.csv
│   ├── statistical_tests.csv
│   ├── effect_sizes.csv
│   └── mutual_information.csv
├── feature_label_correlations.csv          # Leakage analysis
└── [various diagnostic files]
```

---

## Quick Start

### Basic Usage
```python
from meta_spliceai.splice_engine.meta_models.training import Trainer

trainer = (
    Trainer(model_spec="xgboost",
            out_dir="models/xgb_1000gene",
            group_col="gene_id")
    .fit("train_pc_1000/master")
)
trainer.save()  # saves model.pkl, metrics.json, run_meta.json

# Gene-level metrics
print(trainer.gene_metrics_df.head())
print("Avg F1 across genes:", trainer.metrics["gene_avg_f1"])

# Compare against baseline SpliceAI pass
baseline = "/home/bchiu/work/meta-spliceai/data/ensembl/spliceai_eval/full_splice_performance.tsv"
delta_df = trainer.compare_baseline(baseline)
print(delta_df.sort_values("delta", ascending=False).head())
```

### Comprehensive Analysis with Enhanced Features
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/comprehensive_gene_cv \
    --calibrate-per-class --calib-method platt \
    --plot-curves --plot-format pdf --n-roc-points 101 \
    --check-leakage --leakage-threshold 0.95 --auto-exclude-leaky \
    --diag-sample 10000 --neigh-sample 1000 --neigh-window 12 \
    --transcript-topk \
    --splice-sites-path data/ensembl/splice_sites.tsv \
    --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
    --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv \
    --n-folds 5 --verbose
```

---

## Enhanced Features

### 1. CV Metrics Visualization (7 Plots)
- **F1 Score Comparison**: Base vs Meta across CV folds
- **ROC AUC Comparison**: Area under curve analysis with confidence intervals
- **Average Precision Comparison**: Precision-recall performance evaluation
- **Error Reduction Analysis**: Visual analysis of delta_fp and delta_fn improvements
- **Performance Overview**: Multi-metric dashboard with accuracy breakdowns
- **Improvement Summary**: Percentage gains and quadrant analysis
- **Top-k Analysis**: Gene-level accuracy patterns

### 2. Comprehensive Feature Importance Analysis
Four different analytical methods provide complementary insights:
- **XGBoost Internal Importance**: weight, gain, cover, total_gain, total_cover metrics
- **Statistical Hypothesis Testing**: t-tests, Mann-Whitney U, Chi-square, Fisher's exact with FDR correction
- **Effect Size Measurements**: Cohen's d, Cramer's V, rank-biserial correlation  
- **Mutual Information Analysis**: for numerical and categorical features

### 3. Enhanced SHAP Analysis
- **Memory-Efficient Processing**: Incremental SHAP analysis for large datasets
- **Comprehensive Visualizations**: Summary plots, dependence plots, feature rankings
- **Per-Class Analysis**: Separate SHAP analysis for each splice site class
- **Graceful Fallbacks**: Automatic fallback to standard SHAP if memory-efficient version fails

### 4. Feature Quality Control
- **Leakage Detection**: Correlation analysis to identify potentially problematic features
- **Automatic Exclusion**: Option to automatically remove features that exceed correlation thresholds
- **Feature Filtering**: Support for excluding specific features via configuration files
- **Detailed Reporting**: Comprehensive logs of included/excluded features

---

## Recent Updates

### Transcript-Level Top-K Accuracy

Gene-aware CV scripts now support transcript-level top-k accuracy metrics, which evaluate model performance based on prioritization of splice sites within transcripts rather than genome-wide.

```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/meta_model_per_class \
    --splice-sites-path data/ensembl/splice_sites.tsv \
    --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
    --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv \
    --tree-method hist \
    --n-folds 5 \
    --transcript-topk
```

> **Note**: The parameters `--splice-sites-path` and `--annotations` are synonymous and both point to the same annotation file. Either can be used interchangeably in scripts.

### Enhanced Performance Comparison

When using transcript-level top-k accuracy, detailed performance comparisons between meta-model and base model are automatically generated in a TSV file (`perf_meta_vs_base.tsv`) in the output directory.

### Chromosome Encoding as Feature

Chromosome information can now be encoded as a numeric feature during training, allowing models to learn chromosome-specific patterns while maintaining evaluation integrity.

---

## FAQ

**Q: Can I use a different grouping key?**  
Yes – pass `group_col="transcript_id"` (or any column present in the dataset).  Make sure the same column is supplied during both training and inference.

**Q: How do I perform K-fold CV instead of a single split?**  
Use `datasets.cv_splits(X, y, groups=gene_ids, n_splits=5)` and loop manually, or use the enhanced `run_gene_cv_sigmoid.py` script which includes comprehensive K-fold CV with all enhanced features.

**Q: What about stratified grouping?**  
`StratifiedGroupKFold` is available in scikit-learn ≥1.4 – feel free to extend `datasets.cv_splits` accordingly if you need more balanced folds.

**Q: How do I disable specific enhanced features?**  
Use command-line flags like `--no-plot-curves`, `--no-leakage-check`, or set `--diag-sample 0` to disable diagnostics.

**Q: Can I run only specific analysis components?**  
Yes, the script includes graceful error handling so individual analysis components can fail without breaking the entire pipeline.

---

## Performance Expectations

### Typical Results (5-fold Gene CV)
- **F1 Improvement**: 30-40% improvement over base model
- **Error Reduction**: 300-800 fewer errors per fold
- **Top-K Accuracy**: 85-95% gene-level top-k accuracy
- **Processing Time**: 1-3 hours for 1000 genes with full analysis

### Resource Requirements
- **Memory**: 8-32 GB RAM depending on dataset size
- **Disk**: 2-5 GB for comprehensive outputs
- **CPU**: Multi-core recommended for SHAP analysis

---

*Last updated: 2025-01-15*
