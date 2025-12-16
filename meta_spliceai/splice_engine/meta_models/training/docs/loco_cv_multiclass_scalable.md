# Memory-Efficient Chromosome-Wise Cross-Validation (LOCO-CV)

File: `meta_spliceai/splice_engine/meta_models/training/run_loco_cv_multiclass_scalable.py`

## Purpose

This module implements a **scalable Leave-One-Chromosome-Out Cross-Validation (LOCO-CV)** for meta-model training that provides **complete feature parity** with gene-aware CV while handling large datasets (20K+ genes) through advanced memory optimization techniques.

### Key Features

* **Memory-Efficient Processing**: Chunked/streaming data loading and sparse matrix support
* **Comprehensive Analysis Pipeline**: Same enhanced analysis capabilities as gene-aware CV
* **Advanced Feature Selection**: Automated feature selection with detailed reporting
* **Complete Feature Parity**: All analytical methods available in gene-aware CV
* **Scalable Architecture**: Designed for large-scale genomic datasets

## Enhanced Analysis Pipeline (Identical to Gene CV)

### 1. CV Metrics Visualization (7 Plots)
- **F1 Score Comparison**: Performance across CV folds with statistical summaries
- **ROC AUC Comparison**: Area under curve analysis with confidence intervals
- **Average Precision Comparison**: Precision-recall performance evaluation
- **Error Reduction Analysis**: Visual analysis of performance improvements
- **Performance Overview**: Multi-metric dashboard with accuracy breakdowns
- **Improvement Summary**: Percentage gains and quadrant analysis
- **Top-k Analysis**: Accuracy patterns across chromosomes

### 2. Enhanced ROC/PR Curves
- **Meta Model Analysis**: Comprehensive curves for meta-model performance
- **Multiclass Analysis**: Per-class ROC/PR curves with macro-averaging
- **Statistical Analysis**: Confidence intervals and significance testing
- **Publication-Ready Plots**: High-quality visualizations for reports

### 3. Comprehensive Feature Importance Analysis
Four complementary analytical methods (identical to gene CV):
- **XGBoost Internal Importance**: weight, gain, cover, total_gain, total_cover metrics
- **Statistical Hypothesis Testing**: t-tests, Mann-Whitney U, Chi-square, Fisher's exact with FDR correction
- **Effect Size Measurements**: Cohen's d, Cramer's V, rank-biserial correlation
- **Mutual Information Analysis**: Information-theoretic feature ranking

### 4. Enhanced SHAP Analysis
- **Memory-Efficient Processing**: Incremental SHAP analysis for large datasets
- **Comprehensive Visualizations**: Summary plots, dependence plots, feature rankings
- **Per-Class Analysis**: Separate SHAP analysis for each splice site class
- **Graceful Fallbacks**: Automatic fallback to standard SHAP if memory-efficient version fails

### 5. Feature Quality Control
- **Leakage Detection**: Correlation analysis to identify potentially problematic features
- **Automatic Exclusion**: Option to automatically remove features that exceed correlation thresholds
- **Feature Filtering**: Support for excluding specific features via configuration files
- **Detailed Reporting**: Comprehensive logs of included/excluded features

### 6. Diagnostic Analysis Suite
Complete set of 8 diagnostic functions:
- **Richer Metrics**: Extended performance analysis
- **Gene Score Delta**: Analysis of score differences between models
- **Probability Diagnostics**: Probability distribution analysis
- **Base vs Meta Comparison**: Comprehensive performance comparison
- **Meta Splice Performance**: Splice-specific performance evaluation
- **Neighbour Window Diagnostics**: Neighborhood analysis
- **Leakage Probe**: Advanced feature leakage detection
- **Meta Performance Evaluation**: Overall model assessment

## Memory Optimizations

1. **Chunked dataset loading** - Uses the `chunked_datasets` module to stream data in batches
2. **Sparse matrices** - Stores high-dimensional k-mer features in memory-efficient sparse format
3. **Feature filtering** - Optionally reduces feature set to most important features
4. **Aggressive garbage collection** - Explicitly releases memory after processing each fold
5. **Selective feature loading** - Only loads columns needed for analysis

## Entry Point

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset data/training/features \
    --out-dir models/loco_cv_scalable \
    --chunk-size 50000
```

## Command-Line Arguments

### Core Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Path to feature dataset | *(required)* |
| `--out-dir` | Output directory for results | *(required)* |
| `--group-col` | Column for CV grouping | `chrom` |
| `--gene-col` | Column for gene IDs | `gene_id` |
| `--base-tsv` | Include base model columns | None |
| `--errors-only` | Evaluate only base model errors | *(flag)* |
| `--row-cap` | Maximum rows to use (0=no cap) | 0 |
| `--valid-size` | Validation set size (fraction) | 0.15 |
| `--min-rows-test` | Min rows in test fold | 1000 |
| `--heldout-chroms` | Fixed test chromosomes | None |

### Model Configuration
| Argument | Description | Default |
|----------|-------------|---------|
| `--tree-method` | XGBoost tree method | `hist` |
| `--n-estimators` | Number of trees | 100 |
| `--max-depth` | Maximum tree depth | 6 |
| `--subsample` | Subsample ratio | 0.8 |
| `--colsample-bytree` | Column sample ratio per tree | 0.8 |
| `--eta` | Learning rate | 0.3 |
| `--gpu` | Use GPU acceleration | *(flag)* |

### Calibration Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--calibrate` | Use binary calibration | *(flag)* |
| `--calibrate-per-class` | Use per-class calibration | *(flag)* |
| `--calib-method` | Calibration method | `isotonic` |
| `--meta-calibrate` | Apply meta-calibration | *(flag)* |

### Memory Optimization
| Argument | Description | Default |
|----------|-------------|---------|
| `--chunk-size` | Chunk size for streaming | 100000 |
| `--use-chunked-loading` | Enable chunked data loading | *(flag)* |
| `--use-sparse-kmers` | Use sparse k-mer matrices | *(flag)* |
| `--memory-optimize` | Enable memory optimizations | *(flag)* |
| `--max-features` | Maximum features to use | None |

### Visualization and Analysis
| Argument | Description | Default |
|----------|-------------|---------|
| `--plot-curves` | Generate ROC/PR curves | *(flag, enabled by default)* |
| `--no-plot-curves` | Disable curve plotting | *(flag)* |
| `--plot-format` | Plot format (pdf/png/svg) | `pdf` |
| `--n-roc-points` | Points for ROC curves | 101 |
| `--dpi` | Plot DPI | 300 |

### Feature Quality Control
| Argument | Description | Default |
|----------|-------------|---------|
| `--check-leakage` | Enable leakage detection | *(flag)* |
| `--leakage-threshold` | Correlation threshold | 0.95 |
| `--auto-exclude-leaky` | Auto-exclude leaky features | *(flag)* |
| `--exclude-features` | Feature exclusion file | None |
| `--filter-features` | Filter to important features | *(flag)* |
| `--feature-selection` | Enable feature selection | *(flag)* |

### Diagnostic Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--diag-sample` | Sample size for diagnostics | 25000 |
| `--neigh-sample` | Samples for neighborhood analysis | 0 |
| `--neigh-window` | Window size for neighborhood | 50 |
| `--neigh-unseen` | Analyze unseen neighborhoods | *(flag)* |
| `--leakage-probe` | Run leakage probe analysis | *(flag)* |
| `--verbose` | Verbosity level | 1 |

### Transcript-Level Analysis
| Argument | Description | Default |
|----------|-------------|---------|
| `--transcript-topk` | Enable transcript-level top-k | *(flag)* |
| `--splice-sites-path` | Path to splice site annotations | None |
| `--transcript-features-path` | Path to transcript features | None |
| `--gene-features-path` | Path to gene features | None |
| `--top-k` | Top-k value for analysis | 5 |
| `--include-tns` | Include true negatives | *(flag)* |

## Output Structure

The script creates a comprehensive output directory with **complete parity** with gene CV:

```
<out_dir>/
├── loco_metrics.csv                          # Main CV results (chromosome-wise)
├── metrics_aggregate.json                    # Summary statistics
├── fold_{chrom}_metrics.json                 # Per-fold metrics
├── final_model_uncalibrated.json             # Uncalibrated model
├── final_model_calibrated_{type}.pkl         # Calibrated models
├── fold_{chrom}_model.json                   # Per-fold models
├── feature_manifest.csv                      # Feature list
├── feature_selection_info.json               # Feature selection details
├── cv_metrics_visualization/                 # 7 visualization plots
│   ├── f1_comparison.pdf
│   ├── auc_comparison.pdf
│   ├── average_precision_comparison.pdf
│   ├── error_reduction_analysis.pdf
│   ├── performance_overview.pdf
│   ├── improvement_summary.pdf
│   └── top_k_analysis.pdf
├── roc_pr_curves_meta.pdf                    # Meta model ROC/PR
├── multiclass_roc_curves.pdf                 # Multi-class ROC
├── multiclass_pr_curves.pdf                  # Multi-class PR
├── shap_analysis/                            # SHAP reports
│   ├── shap_summary.pdf
│   ├── shap_dependence_plots.pdf
│   └── shap_feature_importance.csv
├── feature_importance_analysis/              # Multi-method analysis
│   ├── feature_importance_comprehensive.xlsx
│   ├── xgboost_importance.csv
│   ├── statistical_tests.csv
│   ├── effect_sizes.csv
│   └── mutual_information.csv
├── feature_label_correlations.csv            # Leakage analysis
├── fold_chrom1/                              # Per-fold outputs
│   ├── model.json
│   ├── model_calibrated.pkl
│   ├── metrics.json
│   └── neighborhood/
└── [comprehensive diagnostic files]
```

## Usage Examples

### Basic LOCO-CV with Chunking
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir models/basic_loco_cv \
    --chunk-size 50000
```

### Comprehensive Analysis (Recommended)
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir models/comprehensive_loco_cv \
    --calibrate-per-class --calib-method platt \
    --plot-curves --plot-format pdf --n-roc-points 101 \
    --check-leakage --leakage-threshold 0.95 --auto-exclude-leaky \
    --diag-sample 10000 --neigh-sample 1000 --neigh-window 50 \
    --transcript-topk \
    --splice-sites-path data/ensembl/splice_sites.tsv \
    --verbose 2
```

### Memory-Optimized Large-Scale LOCO-CV
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_full/master \
    --out-dir models/large_scale_loco_cv \
    --use-chunked-loading --chunk-size 10000 \
    --use-sparse-kmers --memory-optimize \
    --feature-selection --max-features 1000 \
    --calibrate-per-class --row-cap 0 \
    --verbose 2
```

### Fixed Holdout with GPU Acceleration
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir models/holdout_gpu \
    --heldout-chroms "21,22,X" \
    --tree-method gpu_hist --gpu \
    --calibrate-per-class --plot-curves
```

### Unseen Neighborhood Analysis
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir models/loco_cv_neigh \
    --neigh-sample 1000 --neigh-window 50 \
    --neigh-unseen --transcript-topk
```

## Integration with Enhanced Inference Workflow

Models trained with this script can be directly used in the enhanced splice inference workflow:

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_inference_workflow import run_enhanced_splice_inference_workflow

run_enhanced_splice_inference_workflow(
    model_path=Path("models/comprehensive_loco_cv/final_model_calibrated_per_class.pkl"),
    use_calibration=True,
    # ... other parameters
)
```

## Feature Parity with Gene CV

This LOCO-CV implementation provides **complete feature parity** with gene-aware CV:

| Feature Category | Gene CV | LOCO CV | Status |
|------------------|---------|---------|---------|
| **CV Metrics Visualization** | 7 plots | 7 plots | ✅ **Identical** |
| **ROC/PR Curves** | Enhanced | Enhanced | ✅ **Identical** |
| **Feature Importance** | 4 methods | 4 methods | ✅ **Identical** |
| **SHAP Analysis** | Enhanced | Enhanced | ✅ **Identical** |
| **Feature Leakage Detection** | Full | Full | ✅ **Identical** |
| **Calibration Support** | Per-class | Per-class | ✅ **Identical** |
| **Diagnostic Functions** | 8 functions | 8 functions | ✅ **Identical** |
| **Transcript-Level Analysis** | Supported | Supported | ✅ **Identical** |

### Key Advantages Over Gene CV
1. **Memory Efficiency**: Chunked loading and sparse matrices for large datasets
2. **Out-of-Distribution Testing**: Chromosome-wise evaluation for generalization assessment
3. **Scalability**: Designed for datasets with 20K+ genes
4. **Advanced Memory Management**: Optimized for memory-constrained environments

## Technical Details

### Memory Optimization Strategies
1. **Chunked Data Loading**: Process data in manageable batches
2. **Sparse Matrix Support**: Efficient storage of high-dimensional features
3. **Feature Selection**: Automated reduction of feature dimensionality
4. **Garbage Collection**: Aggressive memory cleanup after each fold
5. **Selective Loading**: Only load necessary columns for analysis

### Enhanced Analysis Components

#### Feature Importance Analysis
Four complementary methods provide comprehensive feature insights:
- **XGBoost Internal**: Model-native importance scores
- **Statistical Tests**: P-values and significance testing
- **Effect Sizes**: Magnitude of feature effects
- **Mutual Information**: Information-theoretic relationships

#### SHAP Analysis
Memory-efficient SHAP implementation with:
- Incremental processing for large datasets
- Per-class explanations
- Comprehensive visualizations
- Automatic fallback mechanisms

#### CV Metrics Visualization
Seven publication-ready plots:
- Performance comparisons across folds
- Statistical confidence intervals
- Error reduction analysis
- Multi-metric dashboards

## Unseen Position Analysis

The script includes specialized functionality for analyzing unseen genomic positions:

1. **Generates features** for positions not seen during training
2. **Applies trained models** to these unseen positions
3. **Performs neighborhood analysis** around selected positions
4. **Creates visualizations** of prediction patterns

This provides valuable insight into model generalization to completely unseen genomic regions.

## Performance Expectations

### Typical Results (Leave-One-Chromosome-Out)
- **F1 Improvement**: 25-35% improvement over base model (more conservative than gene CV)
- **Error Reduction**: 200-600 fewer errors per fold
- **Top-K Accuracy**: 80-90% chromosome-level top-k accuracy
- **Processing Time**: 2-5 hours for full genome analysis

### Resource Requirements

| Configuration | Memory | Disk Space | Recommended CPU |
|---------------|--------|------------|-----------------|
| **Standard** | 16-64 GB | 3-8 GB | Multi-core (12+ cores) |
| **Memory-Optimized** | 4-16 GB | 5-15 GB | Multi-core (8+ cores) |
| **Large-Scale** | 32-128 GB | 10-30 GB | Multi-core (16+ cores) |

### Scalability Guidelines
- **Small datasets** (<1K genes): Use standard configuration
- **Medium datasets** (1K-10K genes): Use memory optimization
- **Large datasets** (10K+ genes): Use chunked loading and sparse matrices
- **Very large datasets** (50K+ genes): Use all optimization features

## FAQ

**Q: What's the difference between this and gene-aware CV?**  
LOCO-CV provides out-of-distribution testing by training on all chromosomes except one, while gene CV keeps correlated genes together. LOCO-CV is more conservative and tests extreme generalization.

**Q: Which should I use: gene CV or LOCO CV?**  
Use gene CV for in-domain evaluation and LOCO CV for out-of-distribution testing. Ideally, run both for comprehensive model assessment.

**Q: How do I handle memory issues?**  
Use `--use-chunked-loading`, `--use-sparse-kmers`, `--memory-optimize`, and `--feature-selection` flags. Reduce `--chunk-size` and `--diag-sample` if needed.

**Q: Can I use specific chromosomes for testing?**  
Yes, use `--heldout-chroms "21,22,X"` to specify fixed test chromosomes instead of leave-one-out.

**Q: Are the outputs compatible with gene CV?**  
Yes, both scripts generate compatible output formats with only minor filename differences reflecting the CV methodology.

**Q: How do I interpret the unseen position analysis?**  
This analysis shows how well the model generalizes to completely unseen genomic regions, providing insight into model robustness.

**Q: Can I disable specific analysis components?**  
Yes, use flags like `--no-plot-curves`, `--diag-sample 0`, or exclude specific features to customize the analysis pipeline.

**Q: What's the computational cost compared to gene CV?**  
LOCO-CV is typically 2-3x slower due to chromosome-wise splitting and larger test sets, but provides more rigorous evaluation.

---

*Last updated: 2025-01-15*
