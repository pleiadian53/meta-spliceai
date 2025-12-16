# Gene-wise Cross-Validation with Sigmoid Ensemble

**⚠️ DEPRECATED:** This documentation is outdated. Please use **[Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md)** instead.

**Last Updated:** Pre-January 2025 (OUTDATED)  
**Status:** ❌ **SUPERSEDED**  
**Replaced By:** [Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md)

File: `meta_spliceai/splice_engine/meta_models/training/run_gene_cv_sigmoid.py`

## Purpose

This module implements a **gene-wise K-fold cross-validation** for meta-model training using a **sigmoid ensemble architecture** instead of a single multi-class model. The script provides comprehensive analysis capabilities including visualization, feature importance analysis, and quality control.

### Key Advantages

* **Sigmoid Ensemble Architecture**: Three independent binary XGBoost classifiers (one-vs-rest for neither/donor/acceptor)
* **Per-Class Calibration**: Individual calibration for each splice site class for improved probability estimates
* **Comprehensive Analysis Pipeline**: 7 different analytical methods for complete model evaluation
* **Enhanced Visualization**: Publication-ready plots and comprehensive reporting
* **Feature Quality Control**: Leakage detection and automated feature filtering

## Recent Major Enhancements (2025-01-15)

### 🔧 Critical Position-Based Evaluation Fix
- **Problem Resolved**: Fixed systematic 209bp position offsets in evaluation that were causing all TP=0 results
- **Root Cause**: Meta-models were being evaluated for position discovery rather than classification performance
- **Solution**: Replaced position-based evaluation with proper classification-based evaluation methods:
  - `meta_splice_performance_correct()`: Compares meta vs base at training positions
  - `meta_splice_performance_simple()`: Gene-level classification comparison
- **Impact**: Evaluation now shows realistic performance improvements (30-40% F1 improvement vs 99%+ unrealistic gains)

### 🎯 Enhanced Feature Importance Analysis
- **XGBoost Ensemble Support**: Now properly handles XGBoost feature importance for ensemble models
- **Comprehensive Aggregation**: Feature importance across all ensemble components (`weight`, `gain`, `cover`, `total_gain`, `total_cover`)
- **Statistical Method Improvements**: Enhanced hypothesis testing with:
  - Binary target encoding (splice sites vs non-splice sites) for better statistical power
  - Multiple statistical tests: Shapiro-Wilk → Welch's t-test/Mann-Whitney U for numerical features
  - Fisher's Exact Test/Chi-square for categorical features
  - Benjamini-Hochberg FDR correction for multiple comparisons
- **Enhanced Debugging**: Comprehensive console output showing statistical summaries and significance counts

### 📊 Improved Evaluation Methodology
- **Classification-Based Approach**: Proper evaluation of meta-model classification accuracy improvements
- **Meaningful Metrics**: Realistic TP/FP/FN counts and accuracy/F1 improvements
- **Better Performance Summaries**: Enhanced `display_comprehensive_performance_summary()` with correct file detection
- **Evaluation Methodology Documentation**: Clear explanation of why classification-based evaluation is correct

## Enhanced Analysis Pipeline

### 1. CV Metrics Visualization (7 Plots)
- **F1 Score Comparison**: Base vs Meta across CV folds with statistical summaries
- **ROC AUC Comparison**: Area under curve analysis with confidence intervals
- **Average Precision Comparison**: Precision-recall performance evaluation
- **Error Reduction Analysis**: Visual analysis of delta_fp and delta_fn improvements
- **Performance Overview**: Multi-metric dashboard with accuracy breakdowns
- **Improvement Summary**: Percentage gains and quadrant analysis
- **Top-k Analysis**: Gene-level accuracy patterns

### 2. Enhanced ROC/PR Curves
- **Base Model Analysis**: Comprehensive curves for SpliceAI baseline
- **Meta Model Analysis**: Enhanced curves for sigmoid ensemble
- **Multiclass Analysis**: Per-class ROC/PR curves with macro-averaging
- **Comparison Visualizations**: Side-by-side performance comparison

### 3. Comprehensive Feature Importance Analysis
Four complementary analytical methods:
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

## Key Presentation Highlights

### 🏆 Performance Improvements
- **Realistic F1 Gains**: 30-40% improvement over base model *(post-evaluation-fix)*
- **Error Reduction**: 300-800 fewer errors per fold
- **Top-K Accuracy**: 85-95% gene-level accuracy
- **Statistical Significance**: Proper statistical validation of improvements

### 📈 Analysis Capabilities
- **7 Comprehensive Visualizations**: Publication-ready plots for all key metrics
- **4 Feature Importance Methods**: Multi-perspective analysis of feature contributions
- **Enhanced SHAP Integration**: Memory-efficient explanations for large datasets
- **Quality Control**: Automated detection and handling of problematic features

### 🔬 Technical Innovation
- **Sigmoid Ensemble Architecture**: Superior to single multiclass models
- **Per-Class Calibration**: Improved probability estimates for each splice site type
- **Gene-Aware CV**: Prevents information leakage between correlated splice sites
- **Comprehensive Validation**: Multiple evaluation approaches ensure robust results

## Entry Point

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/training/features \
    --out-dir models/sigmoid_ensemble \
    --n-folds 5
```

## ⚠️ CRITICAL: Default Row Cap Issue

**IMPORTANT**: The current default `--row-cap 100000` is **NOT suitable for production training**. This severely limits the meta-model's learning capacity by using only 100,000 training instances instead of the full dataset (typically 1.3M+ instances).

**For production meta-model training, ALWAYS use `--row-cap 0`** to ensure the model learns from all available splice patterns.

The 100,000 default exists for development/testing purposes only. The memory management should be handled through:
- Incremental training mechanisms
- Data chunking strategies  
- Memory optimization flags (`--memory-optimize`)
- Reduced diagnostic samples (`--diag-sample`, `--max-diag-sample`)

**Each row represents a position-centered training instance** - limiting this artificially constrains the model's ability to learn comprehensive splicing patterns.

### 🔧 Recommended Fix for Default Row Cap

**The current default should be changed in the source code from:**
```python
p.add_argument("--row-cap", type=int, default=100_000)
```

**To:**
```python
p.add_argument("--row-cap", type=int, default=0)  # Use full dataset by default
```

**Until this fix is implemented, ALWAYS explicitly specify `--row-cap 0` in production commands.**

## 🔗 Systematic Resource Management

The gene-aware CV script integrates with the `meta_spliceai.system.genomic_resources` system for **systematic path resolution**. This ensures consistent data location across all workflows in the system.

### 📋 **Systematic Path Resolution**

Instead of hardcoded paths, the script uses **systematic resource management**:

| Resource Type | Systematic Path | Purpose |
|---------------|-----------------|---------|
| **Splice Sites** | `data/ensembl/splice_sites.tsv` | Splice site coordinates from GTF exon boundaries |
| **Gene Features** | `data/ensembl/spliceai_analysis/gene_features.tsv` | Gene-level annotations from GTF analysis |
| **Transcript Features** | `data/ensembl/spliceai_analysis/transcript_features.tsv` | Transcript-level annotations from GTF analysis |
| **Feature Exclusion** | `configs/exclude_features.txt` | Training configuration for problematic features |

### 🔧 **Integration Benefits**

1. **Consistent Paths**: All workflows use the same systematic organization
2. **Multi-Environment Support**: Development, Production, Lakehouse environments
3. **Multi-Genome Support**: GRCh37, GRCh38 with automatic path resolution
4. **Graceful Fallback**: Falls back to hardcoded paths if systematic management unavailable
5. **Validation**: Automatic validation of resource availability before training

### ⚙️ **How It Works**

```python
# Instead of hardcoded defaults:
p.add_argument("--splice-sites-path", default="data/ensembl/splice_sites.tsv")

# Uses systematic resource management:
from .systematic_defaults import get_systematic_defaults
defaults = get_systematic_defaults()
p.add_argument("--splice-sites-path", default=defaults["splice_sites_path"])
```

### 🔍 **Resource Validation**

The script automatically validates systematic resources:

```bash
✅ All critical systematic resources validated
   ✅ splice_sites: /path/to/data/ensembl/splice_sites.tsv
   ✅ gene_features: /path/to/data/ensembl/spliceai_analysis/gene_features.tsv
   ✅ transcript_features: /path/to/data/ensembl/spliceai_analysis/transcript_features.tsv
```

### 💡 **Manual Override**

You can still override systematic paths manually:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/custom_run \
    --splice-sites-path /custom/path/to/splice_sites.tsv \
    --gene-features-path /custom/path/to/gene_features.tsv \
    --row-cap 0
```

## Command-Line Arguments

### Core Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Path to feature dataset | *(required)* |
| `--out-dir` | Output directory for results | *(required)* |
| `--gene-col` | Column name for gene IDs | `gene_id` |
| `--n-folds` | Number of CV folds | 5 |
| `--valid-size` | Validation set size (fraction) | 0.1 |
| `--row-cap` | Maximum training instances to use (0=use all) | 100,000 ⚠️ |

### Model Configuration
| Argument | Description | Default |
|----------|-------------|---------|
| `--tree-method` | XGBoost tree method (`hist`, `gpu_hist`, `approx`) | `hist` |
| `--n-estimators` | Number of trees | 800 |
| `--max-bin` | Maximum bins for histogram tree method | 256 |
| `--device` | Device for training (`auto`, `cpu`, `cuda`) | `auto` |
| `--seed` | Random seed for reproducibility | 42 |

### Calibration Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--calibrate` | Enable binary splice/non-splice calibration | *(flag)* |
| `--calibrate-per-class` | Enable per-class calibration (recommended) | *(flag)* |
| `--calib-method` | Calibration method (`platt`, `isotonic`) | `platt` |
| `--calibration-analysis` | Run comprehensive calibration analysis | *(flag)* |
| `--calibration-sample` | Sample size for calibration analysis | *(use all data)* |
| `--quick-overconfidence-check` | Quick overconfidence detection | *(enabled by default)* |
| `--no-overconfidence-check` | Disable overconfidence detection | *(flag)* |

### Visualization and Analysis
| Argument | Description | Default |
|----------|-------------|---------|
| `--plot-curves` | Generate ROC/PR curves | *(enabled by default)* |
| `--no-plot-curves` | Disable curve plotting | *(flag)* |
| `--plot-format` | Plot format (`pdf`, `png`, `svg`) | `pdf` |
| `--n-roc-points` | Points for ROC curve averaging | 101 |

### Feature Quality Control
| Argument | Description | Default |
|----------|-------------|---------|
| `--check-leakage` | Enable leakage detection | *(enabled by default)* |
| `--no-leakage-check` | Disable leakage detection | *(flag)* |
| `--leakage-threshold` | Correlation threshold for leaky features | 0.95 |
| `--auto-exclude-leaky` | Automatically exclude leaky features | *(flag)* |
| `--exclude-features` | Path to feature exclusion file or comma-separated list | *(systematic path resolution)* |
| `--leakage-probe` | Run comprehensive leakage analysis | *(flag)* |

### Diagnostic Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--diag-sample` | Sample size for diagnostics | 25,000 |
| `--neigh-sample` | Sample size for neighborhood analysis | 0 |
| `--neigh-window` | Window size for neighborhood analysis | 10 |
| `--sample-genes` | Sample only N genes for faster testing | *(use all genes)* |
| `--verbose` | Enable verbose output for debugging | *(flag)* |
| `--skip-eval` | Skip all evaluation steps after training | *(flag)* |

### Transcript-Level Analysis
| Argument | Description | Default |
|----------|-------------|---------|
| `--transcript-topk` | Enable transcript-level top-k accuracy | *(flag)* |
| `--no-transcript-cache` | Disable transcript mapping cache | *(flag)* |
| `--splice-sites-path` | Path to splice site annotations | *(systematic path resolution)* |
| `--transcript-features-path` | Path to transcript features | *(systematic path resolution)* |
| `--gene-features-path` | Path to gene features | *(systematic path resolution)* |
| `--position-col` | Column name for genomic positions | `position` |
| `--chrom-col` | Column name for chromosome | `chrom` |
| `--top-k` | Top-k value for accuracy analysis | 5 |
| `--include-tns` | Include true negatives in evaluation | *(flag)* |

### Base Model Comparison
| Argument | Description | Default |
|----------|-------------|---------|
| `--donor-score-col` | Column with raw donor probability from base model | `donor_score` |
| `--acceptor-score-col` | Column with raw acceptor probability from base model | `acceptor_score` |
| `--splice-prob-col` | Column with combined splice probability (fallback) | `score` |
| `--base-thresh` | Threshold for base model splice site calls | 0.5 |
| `--threshold` | Threshold for meta model splice site calls | *(auto-determined)* |
| `--base-tsv` | Base model predictions file | *(optional)* |

### Overfitting Monitoring
| Argument | Description | Default |
|----------|-------------|---------|
| `--monitor-overfitting` | Enable comprehensive overfitting analysis | *(flag)* |
| `--overfitting-threshold` | Performance gap threshold for overfitting detection | 0.05 |
| `--early-stopping-patience` | Patience for early stopping detection | 20 |
| `--convergence-improvement` | Minimum improvement threshold for convergence | 0.001 |

### Memory Optimization
| Argument | Description | Default |
|----------|-------------|---------|
| `--memory-optimize` | Enable memory optimization (**diagnostics only, NOT training**) | *(flag)* |
| `--max-diag-sample` | Maximum diagnostic sample size for memory optimization | 25,000 |

**⚠️ Important**: `--memory-optimize` does **NOT** affect the training sample size or model training process. It only reduces:
- Diagnostic sample sizes (limited to `--max-diag-sample`)
- Feature importance analysis samples (reduced to 10,000)
- Neighbor analysis samples (reduced to 1,000)

The meta-model training always uses the full dataset as specified by `--row-cap`.

### Error Analysis
| Argument | Description | Default |
|----------|-------------|---------|
| `--errors-only` | Focus analysis on error cases only | *(flag)* |
| `--error-artifact` | Path to error artifact file (TP/FP/FN labels) | *(auto-search)* |

### Legacy/Deprecated
| Argument | Description | Default |
|----------|-------------|---------|
| `--annotations` | **DEPRECATED** - Use `--splice-sites-path` instead | *(backward compatibility)* |

## Output Structure

The script creates a comprehensive output directory with complete analysis results:

```
<out_dir>/
├── gene_cv_metrics.csv                          # Main CV results
├── metrics_aggregate.json                       # Summary statistics
├── metrics_fold{N}.json                         # Per-fold metrics
├── metrics_folds.tsv                            # Fold summary table
├── model_multiclass.pkl                         # Final trained model
├── feature_manifest.csv                         # Feature list
├── feature_selection_info.json                  # Feature selection details
├── cv_metrics_visualization/                    # 7 visualization plots
│   ├── f1_comparison.pdf
│   ├── auc_comparison.pdf
│   ├── average_precision_comparison.pdf
│   ├── error_reduction_analysis.pdf
│   ├── performance_overview.pdf
│   ├── improvement_summary.pdf
│   └── top_k_analysis.pdf
├── roc_pr_curves_base.pdf                       # Base model ROC/PR
├── roc_pr_curves_meta.pdf                       # Meta model ROC/PR
├── multiclass_roc_curves.pdf                    # Multi-class ROC
├── multiclass_pr_curves.pdf                     # Multi-class PR
├── shap_analysis/                               # SHAP reports
│   ├── shap_summary.pdf
│   ├── shap_dependence_plots.pdf
│   └── shap_feature_importance.csv
├── feature_importance_analysis/                 # Multi-method analysis
│   ├── feature_importance_comprehensive.xlsx
│   ├── xgboost_importance.csv
│   ├── statistical_tests.csv
│   ├── effect_sizes.csv
│   └── mutual_information.csv
├── feature_label_correlations.csv               # Leakage analysis
├── fold_0/                                      # Per-fold outputs
│   ├── model_sigmoid.pkl
│   ├── metrics.json
│   ├── importance.csv
│   └── calibration/
  ├── fold_1/
  │   └── ...
└── [comprehensive diagnostic files]
```

## Usage Examples

### Basic Usage
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_basic \
    --n-folds 5
```

## Usage Examples by Complexity Level

### 🚀 Quick Start (Minimal Command)
For basic testing with good defaults - most switches have sensible defaults:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_quick_test \
    --row-cap 0
```

**What this does:**
- ✅ **Uses full dataset** (`--row-cap 0` overrides problematic 100,000 default)
- 5-fold CV (default)
- 100 estimators (default) 
- Basic evaluation with ROC/PR curves
- ~30 minutes runtime

**⚠️ Critical**: Without `--row-cap 0`, only 100k/1.3M+ instances would be used!

### 🎯 Recommended Production (Balanced)
Optimized for performance without excessive complexity:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_pc_1000_3mers_production \
    --n-estimators 800 \
    --row-cap 0 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --monitor-overfitting \
    --transcript-topk \
    --splice-sites-path data/ensembl/splice_sites.tsv \
    --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
    --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv \
    --verbose
```

**Key additions over minimal:**
- Higher model complexity (800 estimators)
- Per-class calibration for better probabilities
- Automatic leaky feature removal
- Overfitting monitoring with early stopping
- Transcript-level analysis
- ~2-3 hours runtime

### 🧬 Large-Scale Regulatory Analysis
For advanced meta-learning with regulatory genes and multi-scale k-mer features:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/gene_cv_reg_10k_kmers_run1 \
    --n-estimators 800 \
    --row-cap 0 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --monitor-overfitting \
    --calibration-analysis \
    --neigh-sample 5000 \
    --early-stopping-patience 30 \
    --verbose 2>&1 | tee -a logs/gene_cv_reg_10k_kmers_run1.log
```

**Dataset Characteristics:**
- **Size**: 9,280 genes (62.3% protein_coding, 37.7% lncRNA), ~3.7M records, 1.91 GB
- **Features**: 1,167 total (3-mer + 5-mer k-mers + genomic features)
- **Purpose**: Advanced regulatory variant analysis and alternative splicing patterns
- **Documentation**: [train_regulatory_10k_kmers](../../../case_studies/data_sources/datasets/train_regulatory_10k_kmers/)

**Key Features:**
- Large-scale regulatory gene dataset
- Multi-scale k-mer analysis (3-mer + 5-mer)
- Enhanced calibration analysis for regulatory patterns
- Increased neighbor sampling for comprehensive context
- Extended early stopping patience for complex patterns
- ~6-8 hours runtime with comprehensive logging

### 🔬 Full Analysis (Research/Development)
Comprehensive analysis with all diagnostics - replicates successful Run 4:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_pc_1000_3mers_run_5 \
    --n-folds 5 --n-estimators 800 \
    --row-cap 0 \
    --calibrate-per-class --calib-method platt \
    --plot-curves --plot-format pdf --n-roc-points 101 \
    --check-leakage --leakage-threshold 0.95 --auto-exclude-leaky \
    --monitor-overfitting --overfitting-threshold 0.05 \
    --early-stopping-patience 30 --convergence-improvement 0.001 \
    --diag-sample 25000 --neigh-sample 1000 --neigh-window 10 \
    --transcript-topk \
    --splice-sites-path data/ensembl/splice_sites.tsv \
    --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
    --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv \
    --calibration-analysis --quick-overconfidence-check \
    --verbose
```

**Key Results from Run 4:**
- F1 Score Improvement: 47.2% (0.648 → 0.954)
- Error Reduction: 47,687 total errors reduced (59.9% FP, 78.3% FN)
- Top-k Accuracy: 96.8% gene-level accuracy
- Features: 124 total (including 3-mer k-mers), 11 automatically excluded
- Training Time: ~2-3 hours with comprehensive analysis
- Memory Usage: ~8-16 GB RAM with full diagnostics

**Critical Parameters for Success:**
- `--n-estimators 800`: Optimal based on overfitting analysis (recommended: 416)
- `--calibrate-per-class`: Essential for multi-class probability calibration
- `--monitor-overfitting`: Prevents overfitting with early stopping
- `--diag-sample 25000`: Balanced sample size for comprehensive diagnostics
- `--auto-exclude-leaky`: Automatically removes problematic features

### 🎛️ Memory-Constrained Systems
For systems with limited RAM (< 16GB):

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_memory_optimized \
    --n-estimators 400 \
    --row-cap 500000 \
    --memory-optimize \
    --diag-sample 10000 \
    --neigh-sample 500 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --verbose
```

**Memory optimizations:**
- Reduced estimators (400 vs 800)
- Row cap to limit dataset size
- Smaller diagnostic samples
- Memory optimization flag enabled
- ~1-2 hours runtime, ~4-8 GB RAM

### 🧪 Development/Testing
For quick iterations during development:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/gene_cv_dev_test \
    --n-estimators 50 \
    --sample-genes 100 \
    --diag-sample 5000 \
    --no-plot-curves \
    --skip-eval \
    --verbose
```

**Development optimizations:**
- Very few estimators (50)
- Sample only 100 genes
- Minimal diagnostics
- Skip plotting and detailed evaluation
- ~5-10 minutes runtime

## 📋 Complete Switch Reference

### 🔧 Switches with Good Defaults (Usually Don't Need to Specify)

| Switch | Default Value | When to Override |
|--------|---------------|------------------|
| `--n-folds` | 5 | For different CV strategies |
| `--n-estimators` | 100 | Always override to 400-800 for production |
| `--row-cap` | 100,000 ⚠️ | **ALWAYS override to `0`** for production (full dataset) |
| `--plot-curves` | True | Use `--no-plot-curves` for speed |
| `--check-leakage` | True | Use `--no-leakage-check` for speed |
| `--leakage-threshold` | 0.95 | Rarely need to change |
| `--calib-method` | platt | isotonic sometimes better for large datasets |
| `--overfitting-threshold` | 0.05 | Rarely need to change |
| `--early-stopping-patience` | 20 | Use 30+ for more thorough training |
| `--convergence-improvement` | 0.001 | Rarely need to change |
| `--plot-format` | pdf | Change to png/svg if needed |
| `--n-roc-points` | 101 | Rarely need to change |
| `--neigh-window` | 10 | Sometimes increase to 12-15 |
| `--quick-overconfidence-check` | True | Use `--no-overconfidence-check` to disable |

### ⚙️ Essential Switches (Always Specify)

| Switch | Purpose | Recommended Values |
|--------|---------|-------------------|
| `--dataset` | Input dataset path | Your dataset path |
| `--out-dir` | Output directory | Unique name for each run |
| `--row-cap` | Maximum rows to use for training | `0` for full dataset, `500000` for memory constraints (default: 100,000) |

### 🎯 Performance-Critical Switches

| Switch | Purpose | Recommended Values | Impact |
|--------|---------|-------------------|--------|
| `--n-estimators` | Model complexity | `400-800` | Higher = better performance, longer training |
| `--calibrate-per-class` | Multi-class calibration | Always enable | Essential for good probabilities |
| `--auto-exclude-leaky` | Remove problematic features | Always enable | Prevents overfitting |
| `--monitor-overfitting` | Early stopping | Always enable | Prevents overfitting |

### 📊 Analysis & Diagnostics Switches

| Switch | Purpose | When to Use | Performance Cost |
|--------|---------|-------------|------------------|
| `--diag-sample` | Diagnostic sample size | `25000` for full analysis, `10000` for speed | Medium |
| `--neigh-sample` | Neighborhood analysis | `1000` recommended | Low |
| `--transcript-topk` | Transcript-level metrics | Research/production runs | Medium |
| `--calibration-analysis` | Detailed calibration | Research runs only | High |
| `--leakage-probe` | Deep leakage analysis | Debugging only | High |

### 🗂️ Data Path Switches (Required for Transcript Analysis)

```bash
# Required together for --transcript-topk
--splice-sites-path data/ensembl/splice_sites.tsv \
--transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
--gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv
```

### 🚀 Memory & Performance Switches

| Switch | Purpose | When to Use |
|--------|---------|-------------|
| `--row-cap` | **Memory Control** | **CRITICAL**: See detailed explanation below |
| `--memory-optimize` | Reduce memory usage | Systems with < 16GB RAM |
| `--max-diag-sample` | Cap diagnostic memory | Memory-constrained systems |
| `--sample-genes` | Limit gene count | Development/testing only |
| `--device` | GPU/CPU selection | `cuda` if available |
| `--tree-method` | XGBoost method | `gpu_hist` for GPU, `hist` for CPU |

#### 🎯 Understanding `--row-cap` (Training Instance Limitation)

**Purpose**: Controls the maximum number of **position-centered training instances** used for meta-model training.

**⚠️ CRITICAL ISSUE**: The default 100,000 limit is **fundamentally inappropriate** for production meta-model training.

**How it works**:
- **Default**: 100,000 instances (**PROBLEMATIC** - severely limits learning)
- **`--row-cap 0`**: Use full dataset (**REQUIRED** for production training)
- **`--row-cap N`**: Limit to N instances (development/testing only)
- **Gene-aware sampling**: When cap is active, samples complete genes (preserves biological structure)

**Why the default is wrong**:
```bash
# Training capacity comparison:
--row-cap 100000    # Only 7.5% of typical dataset (100k/1.3M instances)
--row-cap 0         # Full dataset - learns from ALL splice patterns
```

**Each row = one training instance** representing a genomic position with splice context. Limiting this artificially constrains the model's ability to learn comprehensive splicing patterns across:
- Different gene types and structures
- Diverse sequence contexts  
- Rare but important splice variants
- Tissue-specific splicing patterns

**Proper memory management should use**:
- `--memory-optimize` flag for low-memory systems
- `--max-diag-sample` to limit diagnostic memory usage
- Incremental training mechanisms (built into the script)
- Data chunking strategies (handled internally)

**When to use each setting**:
- **Production**: `--row-cap 0` (**ALWAYS** - use full dataset)
- **Development/Testing**: `--row-cap 100000` (quick iterations only)
- **Never use**: Intermediate values like 500k (either test with small data or train with all data)

### 🎨 Output & Visualization Switches

| Switch | Purpose | Options |
|--------|---------|---------|
| `--plot-format` | Plot file format | `pdf`, `png`, `svg` |
| `--no-plot-curves` | Disable plotting | For speed in development |
| `--verbose` | Detailed logging | Always recommended |

### 🔍 Advanced Analysis Switches

| Switch | Purpose | Use Case |
|--------|---------|----------|
| `--errors-only` | Focus on error cases | Error analysis studies |
| `--include-tns` | Include true negatives | Comprehensive evaluation |
| `--exclude-features` | Manual feature exclusion | Custom feature sets |

## 💡 Practical Recommendations

### 🎯 Command Selection Guide

**Choose your command based on your goal:**

1. **🚀 Quick Test** → Use minimal command (3 switches)
2. **🎯 Production Run** → Use recommended production (8-10 switches)  
3. **🔬 Research Analysis** → Use full analysis (15+ switches)
4. **🎛️ Memory Issues** → Use memory-constrained version
5. **🧪 Development** → Use development/testing version

### 📈 Performance vs Complexity Trade-offs

| Aspect | Minimal | Production | Full Analysis |
|--------|---------|------------|---------------|
| **Runtime** | 30 min | 2-3 hours | 3-4 hours |
| **Memory** | 4-8 GB | 8-16 GB | 16-32 GB |
| **Switches** | 3 | 10 | 15+ |
| **F1 Score** | ~0.92 | ~0.95 | ~0.95+ |
| **Diagnostics** | Basic | Good | Comprehensive |

### 🔧 Most Important Switches to Remember

**Always specify these 4 for production:**
```bash
--n-estimators 800          # Higher performance
--row-cap 0                 # CRITICAL: Use full dataset (override harmful 100k default)
--calibrate-per-class       # Better probabilities
--auto-exclude-leaky        # Prevent overfitting
```

**Add these 3 for research:**
```bash
--monitor-overfitting       # Early stopping
--transcript-topk           # Detailed metrics
--calibration-analysis      # Probability analysis
```

### ⚠️ Common Pitfalls to Avoid

1. **Don't use default `--n-estimators 100`** → Always use 400-800
2. **Don't forget `--row-cap 0`** → **CRITICAL**: Default 100k severely limits learning capacity
3. **Don't use `--calibrate` alone** → Use `--calibrate-per-class` instead
4. **Don't skip `--auto-exclude-leaky`** → Manual feature exclusion is tedious
5. **Don't run full analysis during development** → Use `--sample-genes` for speed

### 🎨 Customization Examples

**GPU-Accelerated (if available):**
```bash
--device cuda --tree-method gpu_hist
```

**Large Dataset Optimization:**
```bash
--calib-method isotonic --max-diag-sample 50000
```

**Publication-Quality Plots:**
```bash
--plot-format svg --n-roc-points 201
```

**Debugging Feature Issues:**
```bash
--leakage-probe --exclude-features problematic_features.txt
```

---

## 🔗 Additional Resources

## Integration with Inference Workflow

Models trained with this script can be directly used in the enhanced splice inference workflow:

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_inference_workflow import run_enhanced_splice_inference_workflow

run_enhanced_splice_inference_workflow(
    model_path=Path("models/comprehensive_gene_cv/model_multiclass.pkl"),
    use_calibration=True,
    # ... other parameters
)
```

## Technical Details

### Sigmoid Ensemble Architecture
The sigmoid ensemble consists of:

1. **Feature preprocessing** - Standard preprocessing pipeline with optional chromosome encoding
2. **Binary models** - Three XGBoost binary classifiers:
   * Neither vs. rest
   * Donor vs. rest
   * Acceptor vs. rest
3. **Ensemble wrapper** - `SigmoidEnsemble` or `PerClassCalibratedSigmoidEnsemble`
4. **Optional calibration** - Isotonic or Platt calibration per class

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

## Performance Expectations

### Typical Results (5-fold Gene CV)
- **F1 Improvement**: 30-50% improvement over base model (Run 4: 47.2%)
- **Error Reduction**: 7,000-10,000 fewer errors per fold (Run 4: 9,537 avg)
- **Top-K Accuracy**: 85-97% gene-level top-k accuracy (Run 4: 96.8%)
- **Processing Time**: 2-4 hours for 1000 genes with full analysis
- **ROC AUC**: 0.999+ for meta-model vs 0.989 for base model
- **Precision/Recall**: Balanced improvements across all splice site classes

### Resource Requirements
- **Memory**: 8-32 GB RAM depending on dataset size and analysis depth
- **Disk**: 2-5 GB for comprehensive outputs
- **CPU**: Multi-core recommended (8+ cores) for SHAP analysis
- **GPU**: Optional for XGBoost training acceleration

### Scalability Guidelines
- **Small datasets** (<1K genes): Use all features with full analysis
- **Medium datasets** (1K-10K genes): Consider feature filtering
- **Large datasets** (10K+ genes): Use memory optimization and consider switching to LOCO-CV

## Statistical Methods Used

### Hypothesis Testing Approach
- **Target Encoding**: Binary classification (splice sites vs non-splice sites) for better statistical power
- **Numerical Features**: Shapiro-Wilk normality test → Welch's t-test (normal) or Mann-Whitney U (non-normal)
- **Categorical Features**: Fisher's Exact Test (2×2 tables) or Chi-square test (larger tables)
- **Multiple Testing Correction**: Benjamini-Hochberg FDR correction

### Effect Size Measures
- **Cohen's d**: For numerical features with normal distributions
- **Rank-biserial correlation**: For non-parametric comparisons
- **Cramer's V**: For categorical associations
- **Interpretation**: Small (0.2), Medium (0.5), Large (0.8) effect sizes

## FAQ

**Q: What's the difference between this and the original gene CV?**  
This script provides a complete analysis pipeline with visualization, feature importance, and quality control, while the original focused only on basic training and evaluation.

**Q: Why was the evaluation methodology changed?**  
The original position-based evaluation was fundamentally flawed for meta-models, which classify existing training instances rather than discover new positions. The new classification-based approach provides realistic performance metrics.

**Q: Should I use calibration?**  
Yes, per-class calibration (`--calibrate-per-class`) is recommended for improved probability estimates, especially for class-imbalanced datasets.

**Q: How do I interpret the feature importance results?**  
The script generates four different importance analyses - focus on features that rank highly across multiple methods for the most robust insights.

**Q: Can I disable specific analysis components?**  
Yes, use flags like `--no-plot-curves`, `--diag-sample 0`, or exclude specific features to customize the analysis pipeline.

**Q: What if I run out of memory?**  
Use `--row-cap`, `--filter-features`, reduce `--diag-sample`, or consider switching to the LOCO-CV script for better memory management.

**Q: How do I choose the right number of folds?**  
5-fold CV is standard; use fewer folds (3) for very small datasets or more folds (10) for very large datasets with many genes.

**Q: What's the optimal n_estimators setting?**  
Use `--monitor-overfitting` to automatically determine the best value. Run 4 used 800 estimators but analysis suggested 416 would be optimal. Start with 800 and let early stopping handle optimization.

**Q: Which calibration method should I use?**  
Use `--calibrate-per-class --calib-method platt` for best results. This provides separate calibration for each splice site class and handles class imbalance better than binary calibration.

**Q: How do I handle memory issues with large datasets?**  
Use `--memory-optimize --max-diag-sample 10000 --diag-sample 10000` and reduce `--neigh-sample` to 500. Consider using `--row-cap` to limit dataset size during development.

**Q: What if SHAP analysis fails?**  
SHAP analysis can fail with non-numeric features or memory constraints. The script automatically falls back to other feature importance methods. Use `--memory-optimize` or reduce `--diag-sample` if needed.

---

*Last updated: 2025-01-15*
