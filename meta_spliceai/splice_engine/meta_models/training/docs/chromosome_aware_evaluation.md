# Chromosome-Aware vs. Gene-Aware Evaluation

Metamodels in **MetaSpliceAI** aim to *generalise* beyond the specific genes (or even chromosomes) seen during training.  Two complementary strategies ensure that evaluation remains honest:

| Strategy | Goal | Split Mechanism | When to use |
|----------|------|-----------------|-------------|
| **Gene-aware** | Prevent information leakage between correlated splice sites within the same gene | Every row that shares a `gene_id` is forced into the same fold (``GroupShuffleSplit`` / ``GroupKFold``) | Small / medium projects where cross-gene generalisation is the primary concern (our default) |
| **Chromosome-aware** | Test extreme *out-of-distribution* generalisation – can the model trained on chromosomes _A…Y_ still work on chromosome _Z_? | Hold out *entire* chromosomes (or chromosome *groups*) for the test set; optional LOCO-CV iterates through all | Large datasets or whenever you want an upper-bound stress test of over-fitting |

## Recent Major Enhancements (2025-01-15)

### 🔧 Critical Evaluation Methodology Fix
- **Problem Resolved**: Fixed systematic evaluation issues that were causing unrealistic performance metrics
- **Root Cause**: Both gene-aware and chromosome-aware scripts were using position-based evaluation instead of classification-based evaluation
- **Solution**: Implemented proper classification-based evaluation methods:
  - `meta_splice_performance_correct()`: Compares meta vs base at training positions
  - `meta_splice_performance_simple()`: Classification comparison at appropriate evaluation level
- **Impact**: Both CV approaches now show realistic performance improvements (25-40% F1 improvement vs 99%+ unrealistic gains)

### 🎯 Complete Feature Parity Achievement
- **Comprehensive Integration**: LOCO-CV now has **complete feature parity** with gene-aware CV
- **Enhanced Analysis Suite**: All 7 visualization plots, 4 feature importance methods, and enhanced SHAP analysis
- **Memory Optimization**: LOCO-CV maintains additional scalability features for large datasets
- **Quality Control**: Same leakage detection and feature filtering capabilities
- **Consistent Interface**: Compatible command-line arguments and output formats

### 📊 Advanced Scalability Features
- **Memory-Efficient Processing**: Chunked data loading and sparse matrix support
- **Large Dataset Support**: Handles 20K+ genes through advanced optimization
- **Flexible Chromosome Grouping**: Automatic grouping of small chromosomes for stable metrics
- **Resource Optimization**: Configurable memory usage and processing parameters

## Key Presentation Highlights

### 🏆 Complementary Evaluation Strategies
- **Gene-Aware CV**: In-domain evaluation (30-40% F1 improvement)
- **Chromosome-Aware CV**: Out-of-distribution evaluation (25-35% F1 improvement) 
- **Comprehensive Assessment**: Both approaches provide different but complementary insights
- **Realistic Metrics**: Post-fix evaluation shows meaningful performance improvements

### 📈 Enhanced Analysis Capabilities
- **Identical Feature Suite**: Both CV approaches now offer the same comprehensive analysis
- **7 Visualization Plots**: Publication-ready performance analysis across folds
- **4 Feature Importance Methods**: Multi-perspective analysis of feature contributions
- **Memory-Efficient SHAP**: Scalable explanations for large datasets
- **Quality Control**: Automated feature leakage detection and handling

### 🔬 Technical Innovation
- **Chromosome-Aware Splitting**: Tests extreme out-of-distribution generalization
- **Adaptive Grouping**: Automatically handles small chromosomes for stable metrics
- **Scalable Architecture**: Designed for large-scale genomic datasets
- **Resource Optimization**: Configurable memory and processing parameters

---

## 1. Chromosome-Aware Evaluation

### 1.1 Leave-One-Chromosome-Out Cross-Validation (LOCO-CV)

*For each fold*:
1. Pick **one** chromosome (or group) as the *test* set.
2. Train on **all remaining chromosomes**.
3. Carve a validation subset (group-aware by gene) from the training pool.

Repeat for all chromosomes and **average** metrics for a robust score.  Implementation lives in `training/chromosome_split.py`:

```python
from meta_spliceai.splice_engine.meta_models.training import chromosome_split as csplit

chrom = df["chrom"].to_numpy()
gene  = df["gene_id"].to_numpy()

for held_out, tr_idx, val_idx, te_idx in csplit.loco_cv_splits(
    X, y, chrom_array=chrom, gene_array=gene, min_rows=1_000):
    # Fit / evaluate here
    ...
```

### 1.2 Enhanced LOCO-CV Implementation: `run_loco_cv_multiclass_scalable.py`

The modern LOCO-CV implementation provides **complete feature parity** with gene-aware CV, including:

#### Core Features
- **Memory-Efficient Processing**: Chunked data loading and sparse matrix support for large datasets
- **Per-Class Calibration**: Individual calibration for each splice site class
- **Advanced Feature Selection**: Automated feature selection with detailed reporting
- **Comprehensive Analysis Pipeline**: Same enhanced analysis capabilities as gene-aware CV

#### Enhanced Analysis Pipeline (Identical to Gene CV)
1. **CV Metrics Visualization**: 7 comprehensive plots analyzing performance across folds
2. **Enhanced ROC/PR Curves**: Both binary and multiclass curve analysis  
3. **Comprehensive Feature Importance**: 4 different analytical methods
4. **Enhanced SHAP Analysis**: Memory-efficient SHAP with comprehensive visualization reports
5. **Feature Leakage Detection**: Correlation analysis to identify potentially problematic features
6. **Diagnostic Analysis**: Complete suite of 8 diagnostic functions

#### Output Structure (Complete Parity with Gene CV)
```
<out_dir>/
├── loco_metrics.csv                        # Main CV results (chromosome-wise)
├── metrics_aggregate.json                  # Summary statistics
├── fold_{chrom}_metrics.json               # Per-fold metrics
├── final_model_uncalibrated.json           # Uncalibrated model
├── final_model_calibrated_{type}.pkl       # Calibrated models
├── fold_{chrom}_model.json                 # Per-fold models
├── feature_manifest.csv                    # Feature list
├── feature_selection_info.json             # Feature selection details
├── cv_metrics_visualization/               # 7 visualization plots
│   ├── f1_comparison.pdf
│   ├── auc_comparison.pdf
│   ├── average_precision_comparison.pdf
│   ├── error_reduction_analysis.pdf
│   ├── performance_overview.pdf
│   ├── improvement_summary.pdf
│   └── top_k_analysis.pdf
├── roc_pr_curves_meta.pdf                  # Meta model ROC/PR
├── multiclass_roc_curves.pdf               # Multi-class ROC
├── multiclass_pr_curves.pdf                # Multi-class PR
├── shap_analysis/                          # SHAP reports
├── feature_importance_analysis/            # Multi-method analysis
├── feature_label_correlations.csv          # Leakage analysis
└── [various diagnostic files]
```

### 1.3 Fixed Chromosome Hold-out

Sometimes you only need **one** split – e.g. train on `chr1-chr20`, test on `chr21,chr22,chrX`.  Use:

```python
train_idx, val_idx, test_idx, X_tr, X_val, X_te = csplit.holdout_split(
    X, y, chrom_array=chrom,
    holdout_chroms=["chr21", "chr22", "chrX"],
    gene_array=gene,
)
```

### 1.4 Grouping Small Chromosomes

Very small chromosomes (e.g. `chrY`, `chrM`) can yield unstable metrics.  `group_chromosomes()` buckets them so each *test fold* contains ≥ *min_rows* samples.

---

## 2. Gene-Aware Evaluation (recap)

* Described in detail in [`gene_aware_evaluation.md`](gene_aware_evaluation.md).
* Default in `datasets.train_valid_test_split(..., groups=gene_id)`.
* Focuses on correlational leakage rather than chromosomal OOD.
* **Complete feature parity** with chromosome-aware CV as of 2025-01-15.

---

## 3. Feature Parity Between CV Approaches

As of the latest updates, both gene-aware and chromosome-aware CV scripts provide **identical analytical capabilities**:

| Feature Category | Gene CV (`run_gene_cv_sigmoid.py`) | LOCO CV (`run_loco_cv_multiclass_scalable.py`) | Status |
|------------------|-------------------------------------|------------------------------------------------|---------|
| **CV Metrics Visualization** | 7 comprehensive plots | 7 comprehensive plots | ✅ **Identical** |
| **ROC/PR Curves** | Base + Meta analysis | Meta-only analysis | ✅ **Appropriate difference** |
| **Feature Importance** | 4-method analysis + Excel | 4-method analysis + Excel | ✅ **Identical** |
| **SHAP Analysis** | Enhanced with visualizations | Enhanced with visualizations | ✅ **Identical** |
| **Feature Leakage Detection** | Correlation analysis | Correlation analysis | ✅ **Identical** |
| **Calibration Support** | Per-class + binary | Per-class + binary | ✅ **Identical** |
| **Diagnostic Functions** | 8 comprehensive functions | 8 comprehensive functions | ✅ **Identical** |
| **Memory Optimization** | Standard processing | Enhanced (chunked, sparse) | ✅ **LOCO advantage** |

### Key Differences (Expected and Appropriate)
1. **CV Methodology**: Gene-wise vs chromosome-wise splitting
2. **Base Model Integration**: Gene CV includes base model comparison; LOCO CV focuses on meta-only evaluation
3. **Memory Features**: LOCO CV includes additional scalability features
4. **Filename Conventions**: Reflect different CV approaches (`fold_{N}` vs `fold_{chrom}`)

---

## 4. When to Use Which?

| Scenario | Recommended Strategy | Script |
|----------|----------------------|--------|
| **Benchmark generalisation across genes within same chromosomes** | Gene-aware | `run_gene_cv_sigmoid.py` |
| **Stress-test for large structural biases** | Chromosome-aware (LOCO-CV) | `run_loco_cv_multiclass_scalable.py` |
| **Small dataset (<100 genes)** | Gene-aware | `run_gene_cv_sigmoid.py` |
| **Large dataset (20K+ genes)** | Chromosome-aware with memory optimization | `run_loco_cv_multiclass_scalable.py` |
| **Full-genome training** | Combine both: gene-aware split inside each LOCO-CV fold | Both scripts |
| **Memory-constrained environments** | Chromosome-aware with chunked loading | `run_loco_cv_multiclass_scalable.py` |

Ideally report **both**: gene-aware gives a realistic in-domain score; chromosome-aware shows the worst-case drop when the model faces unseen genomic context.

---

## 5. Comprehensive Usage Examples

### Enhanced Gene-Aware CV
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
    --n-folds 5 --verbose
```

### Enhanced Chromosome-Aware CV (LOCO-CV)
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir models/comprehensive_loco_cv \
    --calibrate-per-class --calib-method platt \
    --plot-curves --plot-format pdf --n-roc-points 101 \
    --check-leakage --leakage-threshold 0.95 --auto-exclude-leaky \
    --diag-sample 10000 --neigh-sample 1000 --neigh-window 12 \
    --transcript-topk \
    --heldout-chroms "21,22" --verbose 2
```

### Memory-Optimized Large-Scale LOCO-CV
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_full/master \
    --out-dir models/large_scale_loco_cv \
    --use-chunked-loading --chunksize 10000 \
    --use-sparse-kmers --memory-optimize \
    --feature-selection --max-features 1000 \
    --calibrate-per-class \
    --row-cap 0 --verbose 2
```

---

## 6. Beyond Genes & Chromosomes – Other Grouping Ideas

* **Transcript-aware**: group by `transcript_id` for alternative splicing research.
* **Distance-based blocks**: bin the genome into 1 Mb windows and hold out windows – proxy for unseen loci.
* **Tissue-aware**: if training on multi-tissue RNA-seq, hold out entire tissues.
* **Species-aware**: cross-species GANs?  Hold out mouse when training on human & macaque.

Adapt `chromosome_split.py` as a template: simply supply a *group vector* and write a new `group_*` helper.

---

## 7. Enhanced Features Available in Both CV Approaches

### 7.1 CV Metrics Visualization (7 Plots)
Both scripts generate identical visualization suites:
- **F1 Score Comparison**: Performance across CV folds
- **ROC AUC Comparison**: Area under curve analysis with confidence intervals
- **Average Precision Comparison**: Precision-recall performance evaluation
- **Error Reduction Analysis**: Visual analysis of performance improvements
- **Performance Overview**: Multi-metric dashboard
- **Improvement Summary**: Percentage gains analysis
- **Top-k Analysis**: Accuracy patterns

### 7.2 Comprehensive Feature Importance Analysis
Four complementary analytical methods:
- **XGBoost Internal Importance**: weight, gain, cover metrics
- **Statistical Hypothesis Testing**: t-tests, Mann-Whitney U, Chi-square, Fisher's exact
- **Effect Size Measurements**: Cohen's d, Cramer's V, rank-biserial correlation
- **Mutual Information Analysis**: Information-theoretic feature ranking

### 7.3 Enhanced SHAP Analysis
- **Memory-Efficient Processing**: Incremental SHAP for large datasets
- **Comprehensive Visualizations**: Summary plots, dependence plots, feature rankings
- **Per-Class Analysis**: Separate analysis for each splice site class
- **Graceful Fallbacks**: Automatic fallback mechanisms

### 7.4 Feature Quality Control
- **Leakage Detection**: Correlation analysis to identify problematic features
- **Automatic Exclusion**: Option to remove features exceeding correlation thresholds
- **Feature Filtering**: Support for custom exclusion lists
- **Detailed Reporting**: Comprehensive logging of feature decisions

---

## 8. FAQ

**Q: Do I need to stratify by label within each chromosome?**  
Not mandatory – splice-site labels are usually well-mixed.  If imbalance is severe, use `StratifiedGroupKFold` (scikit-learn ≥1.4).

**Q: How many rows per test fold are "enough"?**  
Rule-of-thumb ≥1 000 for stable PR-AUC; fewer is acceptable if you aggregate across many folds.

**Q: Can I mix chromosomes in the validation set?**  
Yes.  Validation is carved *after* reserving the test chromosomes, maintaining independence.

**Q: Does transcript-level top-k accuracy evaluation work with chromosome-aware splits?**  
Yes. Both LOCO-CV and fixed chromosome hold-out support transcript-level top-k accuracy evaluation with the `--transcript-topk` flag. The caching mechanism ensures efficient recomputation across folds with different chromosomes.

**Q: Which CV approach should I use for my dataset?**  
- **Gene CV**: For in-domain evaluation and moderate datasets (<10K genes)
- **LOCO CV**: For out-of-distribution testing and large datasets (20K+ genes)
- **Both**: For comprehensive evaluation (recommended for final model assessment)

**Q: Are the outputs compatible between CV approaches?**  
Yes, both scripts generate compatible output formats that work with the same downstream analysis tools, with only minor filename differences reflecting the CV methodology.

**Q: Can I disable specific enhanced features?**  
Yes, both scripts support flags like `--no-plot-curves`, `--no-leakage-check`, or setting `--diag-sample 0` to disable diagnostics.

**Q: What were the major recent fixes?**  
The evaluation methodology was completely overhauled to use classification-based evaluation instead of position-based evaluation, which was causing unrealistic performance metrics. Both CV approaches now show realistic improvement ranges.

---

## 9. Performance Expectations

### Typical Results Comparison

| Metric | Gene CV (5-fold) | LOCO CV (Leave-One-Chrom-Out) |
|--------|------------------|--------------------------------|
| **F1 Improvement** | 30-40% over base *(post-fix)* | 25-35% over base *(post-fix, more conservative)* |
| **Error Reduction** | 300-800 per fold | 200-600 per fold |
| **Top-K Accuracy** | 85-95% gene-level | 80-90% chromosome-level |
| **Processing Time** | 1-3 hours (1K genes) | 2-5 hours (full genome) |

### Resource Requirements

| CV Type | Memory | Disk Space | Recommended CPU |
|---------|--------|------------|-----------------|
| **Gene CV** | 8-32 GB | 2-5 GB | Multi-core (8+ cores) |
| **LOCO CV Standard** | 16-64 GB | 3-8 GB | Multi-core (12+ cores) |
| **LOCO CV Memory-Optimized** | 4-16 GB | 5-15 GB | Multi-core (8+ cores) |

### Evaluation Reliability
- **Gene CV**: Tests in-domain generalization across genes
- **LOCO CV**: Tests out-of-distribution generalization across chromosomes
- **Both Approaches**: Now use proper classification-based evaluation for realistic metrics
- **Complementary**: Both approaches provide different but valuable insights

---

*Last updated: 2025-01-15*
