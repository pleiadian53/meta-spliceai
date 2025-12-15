# train_regulatory_10k_kmers Dataset

**Large-scale regulatory-focused training dataset for meta-model development**

## Quick Overview

- **Dataset Name**: `train_regulatory_10k_kmers`
- **Location**: `train_regulatory_10k_kmers/master/` (project root)
- **Purpose**: Training dataset with regulatory gene focus and multi-scale k-mer features
- **Gene Count**: 9,280 genes (protein_coding + lncRNA)
- **Training Records**: 3,729,279 position-centric instances
- **K-mer Features**: 3-mers and 5-mers (1,088 total k-mer features)

## Key Statistics

| Metric | Value |
|--------|-------|
| **Total Genes** | 9,280 |
| **Gene Types** | 2 (protein_coding: 62.3%, lncRNA: 37.7%) |
| **Training Records** | 3,729,279 |
| **Total Splice Sites** | 1,400,304 |
| **Dataset Size** | 1.91 GB |
| **Batch Files** | 50 |
| **Features per Instance** | 1,167 |

## Gene Characteristics

### Gene Type Distribution
- **protein_coding**: 5,778 genes (62.3%) - High splice density, reliable annotations
- **lncRNA**: 3,502 genes (37.7%) - Regulatory RNA genes, moderate splice density

### Gene Length Distribution
- **Mean**: 58,198 bp
- **Median**: 23,088 bp  
- **Range**: 269 bp - 1,504,183 bp
- **Long Gene Emphasis**: Includes very long genes for comprehensive regulatory context

### Splice Site Characteristics
- **Total Splice Sites**: 1,400,304
- **Mean per Gene**: 150.9 sites
- **Median per Gene**: 40.0 sites
- **Splice Density**: 5.15 sites/kb (mean), 1.47 sites/kb (median)

## Feature Set

### K-mer Features (1,088 features)
- **3-mers**: 64 features (3^4 possible combinations)
- **5-mers**: 1,024 features (4^5 possible combinations)
- **Representation**: Raw counts from surrounding sequence context
- **Regulatory Focus**: Multi-scale motif detection for regulatory elements

### Enriched Features (74 features)
- **Performance Features**: SpliceAI prediction scores and confidence metrics
- **Gene Characteristics**: Gene length, type, splice density
- **Positional Features**: Distance to nearest splice sites, context ratios
- **Signal Strength**: Peak detection and signal analysis features

## Use Cases

### Primary Applications
1. **Regulatory Variant Analysis**: Enhanced detection of noncoding regulatory variants affecting splicing
2. **Multi-Gene-Type Training**: Balanced representation of protein-coding and lncRNA genes
3. **Multi-Scale Pattern Recognition**: 3-mer and 5-mer features capture different motif scales
4. **Large-Scale Meta-Learning**: 10K gene scale for robust model generalization

### Training Workflows
- **Gene-Aware Cross-Validation**: Use with `run_gene_cv_sigmoid.py`
- **Chromosome-Aware CV**: Use with `run_loco_cv_multiclass_scalable.py`
- **Ablation Studies**: Feature importance analysis with multi-scale k-mers

## Documentation

- **[Dataset Profile](train_regulatory_10k_kmers_profile.md)**: Comprehensive analysis and characteristics
- **[Technical Specifications](train_regulatory_10k_kmers_technical_spec.md)**: Schema, validation rules, performance metrics
- **[Validation Script](validate_train_regulatory_10k_kmers.py)**: Automated quality checks

## Quick Validation

```bash
# Run validation script
cd meta_spliceai/splice_engine/case_studies/data_sources/datasets/train_regulatory_10k_kmers/
python validate_train_regulatory_10k_kmers.py

# Quick manifest check
head -5 ../../../../../../train_regulatory_10k_kmers/master/gene_manifest.csv
```

## Training Command Example

```bash
# Gene-aware cross-validation with full dataset
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_regulatory_10k_kmers/master \
    --out-dir results/regulatory_10k_cv_run1 \
    --n-estimators 800 \
    --row-cap 0 \
    --calibrate-per-class \
    --monitor-overfitting \
    --verbose
```

---

**Created**: August 2025  
**Enhanced Manifest**: ✅ Yes  
**Validation**: ✅ Available  
**Status**: Production-ready
