# Compare Base Models Robust - Usage Guide

**Date**: November 9, 2025  
**Status**: Production Ready

---

## Overview

`compare_base_models_robust.py` is the **unified script** for comparing SpliceAI and OpenSpliceAI base models. It supports both:
1. **Standard comparison** (positions only: TP/TN/FP/FN)
2. **Full coverage testing** (nucleotide-level scores for all positions)

---

## Features

### Core Capabilities

1. **Intersection-Based Gene Sampling**
   - Finds genes present in BOTH GRCh37/Ensembl and GRCh38/MANE
   - Ensures fair comparison between models
   - Samples by category (protein-coding, lncRNA, no splice sites)

2. **Graceful Error Handling**
   - Handles missing genes without failing
   - Reports which genes were processed
   - Continues even if some genes fail

3. **Dual Output Modes**
   - **Standard**: Positions only (TP/FP/TN/FN) - fast, small data
   - **Full Coverage**: Nucleotide-level scores - comprehensive, large data

4. **Performance Metrics**
   - Precision, Recall, F1 Score, Accuracy
   - Per-gene statistics
   - Runtime comparison

---

## Configuration

### Basic Settings

```python
# At the top of the script (lines 35-42)
TEST_NAME = f"base_model_comparison_robust_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
N_PROTEIN_CODING = 10  # Sample size per category
N_LNCRNA = 5
N_NO_SPLICE_SITES = 5
SEED = 42

# NEW: Option to save nucleotide-level scores
SAVE_NUCLEOTIDE_SCORES = False  # Set to True for full coverage testing
```

### Mode Selection

**Standard Mode** (Default):
```python
SAVE_NUCLEOTIDE_SCORES = False
```
- Fast execution (~5-10 minutes)
- Small data volume (~10 MB)
- Positions only (TP/FP/TN/FN)
- **Use for**: Performance comparison, model validation

**Full Coverage Mode**:
```python
SAVE_NUCLEOTIDE_SCORES = True
```
- Slower execution (~15-30 minutes)
- Large data volume (~100-500 MB)
- Nucleotide-level scores for all positions
- **Use for**: Visualization, detailed analysis, nucleotide-level comparison

---

## Usage

### Standard Comparison (Positions Only)

```bash
# 1. Edit script to set mode
# SAVE_NUCLEOTIDE_SCORES = False  (default)

# 2. Run script
cd /Users/pleiadian53/work/meta-spliceai
source ~/.bash_profile && mamba activate surveyor
python scripts/testing/compare_base_models_robust.py
```

**Output**:
```
results/base_model_comparison_robust_YYYYMMDD_HHMMSS/
├── sampled_genes.json           # Gene sampling details
├── test_genes.txt               # List of test genes
├── comparison_results.json      # Performance metrics
└── (positions artifacts in test directories)
```

**Expected Runtime**: 5-10 minutes for 20 genes

---

### Full Coverage Testing (Nucleotide Scores)

```bash
# 1. Edit script to enable full coverage
# Change line 42 to:
# SAVE_NUCLEOTIDE_SCORES = True

# 2. Run script
cd /Users/pleiadian53/work/meta-spliceai
source ~/.bash_profile && mamba activate surveyor
python scripts/testing/compare_base_models_robust.py
```

**Output**:
```
results/base_model_comparison_robust_YYYYMMDD_HHMMSS/
├── sampled_genes.json           # Gene sampling details
├── test_genes.txt               # List of test genes
├── comparison_results.json      # Performance metrics
└── (nucleotide_scores.tsv in test directories)  ← NEW
```

**Expected Runtime**: 15-30 minutes for 20 genes

**Data Volume**: ~100-500 MB (depending on gene sizes)

---

## Output Details

### Standard Mode Output

**Positions DataFrame** (TP/FP/TN/FN):
```
Columns: gene_id, gene_name, position, splice_type, pred_type, 
         donor_score, acceptor_score, neither_score, ...
Rows: ~1,000-10,000 per gene (only significant positions)
```

**Performance Metrics**:
```json
{
  "spliceai": {
    "positions": 50000,
    "tp": 1500,
    "fp": 200,
    "fn": 100,
    "precision": 0.882,
    "recall": 0.938,
    "f1": 0.909,
    "accuracy": 0.994
  },
  "openspliceai": { ... }
}
```

---

### Full Coverage Mode Output

**Nucleotide Scores DataFrame** (ALL positions):
```
Columns: gene_id, gene_name, chrom, strand, position, genomic_position,
         donor_score, acceptor_score, neither_score
Rows: ~100,000 per gene (every nucleotide)
```

**Example**:
```
gene_id          gene_name  position  donor_score  acceptor_score  neither_score
ENSG00000012048  BRCA1      1         0.001        0.002          0.997
ENSG00000012048  BRCA1      2         0.001        0.001          0.998
ENSG00000012048  BRCA1      3         0.002        0.001          0.997
...
ENSG00000012048  BRCA1      81188     0.001        0.001          0.998
```

**Use Cases**:
- Visualize complete splice landscape
- Compare models at nucleotide resolution
- Identify subtle differences in predictions
- Generate plots with `visualize_nucleotide_scores.py`

---

## Gene Sampling Strategy

### Step 1: Find Intersection

```python
# Load gene features from both builds
grch37_genes = load_gene_features('data/ensembl/GRCh37/')
grch38_genes = load_gene_features('data/mane/GRCh38/')

# Find intersection by gene name
intersection = grch37_gene_names & grch38_gene_names
# Result: ~15,000-20,000 genes
```

### Step 2: Sample by Category

**Protein-Coding Genes** (N=10):
```python
protein_coding = filter(
    gene_type == 'protein_coding' &
    n_splice_sites >= 4 &
    gene_length between 5kb-500kb
)
sample(n=10, seed=42)
```

**lncRNA Genes** (N=5):
```python
lncrna = filter(
    gene_type in ['lncRNA', 'lincRNA', ...] &
    n_splice_sites >= 2 &
    gene_length between 1kb-200kb
)
sample(n=5, seed=42)
```

**Genes Without Splice Sites** (N=5):
```python
no_splice_sites = filter(
    n_splice_sites == 0 &
    gene_length between 500bp-50kb
)
sample(n=5, seed=42)
```

**Total**: 20 genes from intersection

---

## Performance Comparison

### Standard Mode

| Metric | Value |
|--------|-------|
| Genes | 20 |
| Positions | ~50,000 |
| Runtime | 5-10 min |
| Data Volume | ~10 MB |
| Memory | ~500 MB |

### Full Coverage Mode

| Metric | Value |
|--------|-------|
| Genes | 20 |
| Positions | ~50,000 (TP/FP/TN/FN) |
| Nucleotide Scores | ~2,000,000 (all nucleotides) |
| Runtime | 15-30 min |
| Data Volume | ~200 MB |
| Memory | ~2 GB |

---

## Example Output

### Standard Mode Console Output

```
================================================================================
ROBUST BASE MODEL COMPARISON
================================================================================

Test Name: base_model_comparison_robust_20251109_180000
Strategy: Sample from gene intersection between builds
Categories: 10 protein-coding, 5 lncRNA, 5 no splice sites
Nucleotide Scores: DISABLED (Positions Only)

================================================================================
STEP 1: Find Gene Intersection
================================================================================

✅ Found 18,234 genes in both builds

================================================================================
STEP 2: Sample Genes by Category
================================================================================

PROTEIN-CODING GENES:
  Available: 12,456
  ✅ Sampled: 10
     Examples: BRCA1, TP53, EGFR, MYC, KRAS

...

================================================================================
STEP 4: Run SpliceAI (GRCh37/Ensembl)
================================================================================

✅ SpliceAI completed in 245.3 seconds
   Positions analyzed: 48,234
   Genes processed: 20/20

================================================================================
STEP 5: Run OpenSpliceAI (GRCh38/MANE)
================================================================================

✅ OpenSpliceAI completed in 189.7 seconds
   Positions analyzed: 49,123
   Genes processed: 20/20

================================================================================
STEP 6: Performance Comparison
================================================================================

OVERALL COMPARISON
--------------------------------------------------------------------------------
Metric               SpliceAI                  OpenSpliceAI             
--------------------------------------------------------------------------------
Build                GRCh37/Ensembl            GRCh38/MANE              
Genes Processed      20                        20                       
Positions            48,234                    49,123                   
TP                   1,456                     1,523                    
FP                   234                       198                      
FN                   87                        76                       
Precision            0.8615                    0.8850                   
Recall               0.9436                    0.9525                   
F1 Score             0.9007                    0.9175                   
Accuracy             0.9933                    0.9944                   
--------------------------------------------------------------------------------
Runtime (sec)        245.3                     189.7                    
```

---

### Full Coverage Mode Console Output

```
================================================================================
ROBUST BASE MODEL COMPARISON
================================================================================

Test Name: base_model_comparison_robust_20251109_180000
Strategy: Sample from gene intersection between builds
Categories: 10 protein-coding, 5 lncRNA, 5 no splice sites
Nucleotide Scores: ENABLED (Full Coverage)  ← NOTICE THIS

...

================================================================================
STEP 4: Run SpliceAI (GRCh37/Ensembl)
================================================================================

✅ SpliceAI completed in 456.8 seconds
   Positions analyzed: 48,234
   Nucleotide scores: 1,987,456  ← NEW
   Genes processed: 20/20

================================================================================
STEP 5: Run OpenSpliceAI (GRCh38/MANE)
================================================================================

✅ OpenSpliceAI completed in 398.2 seconds
   Positions analyzed: 49,123
   Nucleotide scores: 1,989,234  ← NEW
   Genes processed: 20/20
```

---

## Troubleshooting

### Issue: No genes sampled

**Symptom**:
```
❌ No genes sampled - cannot proceed
```

**Solution**:
- Check that gene_features.tsv exists in both builds
- Verify Registry paths are correct
- Ensure intersection is not empty

---

### Issue: Some genes missing

**Symptom**:
```
⚠️  Missing genes: 3
   GENE1, GENE2, GENE3...
```

**Solution**:
- This is NORMAL and handled gracefully
- Script continues with genes that were found
- Missing genes are reported in comparison_results.json

---

### Issue: Out of memory (Full Coverage Mode)

**Symptom**:
```
MemoryError: Unable to allocate array
```

**Solution**:
- Reduce N_PROTEIN_CODING, N_LNCRNA, N_NO_SPLICE_SITES
- Process genes in smaller batches
- Use standard mode instead

---

## Related Scripts

**Visualization**:
```bash
# After running with SAVE_NUCLEOTIDE_SCORES=True
python scripts/testing/visualize_nucleotide_scores.py \
    --input results/.../nucleotide_scores.tsv \
    --gene BRCA1
```

**Simple Comparison** (for quick testing):
```bash
python scripts/testing/compare_base_models_simple.py
```

---

## Summary

### Key Points

1. **One Script, Two Modes**:
   - Standard: Fast, positions only
   - Full Coverage: Comprehensive, nucleotide-level

2. **Proven Gene Sampling**:
   - Intersection-based
   - Fair comparison
   - Graceful error handling

3. **Easy Configuration**:
   - One parameter: `SAVE_NUCLEOTIDE_SCORES`
   - Set to `True` for full coverage
   - Set to `False` for standard mode

4. **Production Ready**:
   - Robust error handling
   - Comprehensive reporting
   - Well-tested approach

---

*Last Updated: November 9, 2025*  
*Status: Production Ready*


