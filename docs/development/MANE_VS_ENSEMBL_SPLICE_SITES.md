# MANE vs Ensembl Splice Sites Comparison

**Date**: 2025-11-06  
**Issue**: MANE has 5.4x fewer splice sites than Ensembl  
**Status**: ⚠️ **CRITICAL** - Need to use full Ensembl GRCh38 instead of MANE

---

## The Problem

We downloaded GRCh38 **MANE** annotations for OpenSpliceAI, but MANE has significantly fewer splice sites than Ensembl:

| Dataset | Splice Sites | Genes | Transcripts | Trans/Gene |
|---------|-------------|-------|-------------|------------|
| **GRCh37 Ensembl** | 1,998,527 | 35,306 | 171,558 | ~4.9 |
| **GRCh38 MANE** | 369,919 | 18,200 | 18,264 | ~1.0 |
| **Difference** | **5.4x fewer** | 1.9x fewer | 9.4x fewer | - |

---

## Why MANE Has Fewer Splice Sites

### MANE (Matched Annotation from NCBI and EMBL-EBI)

**Purpose**: Clinical reporting and variant interpretation

**Design Philosophy**:
- ✅ **One transcript per gene**: Only the most clinically relevant isoform
- ✅ **High quality**: Manually curated, high-confidence transcripts
- ✅ **Clinical focus**: Prioritizes transcripts used in medical genetics
- ✅ **Reduced redundancy**: Avoids multiple similar isoforms

**Coverage**:
- ~18,000 genes (mostly protein-coding)
- ~18,000 transcripts (1:1 ratio)
- ~370,000 splice sites

### Ensembl

**Purpose**: Comprehensive genomic research

**Design Philosophy**:
- ✅ **All isoforms**: Includes all known transcript variants
- ✅ **All biotypes**: Protein-coding, lncRNA, pseudogenes, etc.
- ✅ **Research focus**: Maximizes coverage for discovery
- ✅ **Comprehensive**: Includes experimental and predicted transcripts

**Coverage**:
- ~35,000 genes (all biotypes)
- ~170,000 transcripts (~5 per gene)
- ~2,000,000 splice sites

---

## Impact on OpenSpliceAI

### What OpenSpliceAI Was Likely Trained On

Looking at the OpenSpliceAI code, it supports:
- `--parse-type`: 'canonical' or 'all_isoforms'
- `--biotype`: 'protein-coding', 'non-coding', or 'all'
- `--canonical-only`: Boolean flag

**Most likely training configuration**:
- Full Ensembl GRCh38 annotations
- Multiple isoforms per gene
- Protein-coding + non-coding transcripts

**Evidence**:
1. The pre-trained models are named "openspliceai-mane" but this might just indicate they're compatible with MANE, not trained exclusively on it
2. Training on only MANE would severely limit the model's ability to predict splicing in:
   - Alternative isoforms
   - Non-coding RNAs
   - Tissue-specific transcripts
   - Novel splice variants

### Performance Implications

Using MANE-only annotations for evaluation would:

❌ **Underestimate performance**:
- Miss ~80% of known splice sites
- Can't evaluate on alternative isoforms
- Limited to canonical transcripts only

❌ **Incomparable to SpliceAI**:
- SpliceAI evaluated on full Ensembl (GRCh37)
- Can't do fair comparison with different annotation sets

❌ **Limited applicability**:
- Can't predict splicing in non-MANE transcripts
- Misses tissue-specific isoforms
- Incomplete for research applications

---

## Solution: Use Full Ensembl GRCh38

### Recommended Approach

**Download Ensembl GRCh38** (not MANE) for OpenSpliceAI:

```bash
# Download full Ensembl GRCh38 annotations
source ~/.bash_profile && mamba activate surveyor
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --build GRCh38 \
  --release 112 \
  --verbose

# Derive splice sites
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh38 \
  --release 112 \
  --splice-sites \
  --consensus-window 2 \
  --verbose
```

**Expected Output**:
```
data/ensembl/GRCh38/
├── Homo_sapiens.GRCh38.112.gtf           # ~1.5 GB
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa  # ~3.0 GB
└── splice_sites_enhanced.tsv              # ~140 MB, ~2M sites
```

### Why This is Better

✅ **Comparable coverage**:
- ~2M splice sites (similar to GRCh37)
- ~35K genes
- ~170K transcripts

✅ **Fair comparison**:
- Same annotation philosophy as SpliceAI's training data
- Can compare GRCh37 vs GRCh38 performance
- Apples-to-apples evaluation

✅ **Full functionality**:
- Predict on all transcript isoforms
- Support alternative splicing analysis
- Research-grade coverage

---

## When to Use MANE vs Ensembl

### Use MANE When:

✅ **Clinical applications**:
- Variant interpretation for clinical reports
- Pathogenicity prediction for known genes
- Focused analysis on canonical transcripts

✅ **Computational efficiency**:
- Need faster processing (5x fewer sites)
- Limited computational resources
- Quick validation studies

✅ **High-confidence only**:
- Don't want experimental/predicted transcripts
- Need manually curated annotations
- Clinical-grade quality requirements

### Use Ensembl When:

✅ **Research applications**:
- Comprehensive splicing analysis
- Alternative isoform discovery
- Tissue-specific expression studies

✅ **Model training/evaluation**:
- Training splice prediction models
- Benchmarking model performance
- Comparing across studies

✅ **Maximum coverage**:
- Novel transcript discovery
- Non-coding RNA analysis
- Comprehensive variant analysis

---

## Action Plan

### Immediate (Required)

1. **Download Ensembl GRCh38** (~30 minutes)
   ```bash
   ./scripts/setup/download_grch38_ensembl_data.sh
   ```

2. **Update OpenSpliceAI configuration** to use Ensembl GRCh38
   - Modify `base_model` routing to use `GRCh38` (Ensembl) not `GRCh38_MANE`

3. **Re-test OpenSpliceAI** with full annotations
   ```python
   from meta_spliceai import run_base_model_predictions
   
   results = run_base_model_predictions(
       base_model='openspliceai',
       target_genes=['BRCA1', 'TP53'],
       mode='test'
   )
   ```

### Future (Optional)

1. **Support both MANE and Ensembl**
   - Add `annotation_source` parameter to `BaseModelConfig`
   - Allow users to choose based on use case

2. **Create MANE-specific workflows**
   - Clinical variant analysis pipeline
   - Focused canonical transcript evaluation

3. **Benchmark comparison**
   - Compare OpenSpliceAI performance on MANE vs Ensembl
   - Document trade-offs

---

## Comparison Table

| Aspect | MANE | Ensembl GRCh38 | Recommendation |
|--------|------|----------------|----------------|
| **Splice Sites** | 370K | ~2M | ✅ Ensembl |
| **Genes** | 18K | 35K | ✅ Ensembl |
| **Transcripts** | 18K | 170K | ✅ Ensembl |
| **Quality** | ✅ Curated | Mixed | MANE for clinical |
| **Coverage** | Canonical only | All isoforms | ✅ Ensembl for research |
| **Clinical Use** | ✅ Optimized | General | MANE |
| **Research Use** | Limited | ✅ Comprehensive | ✅ Ensembl |
| **Model Training** | Too limited | ✅ Appropriate | ✅ Ensembl |
| **File Size** | 25 MB | 140 MB | Ensembl (acceptable) |
| **Processing Time** | Fast | Slower | Ensembl (worth it) |

---

## Conclusion

**MANE is NOT appropriate for OpenSpliceAI evaluation** because:

1. ❌ 5.4x fewer splice sites than Ensembl
2. ❌ Only canonical transcripts (no isoforms)
3. ❌ Can't compare fairly with SpliceAI (which uses full Ensembl)
4. ❌ OpenSpliceAI was likely trained on full Ensembl, not MANE

**Solution**: Download and use **Ensembl GRCh38** (release 112) for:
- OpenSpliceAI predictions
- Model evaluation
- Fair comparison with SpliceAI

**Keep MANE for**: Clinical-specific workflows and focused canonical transcript analysis

---

**Status**: ⚠️ **ACTION REQUIRED** - Download Ensembl GRCh38  
**Priority**: **HIGH** - Needed for proper OpenSpliceAI evaluation  
**Estimated Time**: 30-40 minutes

---

*Last Updated: 2025-11-06*

