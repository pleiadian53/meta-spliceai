# Feature Mismatch - Quick Reference Guide

## Three Cases (User's Complete Analysis)

| # | Scenario | Action | Reason |
|---|----------|--------|--------|
| **1** | **More features** | ✅ Drop extra | New k-mers not in training |
| **2a** | **Fewer: missing k-mers** | ✅ Fill with 0 | K-mer not in test sequence (normal) |
| **2b** | **Fewer: missing genomic** | ❌ Raise error | Incomplete feature extraction (bug) |

---

## What You'll See

### ✅ Case 1: Extra Features (OK)

```
ℹ️  Extra 4 k-mers (not in training, will drop)
   K-mers: ['AAN', 'CNN', 'GNN', 'NNN']
✅ Features aligned: 121 columns
```

**Action:** None needed - automatically handled

---

### ✅ Case 2a: Missing K-mers (OK)

```
ℹ️  Missing 19 k-mers (not in test sequence, will fill with 0)
   K-mers: ['AAG', 'AAT', 'ACA', 'ACG', ...]
✅ Features aligned: 121 columns
```

**Action:** None needed - automatically handled

**Why:** Test gene doesn't contain all possible k-mers → count = 0 is correct

---

### ❌ Case 2b: Missing Genomic Features (ERROR)

```
❌ Missing 3 CRITICAL features
   Features: ['tx_start', 'tx_end', 'num_overlaps']
❌ CRITICAL: Inference is missing 3 non-k-mer features.
   This indicates incomplete feature generation.
```

**Action:** Debug feature extraction pipeline

**Common Causes:**
1. `GenomicFeatureEnricher` not called
2. Genomic enrichment called too late (after feature extraction)
3. Missing data files (`transcript_features.tsv`, `overlapping_gene_counts.tsv`)
4. Sequence feature calculation not running

**Fix:** Run diagnostic tool:
```bash
python scripts/testing/diagnose_feature_mismatch.py
```

---

## Feature Types

### K-mers (Can Be Missing)
- 3-mers: `AAA`, `AAC`, `AAG`, ..., `TTT` (64 total)
- 4-mers: `AAAA`, `AAAC`, ..., `TTTT` (256 total)

**Missing k-mers = normal** (test gene doesn't contain all motifs)

### Genomic Features (Must Be Present)

**Base Scores:**
- `donor_score`, `acceptor_score`, `neither_score`

**Probability Features:**
- `relative_donor_probability`, `splice_probability`
- `donor_acceptor_diff`, `splice_neither_diff`
- `probability_entropy`

**Context Features:**
- `context_neighbor_mean`, `context_asymmetry`, `context_max`
- `donor_diff_m1`, `donor_surge_ratio`, `donor_is_local_peak`
- (Similar for acceptor)

**Genomic Features:**
- `gene_start`, `gene_end`, `tx_start`, `tx_end`
- `num_overlaps`, `transcript_length`
- `chrom` (encoded)

**Sequence Features:**
- `gc_content`, `sequence_length`, `sequence_complexity`

**Missing any of these = bug** (should always be computable)

---

## Quick Diagnosis

### Step 1: Identify Feature Type

```python
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import is_kmer_feature

# Check if feature is a k-mer
is_kmer_feature('AAA')  # True
is_kmer_feature('tx_start')  # False
```

### Step 2: Determine Severity

```
Missing feature: 'AAG'
  → is_kmer_feature('AAG') = True
  → ✅ Non-critical (fill with 0)

Missing feature: 'tx_start'
  → is_kmer_feature('tx_start') = False
  → ❌ CRITICAL (bug in feature extraction)
```

### Step 3: Take Action

**If k-mer missing:**
- ✅ Automatically filled with 0
- ✅ No action needed

**If genomic feature missing:**
- ❌ Run diagnostic: `python scripts/testing/diagnose_feature_mismatch.py`
- ❌ Check feature extraction pipeline
- ❌ Verify data files exist

---

## Troubleshooting

### All K-mers Missing (64 missing)

**Symptom:**
```
❌ Missing 64 CRITICAL features
   Features: ['AAA', 'AAC', 'AAG', ..., 'TTT']
```

**Diagnosis:** K-mer generation not running

**Fix:**
```python
# Verify this line exists in _generate_complete_base_model_predictions()
complete_predictions = self._generate_kmer_features(complete_predictions, kmer_sizes=[3])
```

---

### Genomic Features Missing

**Symptom:**
```
❌ Missing 5 CRITICAL features
   Features: ['tx_start', 'tx_end', 'num_overlaps', 'transcript_length', 'gene_length']
```

**Diagnosis:** Genomic enrichment not running or incomplete

**Fix:**
```python
# Verify this line exists and runs BEFORE k-mer generation
complete_predictions = self.genomic_enricher.enrich(complete_predictions)
```

**Check data files exist:**
```bash
ls data/ensembl/spliceai_analysis/transcript_features.tsv
ls data/ensembl/overlapping_gene_counts.tsv
ls data/ensembl/spliceai_analysis/gene_features.tsv
```

---

### Sequence Features Missing

**Symptom:**
```
❌ Missing 3 CRITICAL features
   Features: ['gc_content', 'sequence_length', 'sequence_complexity']
```

**Diagnosis:** Sequence feature calculation not running

**Fix:**
```python
# Verify _calculate_sequence_features() is called in _generate_kmer_features()
df = self._calculate_sequence_features(df)
```

---

## Summary

### Normal Behavior ✅

```
Feature alignment:
  Model expects: 121 features
  Inference has: 102-125 features
  ℹ️  Missing k-mers: filled with 0
  ℹ️  Extra k-mers: dropped
✅ Features aligned: 121 columns
```

### Error Behavior ❌

```
Feature alignment:
  Model expects: 121 features
  Inference has: <121 features
  ❌ Missing non-k-mer features
❌ CRITICAL: Incomplete feature generation
```

---

## Key Takeaway

**K-mers can be missing (normal) → Fill with 0**

**Genomic features cannot be missing (bug) → Raise error**

This distinction is critical for correct inference behavior! 🎯

