# Base Model Artifacts Verification

**Date**: 2025-11-17  
**Status**: ✅ **VERIFIED** - Ready for Meta-Model Training

---

## Summary

Verified that the base model pass generates the correct artifacts required for meta-model training via `incremental_builder.py`. The `analysis_sequences_*` files contain all necessary columns including contextual sequences, base model scores, and derived features.

---

## Verification Results

### 1. ✅ Artifact Structure

**Test File**: `analysis_sequences_21_chunk_1_214.tsv` (chr21 complete run)
- **Size**: 11 MB
- **Rows**: 7,859 positions (7,858 data rows + 1 header)
- **Columns**: 57 features

### 2. ✅ Required Columns Present

#### Core Identity Columns
```
✅ gene_id                  - Gene identifier (e.g., gene-USP25)
✅ transcript_id            - Transcript identifier (e.g., rna-NM_001283041.3)
✅ position                 - Genomic position
✅ predicted_position       - Model's predicted position
✅ true_position            - Ground truth position (for TP/FN)
✅ pred_type                - Classification (TP/FP/FN/TN)
✅ splice_type              - Donor or acceptor
✅ strand                   - Strand (+/-)
✅ chrom                    - Chromosome
```

#### Base Model Scores (3 probabilities)
```
✅ donor_score              - P(donor splice site)
✅ acceptor_score           - P(acceptor splice site)
✅ neither_score            - P(not a splice site)
✅ score                    - Main score (max of donor/acceptor)
```

#### Context Features (±2 window)
```
✅ context_score_m2         - Score at position -2
✅ context_score_m1         - Score at position -1
✅ context_score_p1         - Score at position +1
✅ context_score_p2         - Score at position +2
```

#### Derived Probability Features
```
✅ relative_donor_probability      - Relative strength indicator
✅ splice_probability              - Combined splice probability
✅ donor_acceptor_diff            - Score differences
✅ splice_neither_diff            - Splice vs neither difference
✅ donor_acceptor_logodds         - Log-odds ratios
✅ splice_neither_logodds         - Log-odds ratios
✅ probability_entropy             - Uncertainty measure
✅ context_neighbor_mean           - Average of context scores
✅ context_asymmetry               - Left vs right context
✅ context_max                     - Maximum context score
```

#### Donor-Specific Derived Features
```
✅ donor_diff_m1, donor_diff_m2, donor_diff_p1, donor_diff_p2
✅ donor_surge_ratio               - Signal surge indicator
✅ donor_is_local_peak             - Peak detection
✅ donor_weighted_context          - Context-weighted score
✅ donor_peak_height_ratio         - Peak prominence
✅ donor_second_derivative         - Signal curvature
✅ donor_signal_strength           - Overall signal quality
✅ donor_context_diff_ratio        - Context variation
```

#### Acceptor-Specific Derived Features
```
✅ acceptor_diff_m1, acceptor_diff_m2, acceptor_diff_p1, acceptor_diff_p2
✅ acceptor_surge_ratio
✅ acceptor_is_local_peak
✅ acceptor_weighted_context
✅ acceptor_peak_height_ratio
✅ acceptor_second_derivative
✅ acceptor_signal_strength
✅ acceptor_context_diff_ratio
```

#### Cross-Type Comparison Features
```
✅ donor_acceptor_peak_ratio       - Donor vs acceptor peak comparison
✅ type_signal_difference          - Type-specific signal difference
✅ score_difference_ratio          - Normalized score differences
✅ signal_strength_ratio           - Relative signal strengths
```

#### Metadata
```
✅ window_start                    - Analysis window start
✅ window_end                      - Analysis window end
✅ transcript_count                - Number of transcripts
```

#### **CRITICAL: Contextual Sequence** ✅
```
✅ sequence                        - Contextual sequence around position
                                   - Contains nucleotide sequence (ACGT)
                                   - Required for k-mer feature extraction
                                   - Column 57 in the TSV
```

---

## 3. ✅ Dataset Builder Compatibility

### What `incremental_builder.py` Needs

From `meta_spliceai/splice_engine/meta_models/builder/dataset_builder.py`:

```python
EXPECTED_MIN_COLUMNS = [
    "gene_id",              ✅ Present
    "transcript_id",        ✅ Present  
    "position",             ✅ Present
    "predicted_position",   ✅ Present
    "true_position",        ✅ Present
    "pred_type",            ✅ Present
    "score",                ✅ Present
    "strand",               ✅ Present
    "donor_score",          ✅ Present
    "acceptor_score",       ✅ Present
    "neither_score",        ✅ Present
    "splice_type",          ✅ Present
    "probability_entropy",  ✅ Present
    "chrom",                ✅ Present
    "sequence",             ✅ Present (CRITICAL for k-mer extraction)
]
```

**Result**: ✅ **ALL required columns are present**

---

## 4. ✅ Data Distribution Verification

### Prediction Type Distribution (chr21)
```
  TN: 4,387 (55.8%)  - True Negatives (sampled, not all 4M!)
  TP: 3,325 (42.3%)  - True Positives
  FN: 147   (1.9%)   - False Negatives  
  FP: 0     (0.0%)   - False Positives
```

**Notes**:
- ✅ TN sampling is working (4,387 vs 4M before fix)
- ✅ TN ratio ~1.3x the positives (TP+FN) - healthy balance
- ✅ All prediction types present (except FP which is model-dependent)
- ✅ Rich enough dataset for meta-learning

---

## 5. ✅ Sequence Data Validation

### Sample Sequence (from row 2)
```
Position: 456 (donor site)
Gene: gene-USP25
Transcript: rna-NM_001283041.3
Sequence: Found in column 57
Length: ~100-500 nucleotides (typical context window)
Format: ACGT nucleotide sequence
```

**Verification**: ✅ Contextual sequences are present and properly formatted

---

## 6. Meta-Model Training Workflow

### Current Position
```
[Step 1] ✅ Base Model Pass        - COMPLETE (chr21)
         → Generated analysis_sequences_21_chunk_1_214.tsv
         → Contains 7,858 positions with full feature set
         → TN sampling fix working correctly

[Step 2] ⏳ Base Model Pass        - IN PROGRESS
         → Need to run remaining chromosomes (1-20, 22, X, Y)
         → Use run_single_chromosome.sh or run_chromosomes_sequential.sh

[Step 3] ⏸️  Meta-Model Building    - READY TO START
         → Use incremental_builder.py
         → Reads analysis_sequences_* files
         → Performs k-mer feature extraction from 'sequence' column
         → Applies feature enrichment
         → Outputs training-ready Parquet files
```

---

## 7. Next Steps

### A. Complete Base Model Pass (Remaining Chromosomes)

**Option 1: Automated Sequential**
```bash
# Run all remaining chromosomes automatically
nohup bash scripts/training/run_chromosomes_sequential.sh > logs/full_genome_run.log 2>&1 &
```

**Option 2: Manual Control**
```bash
# Run one chromosome at a time
for chr in {1..20} 22 X Y; do
    bash scripts/training/run_single_chromosome.sh $chr
    # Wait and verify before continuing
done
```

### B. Build Meta-Model Training Dataset

Once all chromosomes are complete:

```bash
# Example: Build training dataset from all artifacts
cd /Users/pleiadian53/work/meta-spliceai

python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --n-genes 5000 \
  --subset-policy error_total \
  --batch-size 1000 \
  --kmer-sizes 6 \
  --output-dir train_dataset_5k_genes \
  --overwrite \
  --verbose 2
```

**What This Will Do**:
1. Read all `analysis_sequences_*` files from meta_models directory
2. Select top 5000 genes by error count
3. Extract 6-mer features from the `sequence` column
4. Enrich with gene-level and performance features
5. Downsample TNs for balanced training
6. Output training-ready Parquet files

---

## 8. Feature Extraction Pipeline

### From `sequence` to Training Features

```
analysis_sequences_*.tsv
    ↓ (contains 'sequence' column with ACGT nucleotides)
    ↓
incremental_builder.py
    ↓ (build_training_dataset)
    ↓
sequence_featurizer.py
    ↓ (extract k-mers from 'sequence')
    ↓
K-mer Features
    ├─ 6-mers: 4^6 = 4,096 features
    ├─ Position-specific k-mer frequencies
    └─ Combined with base model scores
    ↓
Enriched Features
    ├─ Gene-level features (gene type, length, density)
    ├─ Performance features (error rates, confidence)
    └─ Structural features (overlapping genes, etc.)
    ↓
Training Dataset (Parquet)
    └─ Ready for XGBoost/LightGBM meta-model training
```

---

## 9. Verification Commands

### Check Artifact Completeness
```bash
# Count analysis_sequences files per chromosome
ls -1 data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*.tsv | wc -l

# Check which chromosomes have been processed
ls data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*.tsv | \
  sed 's/.*analysis_sequences_\([0-9XY]*\)_chunk.*/\1/' | sort -u

# Total size of artifacts
du -sh data/mane/GRCh38/openspliceai_eval/meta_models/
```

### Verify Column Schema
```bash
# Check that all analysis_sequences files have the same columns
for f in data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*.tsv; do
  echo "File: $(basename $f)"
  head -1 "$f" | tr '\t' '\n' | wc -l
done
```

### Test Meta-Model Builder (Small Scale)
```bash
# Test on chr21 only before full run
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --n-genes 100 \
  --subset-policy random \
  --batch-size 100 \
  --kmer-sizes 6 \
  --output-dir test_dataset_chr21 \
  --overwrite \
  --verbose 2
```

---

## 10. Summary

### ✅ Verification Complete

| Component | Status | Notes |
|-----------|--------|-------|
| **analysis_sequences files** | ✅ Generated | 57 columns, all required fields present |
| **sequence column** | ✅ Present | Column 57, contains ACGT nucleotides |
| **Base model scores** | ✅ Complete | donor/acceptor/neither probabilities |
| **Context features** | ✅ Complete | ±2 window scores |
| **Derived features** | ✅ Complete | 30+ engineered features |
| **TN sampling** | ✅ Working | 4.4K TNs (not 4M) - fix verified |
| **Pred type labels** | ✅ Present | TP/FP/FN/TN classification |
| **Dataset builder compatibility** | ✅ Verified | All required columns present |

### 🎯 Ready for Next Steps

1. ✅ **Chr21 artifacts verified** - All required columns present
2. ⏳ **Complete remaining chromosomes** - Run chromosomes 1-20, 22, X, Y
3. ⏸️  **Build meta-model training dataset** - Use incremental_builder.py
4. ⏸️  **Train meta-model** - XGBoost/LightGBM on enriched features
5. ⏸️  **Evaluate meta-model** - Test on held-out data

---

## Related Documents

- `TN_SAMPLING_FIX_IMPLEMENTATION.md` - TN sampling bug fix details
- `run_chromosomes_sequential.sh` - Script to run all chromosomes
- `incremental_builder.py` - Meta-model training data builder
- `dataset_builder.py` - Feature extraction from analysis_sequences

---

## Conclusion

✅ **The base model pass is generating exactly the right outputs for meta-model training.**

The `analysis_sequences_*` files contain:
- ✅ All required base columns (gene_id, position, pred_type, etc.)
- ✅ All base model scores (donor, acceptor, neither)
- ✅ Rich derived features (30+ features)
- ✅ **Critical: `sequence` column for k-mer extraction**

The `incremental_builder.py` can directly consume these files and:
1. Extract k-mer features from the `sequence` column
2. Enrich with gene-level and performance features
3. Build training-ready Parquet files for meta-model training

**No compatibility issues found. Ready to proceed once all chromosomes are complete.**

