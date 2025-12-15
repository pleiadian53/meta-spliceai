# GRCh37 Full Workflow Guide

**Created**: November 1, 2025  
**Purpose**: Run complete splice prediction workflow for GRCh37 genome build

---

## 🎯 Objective

Run the **COMPLETE** `splice_prediction_workflow.py` for GRCh37, generating:
- ✓ All derived genomic features (gene_features, transcript_features, exon_features)
- ✓ Splice site predictions for all positions
- ✓ Analysis sequences (±250bp windows)
- ✓ Error analysis (TP/FP/FN classification)
- ✓ All artifacts in GRCh37-specific directories

---

## ✅ Prerequisites (All Complete)

### 1. GRCh37 Reference Data Downloaded ✅
- **GTF**: `data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf`
- **FASTA**: `data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa`

### 2. Registry Fixed (Build-Specific Path Resolution) ✅
- Prioritizes `data/ensembl/GRCh37/` over `data/ensembl/`
- Prevents cross-build contamination

### 3. Workflow Fixed (Uses Registry for Gene Features) ✅
- No more hardcoded paths
- Correctly handles GRCh37 file schema (`seqname` column)

---

## 🚀 Recommended Workflow

### STEP 1: Test on Chromosomes 21 & 22 (~30-60 minutes)

```bash
python scripts/setup/run_grch37_full_workflow.py \
  --chromosomes 21,22 \
  --test-mode \
  --verbose
```

**This will:**
- Extract all genomic features for chr21 & chr22
- Run SpliceAI predictions
- Generate analysis sequences
- Save artifacts to: `data/ensembl/GRCh37/spliceai_eval/meta_models/`

**Expected output files:**
```
data/ensembl/GRCh37/
├── gene_features.tsv
├── transcript_features.tsv
├── exon_features.tsv
├── splice_sites_enhanced.tsv
├── gene_sequence_21.parquet
├── gene_sequence_22.parquet
└── spliceai_eval/
    └── meta_models/
        ├── analysis_sequences_21_chunk_1_50.parquet
        ├── analysis_sequences_22_chunk_1_50.parquet
        ├── error_analysis_21_chunk_1_50.parquet
        ├── error_analysis_22_chunk_1_50.parquet
        ├── splice_positions_enhanced_21_chunk_1_50.parquet
        └── splice_positions_enhanced_22_chunk_1_50.parquet
```

---

### STEP 2: Verify Artifacts (~1 minute)

```bash
# Check that files were created
ls -lh data/ensembl/GRCh37/*.tsv
ls -lh data/ensembl/GRCh37/spliceai_eval/meta_models/

# Verify coordinates are GRCh37
python -c "
import polars as pl

# Load a sample analysis sequence file
df = pl.read_parquet('data/ensembl/GRCh37/spliceai_eval/meta_models/analysis_sequences_21_chunk_1_50.parquet')
print(f'Sample positions from chr21: {df[\"position\"].head(10).to_list()}')

# Load gene features
gf = pl.read_csv('data/ensembl/GRCh37/gene_features.tsv', separator='\t', schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8})
print(f'Total genes: {len(gf)}')
print(f'Sample gene coordinates:')
print(gf.select(['gene_id', 'start', 'end']).head(5))
"
```

---

### STEP 3: Run Full Workflow (Optional, ~4-8 hours)

```bash
# Process all chromosomes
python scripts/setup/run_grch37_full_workflow.py \
  --chromosomes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y \
  --verbose
```

---

## 📂 Output Directory Structure

```
data/ensembl/
├── GRCh37/                              ← Build-specific directory
│   ├── Homo_sapiens.GRCh37.87.gtf       ← Reference GTF
│   ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa  ← Reference FASTA
│   ├── gene_features.tsv                ← Derived features
│   ├── transcript_features.tsv
│   ├── exon_features.tsv
│   ├── splice_sites_enhanced.tsv
│   ├── overlapping_genes.tsv
│   ├── gene_sequence_*.parquet          ← Per-chromosome sequences
│   └── spliceai_eval/                   ← Evaluation artifacts
│       └── meta_models/
│           ├── analysis_sequences_*_chunk_*.parquet
│           ├── error_analysis_*_chunk_*.parquet
│           ├── splice_positions_enhanced_*_chunk_*.parquet
│           ├── splice_positions_enhanced_aggregated.parquet
│           └── error_analysis_aggregated.parquet
│
└── GRCh38/                              ← Separate GRCh38 data (existing)
    ├── Homo_sapiens.GRCh38.112.gtf
    └── ...
```

---

## 💡 Key Differences from Previous Approach

### Previous (Incomplete):
- ❌ Only generated predictions for 10 genes
- ❌ Used simple inference workflow
- ❌ No analysis sequences
- ❌ No error analysis
- ❌ Missing transcript/exon features

### Current (Complete):
- ✅ Runs full `splice_prediction_workflow.py`
- ✅ Generates ALL derived features
- ✅ Processes all genes on specified chromosomes
- ✅ Extracts analysis sequences (±250bp windows)
- ✅ Performs error analysis (TP/FP/FN)
- ✅ Saves artifacts in GRCh37-specific directories
- ✅ Compatible with meta-model training pipeline

---

## ⚠️ Important Notes

### 1. Processing Time
- **Chr 21 & 22 (test)**: ~30-60 minutes
- **All chromosomes**: ~4-8 hours (depends on hardware)
- **Recommendation**: Start with test mode

### 2. Disk Space
- GRCh37 artifacts will require **~10-50 GB** depending on chromosomes
- Ensure sufficient disk space before running full workflow

### 3. Memory Usage
- **Test mode**: ~8-16 GB RAM
- **Full mode**: ~16-32 GB RAM recommended
- Workflow uses chunking to manage memory

### 4. Coordinate System
- All artifacts will use **GRCh37 coordinates**
- Completely isolated from GRCh38 data
- No cross-contamination

---

## 🎯 Next Steps After Workflow Completes

### 1. Evaluate Base Model Performance
- Calculate PR-AUC, Top-k Accuracy, F1 scores
- Compare to SpliceAI paper metrics (PR-AUC 0.97)
- **Expected**: Much better than GRCh38 results

### 2. Check Score Adjustments
- Run empirical adjustment detection
- **Expected**: Zero adjustments (base model aligned)

### 3. Train Meta-Model (if desired)
- Use GRCh37 artifacts as training data
- Meta-model will learn from GRCh37 base model predictions

### 4. Compare GRCh37 vs GRCh38
- Document performance differences
- Understand impact of genome build mismatch

---

## 📝 Command Reference

### Test Run (Recommended First)
```bash
python scripts/setup/run_grch37_full_workflow.py \
  --chromosomes 21,22 \
  --test-mode \
  --verbose
```

### Specific Genes
```bash
python scripts/setup/run_grch37_full_workflow.py \
  --target-genes BRCA1,TP53,EGFR \
  --chromosomes 17,11,7 \
  --verbose
```

### Full Run (All Chromosomes)
```bash
python scripts/setup/run_grch37_full_workflow.py \
  --chromosomes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y \
  --verbose
```

### Help
```bash
python scripts/setup/run_grch37_full_workflow.py --help
```

---

## 🐛 Troubleshooting

### Issue: "GTF file not found"
**Solution**: Ensure GRCh37 data was downloaded:
```bash
bash scripts/setup/download_grch37_data.sh
```

### Issue: "Out of memory"
**Solution**: Use test mode or process fewer chromosomes at a time:
```bash
python scripts/setup/run_grch37_full_workflow.py --chromosomes 21 --test-mode
```

### Issue: "Column 'seqname' parsing error"
**Solution**: This should be fixed in the latest code. If you still see this, ensure you have the latest version of `enhanced_selective_inference.py` with the schema override fix.

---

## 📚 Related Documentation

- **Genome Build Compatibility**: `docs/base_models/GENOME_BUILD_COMPATIBILITY.md`
- **GRCh37 Download Guide**: `docs/base_models/GRCH37_DOWNLOAD_GUIDE.md`
- **Coordinate Fix Details**: `docs/testing/GRCH37_COORDINATE_FIX_2025-11-01.md`
- **Session Summary**: `docs/testing/SESSION_COMPLETE_2025-11-01_GRCH37_EVALUATION.md`

---

## ✅ Status

**All prerequisites complete** ✅  
**Script ready to run** ✅  
**Infrastructure tested** ✅  

**Next**: Run STEP 1 (test on chr21 & chr22) to verify the complete workflow works correctly with GRCh37.

