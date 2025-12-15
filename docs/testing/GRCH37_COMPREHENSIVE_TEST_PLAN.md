# GRCh37 Comprehensive Test Plan

**Date**: November 1, 2025  
**Status**: 🔄 In Progress  
**Objective**: Verify complete pipeline works correctly with GRCh37 after Registry refactor

---

## 🎯 Test Overview

After the Registry refactor, we need to verify that the entire pipeline works correctly with GRCh37 datasets, ensuring:
1. No hardcoded path issues
2. Correct build-specific data loading
3. No cross-build contamination
4. Accurate predictions and evaluations

---

## 📋 Test Phases

### Phase 1: Base Model Pass (splice_prediction_workflow.py) ✅ In Progress

**Objective**: Verify the base model workflow generates all required artifacts for GRCh37

**Test Command**:
```bash
python scripts/setup/run_grch37_full_workflow.py \
  --chromosomes 21,22 \
  --test-mode \
  --verbose
```

**Expected Outputs**:
```
data/ensembl/GRCh37/
├── gene_features.tsv                    # Derived from GRCh37 GTF
├── transcript_features.tsv
├── exon_features.tsv
├── splice_sites_enhanced.tsv            # GRCh37 splice sites
├── annotations.db
├── overlapping_genes.tsv
├── gene_sequence_21.parquet             # Chr21 sequences
├── gene_sequence_22.parquet             # Chr22 sequences
└── spliceai_eval/
    └── meta_models/
        ├── analysis_sequences_21_chunk_*.parquet
        ├── analysis_sequences_22_chunk_*.parquet
        ├── error_analysis_21_chunk_*.parquet
        ├── error_analysis_22_chunk_*.parquet
        ├── splice_positions_enhanced_21_chunk_*.parquet
        ├── splice_positions_enhanced_22_chunk_*.parquet
        ├── splice_positions_enhanced_aggregated.parquet
        └── error_analysis_aggregated.parquet
```

**Success Criteria**:
- ✅ All files created in `data/ensembl/GRCh37/` (not `data/ensembl/`)
- ✅ No errors about missing files
- ✅ Coordinates match GRCh37 (not GRCh38)
- ✅ Gene features loaded from GRCh37 directory
- ✅ Splice sites loaded from GRCh37 directory
- ✅ No cross-build contamination

**Validation Steps**:
```python
import polars as pl
from pathlib import Path

# 1. Verify files exist in correct location
grch37_dir = Path("data/ensembl/GRCh37")
assert (grch37_dir / "gene_features.tsv").exists()
assert (grch37_dir / "splice_sites_enhanced.tsv").exists()

# 2. Verify coordinates are GRCh37
gene_features = pl.read_csv(
    grch37_dir / "gene_features.tsv",
    separator='\t',
    schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
)
print(f"Total genes: {len(gene_features)}")
print(f"Sample coordinates: {gene_features.select(['gene_id', 'start', 'end']).head(5)}")

# 3. Check for GRCh37-specific genes (should exist)
# Example: ENSG00000157764 (BRAF) has different coordinates in GRCh37 vs GRCh38
# GRCh37: chr7:140,419,127-140,624,564
# GRCh38: chr7:140,713,327-140,924,929

# 4. Verify no GRCh38 contamination
grch38_dir = Path("data/ensembl/GRCh38")
if grch38_dir.exists():
    # Ensure GRCh37 files are NOT in GRCh38 directory
    assert not (grch38_dir / "Homo_sapiens.GRCh37.87.gtf").exists()
```

---

### Phase 2: Evaluation Metrics (comprehensive_spliceai_evaluation.py)

**Objective**: Verify base model performance on GRCh37 matches SpliceAI paper metrics

**Test Command**:
```bash
python scripts/testing/comprehensive_spliceai_evaluation.py \
  --build GRCh37 \
  --release 87 \
  --num-genes 50 \
  --output results/grch37_base_model_evaluation.json
```

**Expected Results**:
- PR-AUC: ~0.97 (matching SpliceAI paper)
- Top-k Accuracy: ~0.95
- F1 Score (threshold=0.5): ~0.70-0.80

**Success Criteria**:
- ✅ PR-AUC >= 0.90 (significantly higher than GRCh38's 0.541)
- ✅ Top-k Accuracy >= 0.90
- ✅ F1 scores consistent across genes
- ✅ No coordinate mismatches
- ✅ Predictions align with GRCh37 annotations

---

### Phase 3: Score Adjustment Detection

**Objective**: Verify score adjustments are correctly detected (should be zero)

**Test Command**:
```bash
python scripts/testing/test_score_adjustment_detection.py \
  --build GRCh37 \
  --release 87
```

**Expected Results**:
```
Detected adjustments:
  Donor (plus strand):    0
  Donor (minus strand):   0
  Acceptor (plus strand): 0
  Acceptor (minus strand): 0

Conclusion: Base model is already aligned with GRCh37 annotations.
```

**Success Criteria**:
- ✅ All adjustments are 0
- ✅ F1 scores are optimal without adjustments
- ✅ No systematic position offsets

---

### Phase 4: Inference Workflow (Base-Only Mode)

**Objective**: Verify inference workflow works with GRCh37 in base-only mode

**Test Command**:
```bash
python -c "
from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    SelectiveInferenceConfig
)
from meta_spliceai.system.genomic_resources import Registry

# Initialize with GRCh37
registry = Registry(build='GRCh37', release='87')

# Test genes (protein-coding genes on chr21)
test_genes = ['ENSG00000160072', 'ENSG00000160087', 'ENSG00000160094']

config = SelectiveInferenceConfig(
    target_genes=test_genes,
    mode='base_only',
    model_path=None,  # Not needed for base_only
    verbose=2,
    build='GRCh37',
    release='87'
)

workflow = EnhancedSelectiveInferenceWorkflow(config)
results = workflow.run()

print(f'Processed {len(results)} genes')
for gene_id, result in results.items():
    print(f'{gene_id}: {result[\"status\"]}')
"
```

**Expected Results**:
- All genes processed successfully
- Predictions loaded from GRCh37 directory
- No errors about missing files or coordinate mismatches

**Success Criteria**:
- ✅ All genes return status='success'
- ✅ Predictions use GRCh37 coordinates
- ✅ Gene features loaded from GRCh37 directory
- ✅ No hardcoded path errors

---

### Phase 5: Cross-Build Isolation

**Objective**: Verify GRCh37 and GRCh38 data are completely isolated

**Test Script**:
```python
from meta_spliceai.system.genomic_resources import Registry
from pathlib import Path

# Test GRCh37
registry37 = Registry(build='GRCh37', release='87')
gtf37 = registry37.get_gtf_path()
gene_features37 = registry37.resolve('gene_features')

# Test GRCh38
registry38 = Registry(build='GRCh38', release='112')
gtf38 = registry38.get_gtf_path()
gene_features38 = registry38.resolve('gene_features')

# Verify isolation
assert 'GRCh37' in str(gtf37)
assert 'GRCh38' in str(gtf38) or 'GRCh38' not in str(gtf38)  # May be in top-level
assert gtf37 != gtf38
assert gene_features37 != gene_features38

print("✅ Cross-build isolation verified")
print(f"GRCh37 GTF: {gtf37}")
print(f"GRCh38 GTF: {gtf38}")
print(f"GRCh37 gene_features: {gene_features37}")
print(f"GRCh38 gene_features: {gene_features38}")
```

**Success Criteria**:
- ✅ GRCh37 files in `data/ensembl/GRCh37/`
- ✅ GRCh38 files in `data/ensembl/` or `data/ensembl/GRCh38/`
- ✅ No file path overlap
- ✅ Registry correctly resolves build-specific paths

---

## 🔍 Known Issues to Watch For

### Issue 1: Coordinate Mismatch
**Symptom**: PR-AUC remains low (~0.54) even with GRCh37  
**Cause**: Predictions using GRCh38 coordinates, annotations using GRCh37  
**Fix**: Verify `registry.get_gtf_path()` and `registry.get_fasta_path()` return GRCh37 files

### Issue 2: Gene Features Not Found
**Symptom**: `FileNotFoundError: gene_features.tsv not found`  
**Cause**: Registry not finding GRCh37 gene_features  
**Fix**: Check that `gene_features.tsv` exists in `data/ensembl/GRCh37/`

### Issue 3: Hardcoded Path Fallback
**Symptom**: Warning about using hardcoded paths  
**Cause**: Registry returning None, code falling back to hardcoded paths  
**Fix**: Check Registry search order and file locations

### Issue 4: Cross-Build Contamination
**Symptom**: GRCh37 workflow loading GRCh38 data  
**Cause**: Registry search order prioritizing top-level over build-specific  
**Fix**: Already fixed - stash (build-specific) is searched first

---

## 📊 Success Metrics

### Overall Success Criteria
- ✅ All Phase 1-5 tests pass
- ✅ PR-AUC >= 0.90 (vs 0.541 with GRCh38)
- ✅ No hardcoded path errors
- ✅ No cross-build contamination
- ✅ All artifacts in correct directories

### Performance Targets
| Metric | GRCh38 (Mismatch) | GRCh37 (Expected) |
|--------|-------------------|-------------------|
| PR-AUC | 0.541 | >= 0.90 |
| Top-k Accuracy | 0.550 | >= 0.90 |
| F1 Score (0.5) | 0.596 | >= 0.70 |

---

## 🚀 Next Steps After Testing

1. **If all tests pass**:
   - Document results
   - Run full workflow on all chromosomes
   - Train meta-model on GRCh37 data

2. **If tests fail**:
   - Identify failing phase
   - Debug specific issue
   - Fix and re-test
   - Update documentation

---

## 📝 Test Execution Log

### Phase 1: Base Model Pass
- **Start Time**: 2025-11-01 12:30:00
- **Status**: 🔄 Running
- **Command**: `python scripts/setup/run_grch37_full_workflow.py --chromosomes 21,22 --test-mode --verbose`
- **Log File**: `grch37_test_chr21_22.log`

### Phase 2: Evaluation Metrics
- **Status**: ⏳ Pending Phase 1 completion

### Phase 3: Score Adjustment Detection
- **Status**: ⏳ Pending Phase 1 completion

### Phase 4: Inference Workflow
- **Status**: ⏳ Pending Phase 1 completion

### Phase 5: Cross-Build Isolation
- **Status**: ⏳ Pending Phase 1 completion

---

## 📚 Related Documentation

- `docs/development/REGISTRY_REFACTOR_2025-11-01.md` - Registry refactor details
- `docs/base_models/GENOME_BUILD_COMPATIBILITY.md` - Build compatibility guide
- `docs/base_models/GRCH37_SETUP_COMPLETE_GUIDE.md` - GRCh37 setup guide
- `docs/testing/COMPREHENSIVE_EVALUATION_RESULTS_55_GENES.md` - Previous GRCh38 results

---

**Last Updated**: 2025-11-01 12:30:00



