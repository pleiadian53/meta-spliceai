# GRCh37 Coordinate Fix - 2025-11-01

## Critical Issue Discovered

When attempting to evaluate SpliceAI predictions on GRCh37, we discovered **zero performance** (PR-AUC = 0.000) despite using the correct genome build. Investigation revealed **two critical bugs** in the genomic resources system.

---

## Bug #1: Registry Search Order

### Problem
The `Registry.resolve()` method searched paths in this order:
1. `data/ensembl/` (top-level, default build)
2. `data/ensembl/{build}/` (build-specific stash)
3. `data/ensembl/spliceai_analysis/` (legacy)

This meant that when requesting GRCh37 resources, the Registry would find and return **GRCh38 files** from `data/ensembl/` before checking the build-specific directory.

### Impact
- `splice_sites_enhanced.tsv` for GRCh38 was loaded when GRCh37 was requested
- Evaluation compared GRCh37 predictions against GRCh38 annotations
- Result: Zero overlap, zero performance

### Fix
Changed search order to prioritize build-specific directories:
1. `data/ensembl/{build}/` (build-specific stash) ← **NOW FIRST**
2. `data/ensembl/` (top-level, default build)
3. `data/ensembl/spliceai_analysis/` (legacy)

**File**: `meta_spliceai/system/genomic_resources/registry.py`

```python
# OLD (WRONG)
for root in [self.top, self.stash, self.legacy]:
    p = Path(root) / name
    if p.exists():
        return str(p.resolve())

# NEW (CORRECT)
# CRITICAL: Prioritize build-specific stash over top-level to avoid cross-build contamination
for root in [self.stash, self.top, self.legacy]:
    p = Path(root) / name
    if p.exists():
        return str(p.resolve())
```

---

## Bug #2: Missing GRCh37 Gene Features

### Problem
The `gene_features.tsv` file was not generated for GRCh37, only for GRCh38. This file contains critical gene coordinate information (start, end, chromosome, strand).

When generating predictions for GRCh37 genes, the workflow fell back to the GRCh38 `gene_features.tsv`, resulting in:
- Predictions generated using **GRCh38 coordinates**
- Annotations using **GRCh37 coordinates**
- ~467kb coordinate mismatch for gene ENSG00000175130

### Example
**Gene**: ENSG00000175130 (MARCKSL1)

| Build | Coordinate Range | Source |
|-------|------------------|--------|
| GRCh38 (wrong) | 32,333,839 - 32,336,233 | gene_features.tsv (used by predictions) |
| GRCh37 (correct) | 32,800,699 - 32,801,547 | splice_sites_enhanced.tsv (used by evaluation) |
| **Difference** | **~467 kb** | **No overlap!** |

### Impact
- Predictions and annotations had **zero coordinate overlap**
- Evaluation metrics: PR-AUC = 0.000, F1 = 0.000
- Complete evaluation failure

### Fix
Generated GRCh37 gene features using the derive command:

```bash
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --release 87 \
  --gene-features \
  --verbose
```

Result: `data/ensembl/GRCh37/gene_features.tsv` with 57,905 genes

---

## Resolution Steps

### 1. Fixed Registry Search Order ✅
- Modified `registry.py` to prioritize build-specific directories
- Verified GRCh37 and GRCh38 now resolve to different paths

### 2. Generated GRCh37 Gene Features ✅
- Created `data/ensembl/GRCh37/gene_features.tsv`
- Contains correct GRCh37 coordinates for 57,905 genes

### 3. Cleared Old Predictions ✅
- Removed predictions generated with GRCh38 coordinates
- Ensured clean slate for regeneration

### 4. Regenerating Predictions (In Progress)
- Using correct GRCh37 gene features
- Using correct GRCh37 FASTA reference
- Expected: Predictions with GRCh37 coordinates

### 5. Re-evaluation (Pending)
- Will use correct GRCh37 annotations
- Expected: PR-AUC 0.85-0.95 (vs 0.000 before fix)

---

## Lessons Learned

### 1. Build-Specific Resources Must Be Isolated
The genomic resources system now properly isolates build-specific data to prevent cross-contamination.

### 2. Complete Derivation Required
When downloading a new genome build, **all** derived datasets must be generated:
- ✅ GTF (downloaded)
- ✅ FASTA (downloaded)
- ✅ `splice_sites_enhanced.tsv` (derived)
- ✅ `gene_features.tsv` (derived) ← **Was missing!**
- ⚠️ `transcript_features.tsv` (optional)
- ⚠️ `exon_features.tsv` (optional)

### 3. Verification is Critical
Always verify coordinates match between predictions and annotations:
```python
pred_positions = set(predictions_df["position"].to_list())
ann_positions = set(annotations_df["position"].to_list())
overlap = pred_positions & ann_positions
assert len(overlap) > 0, "No coordinate overlap!"
```

---

## Updated Workflow

### Complete GRCh37 Setup

```bash
# 1. Download GTF and FASTA
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --build GRCh37 \
  --release 87 \
  --verbose

# 2. Derive ALL required datasets
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --release 87 \
  --splice-sites \
  --gene-features \
  --consensus-window 2 \
  --verbose

# 3. Generate predictions
python scripts/testing/generate_grch37_predictions.py \
  --num-genes 10 \
  --build GRCh37 \
  --release 87

# 4. Evaluate
python scripts/testing/comprehensive_spliceai_evaluation.py \
  --build GRCh37 \
  --release 87 \
  --num-genes 10
```

---

## Files Modified

1. **meta_spliceai/system/genomic_resources/registry.py**
   - Changed search order to prioritize build-specific directories
   - Prevents cross-build contamination

2. **data/ensembl/GRCh37/gene_features.tsv** (NEW)
   - Generated from GRCh37 GTF
   - Contains correct GRCh37 coordinates for 57,905 genes

---

## Expected Outcome

After regenerating predictions with correct coordinates:

| Metric | Before Fix | After Fix (Expected) |
|--------|------------|---------------------|
| PR-AUC | 0.000 | 0.85-0.95 |
| Top-k Accuracy | 0.000 | 0.80-0.95 |
| F1 Score | 0.000 | 0.70-0.85 |

This should match or closely approach SpliceAI's reported performance (PR-AUC 0.97) since we're now using the same genome build (GRCh37) that SpliceAI was trained on.

---

## Status

- ✅ Bug #1 Fixed (Registry search order)
- ✅ Bug #2 Fixed (Generated GRCh37 gene features)
- ✅ Old predictions cleared
- 🔄 Regenerating predictions with correct coordinates
- ⏳ Re-evaluation pending

**Next**: Wait for prediction generation to complete (~15-20 minutes), then run evaluation.

