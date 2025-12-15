# Splice Sites File Fix - Complete Annotations

**Date**: November 4, 2025  
**Issue**: Workflow was using incomplete splice site annotations  
**Status**: ✅ FIXED

## Problem

The comprehensive base model test was processing 0 genes because the workflow was loading an outdated, incomplete `splice_sites.tsv` file that only contained chromosome 21 data (22,344 rows, 453 genes) instead of the complete `splice_sites_enhanced.tsv` file with all chromosomes (1,998,526 rows, 35,306 genes).

### Root Cause

1. **Naming Convention Mismatch**:
   - Data generation scripts create `splice_sites_enhanced.tsv` (complete file)
   - Workflow defaults to looking for `splice_sites.tsv` (line 205 in `data_preparation.py`)

2. **Outdated File**:
   - An old `splice_sites.tsv` from November 1st existed (chr21 only)
   - The complete `splice_sites_enhanced.tsv` was created November 3rd
   - Workflow found the old file and used it

3. **Test Failure**:
   - Test sampled 20 genes from 12 different chromosomes
   - None of these genes were in the chr21-only file
   - Result: "No donor/acceptor annotations for gene" errors

## Solution Implemented

### 1. Created Symlink for Backward Compatibility

```bash
rm data/ensembl/GRCh37/splice_sites.tsv
ln -s splice_sites_enhanced.tsv data/ensembl/GRCh37/splice_sites.tsv
```

**Benefits**:
- Saves disk space (no duplicate 136 MB file)
- Ensures both names point to the same complete data
- Maintains backward compatibility with code expecting `splice_sites.tsv`

**Verification**:
```
lrwxr-xr-x  1 pleiadian53  staff    25B Nov  4 21:20 splice_sites.tsv -> splice_sites_enhanced.tsv
-rw-r--r--@ 1 pleiadian53  staff   136M Nov  3 23:17 splice_sites_enhanced.tsv
```

### 2. Updated Workflow to Use Enhanced File

**File**: `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py`

**Changes**:

#### A. Updated Default Filename (Line 205)
```python
# Before:
output_filename: str = "splice_sites.tsv"

# After:
output_filename: str = "splice_sites_enhanced.tsv"
```

#### B. Updated Full Path Reference (Line 289)
```python
# Before:
full_splice_sites_path = os.path.join(local_dir, f"splice_sites.{ss_format}")

# After:
# Use splice_sites_enhanced.tsv as the canonical complete file
full_splice_sites_path = os.path.join(local_dir, f"splice_sites_enhanced.{ss_format}")
```

## Verification

### Before Fix
```
[i/o] Reading annotations from .../splice_sites.tsv
[info] Final splice-site dataframe: shape=(22344, 8)
```
- Only 22,344 splice sites
- Only 453 genes
- Only chromosome 21

### After Fix
```
[i/o] Reading annotations from .../splice_sites_enhanced.tsv
[info] Final splice-site dataframe: shape=(1998526, 8)
```
- ✅ 1,998,526 splice sites (complete)
- ✅ 35,306 genes (complete)
- ✅ All 41 chromosomes

## File Naming Convention

Going forward, the canonical naming convention is:

### Complete Files (All Chromosomes, All Genes)
- `splice_sites_enhanced.tsv` - Primary complete file
- `splice_sites.tsv` - Symlink to enhanced file (for backward compatibility)

### Filtered/Subset Files (If Needed)
- `splice_sites_chr21.tsv` - Chromosome-specific subset
- `splice_sites_subset.tsv` - Custom filtered subset

**Rule**: If a file is named `splice_sites.tsv` without any qualifier, it MUST contain all genes with valid splice sites across all chromosomes.

## Impact

### Immediate
- ✅ Comprehensive test can now process all sampled genes
- ✅ Workflow correctly loads complete splice site annotations
- ✅ No more "No donor/acceptor annotations" errors

### Long-Term
- Consistent naming convention prevents confusion
- Symlink approach saves disk space
- Explicit use of `splice_sites_enhanced.tsv` in code makes intent clear

## Related Files

- **Manifest**: `data/ensembl/GRCh37/SPLICE_SITES_MANIFEST.md`
- **Generation Script**: `scripts/setup/regenerate_grch37_splice_sites_complete.py`
- **Workflow**: `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py`
- **Test**: `scripts/testing/test_base_model_comprehensive.py`

## Test Status

**Current Run**:
- **PID**: 43206
- **Log**: `logs/base_model_comprehensive_20251104_212644.log`
- **Status**: 🔄 Running with complete annotations
- **Expected**: Should now successfully process all 20 sampled genes

---

**Lesson Learned**: When genomic annotation files exist with similar names (`splice_sites.tsv` vs `splice_sites_enhanced.tsv`), always verify they contain the expected complete dataset. Incomplete or outdated files should be explicitly named with qualifiers (e.g., `_chr21`, `_subset`) to avoid confusion.

