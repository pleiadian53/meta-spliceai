# Critical Bug: Splice Sites Loaded from Wrong Directory

**Date**: November 1, 2025  
**Status**: 🐛 **IDENTIFIED** → 🔧 **FIXING**  
**Severity**: **CRITICAL** - Breaks multi-build support

---

## 🐛 Bug Description

The GRCh37 workflow failed with:
```
AssertionError: true_donor_positions contain indices out of range for donor_probabilities
```

**Root Cause**: Splice sites were loaded from the wrong directory, causing coordinate mismatch.

---

## 🔍 Investigation

### What Happened

The workflow loaded:
- ❌ **WRONG**: `/data/ensembl/splice_sites.tsv` (GRCh38 coordinates)
- ✅ **SHOULD**: `/data/ensembl/GRCh37/splice_sites.tsv` (GRCh37 coordinates)

This caused:
- Gene sequences: GRCh37 coordinates ✅
- Splice sites: GRCh38 coordinates ❌
- **Result**: Coordinate mismatch → AssertionError

### Files Generated (All Correct)

```
data/ensembl/GRCh37/
├── gene_features.tsv (12M) ✅
├── splice_sites.tsv (6.2M) ✅ Generated but NOT USED
├── splice_sites_enhanced.tsv (136M) ✅
├── gene_sequence_21.parquet (9.5M) ✅ USED
├── gene_sequence_22.parquet (8.1M) ✅ USED
└── annotations_all_transcripts.tsv (157M) ✅
```

### Files Loaded (Mixed Build!)

```
✅ Gene sequences: data/ensembl/GRCh37/gene_sequence_21.parquet (GRCh37)
✅ Gene sequences: data/ensembl/GRCh37/gene_sequence_22.parquet (GRCh37)
❌ Splice sites:   data/ensembl/splice_sites.tsv (GRCh38) ← WRONG!
```

---

## 🔧 Root Cause Analysis

### The Problem Code

In `data_preparation.py::prepare_splice_site_annotations()`:

```python
# Determine paths for shared resources
if use_shared_db:
    # Use the parent directory of the Analyzer's eval_dir for shared resources
    shared_dir = Analyzer.shared_dir  # os.path.dirname(Analyzer.eval_dir)
    shared_db_file = os.path.join(shared_dir, 'annotations.db')
    shared_splice_sites_file = os.path.join(shared_dir, f"splice_sites.{ss_format}")
    # ^^^ THIS IS THE BUG! ^^^
```

### Why This Is Wrong

1. **Splice sites are BUILD-SPECIFIC**, not shared across builds
   - GRCh37 and GRCh38 have different coordinates
   - Cannot be shared like `annotations.db`

2. **`Analyzer.shared_dir` = `"data/ensembl"`**
   - Points to top-level directory
   - Contains GRCh38 data (default build)

3. **Should use `local_dir` instead**
   - `local_dir` is build-specific (e.g., `"data/ensembl/GRCh37"`)
   - Passed as parameter to the function

---

## ✅ The Fix

### What Needs to Change

**File**: `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py`

**Function**: `prepare_splice_site_annotations()`

**Change**: Use `local_dir` instead of `shared_dir` for splice_sites

### Shared vs Build-Specific Resources

| Resource | Type | Directory | Reason |
|----------|------|-----------|--------|
| `annotations.db` | Shared | `data/ensembl/` | Transcript IDs are build-agnostic |
| `splice_sites.tsv` | **Build-Specific** | `data/ensembl/{build}/` | Coordinates differ by build |
| `gene_features.tsv` | **Build-Specific** | `data/ensembl/{build}/` | Coordinates differ by build |
| `gene_sequence_*.parquet` | **Build-Specific** | `data/ensembl/{build}/` | Sequences differ by build |

### Implementation Strategy

1. **Keep `shared_dir` for `annotations.db`** (truly shared)
2. **Use `local_dir` for splice_sites** (build-specific)
3. **Ensure consistency with Registry** (all paths resolved via Registry)

---

## 🎯 Broader Implications

This bug reveals a **fundamental misconception** in the codebase:

### Old Assumption (Wrong)
> "Splice sites can be shared across builds to save space"

### Reality
> "Only transcript-level annotations are build-agnostic. All coordinate-based data is build-specific."

### What This Means

**Build-Specific (Cannot Share)**:
- Splice sites (coordinates)
- Gene features (coordinates)
- Exon features (coordinates)
- Transcript features (coordinates if present)
- Gene sequences (extracted from build-specific FASTA)
- Junctions (coordinates)

**Truly Shared (Can Share)**:
- `annotations.db` (transcript metadata without coordinates)
- Maybe: Gene symbols, biotypes (if no coordinates)

---

## 🔄 Fix Implementation Plan

### Step 1: Update `data_preparation.py`

```python
def prepare_splice_site_annotations(
    local_dir: str,  # Build-specific directory
    gtf_file: str,
    do_extract: bool = True,
    # ... other params ...
):
    # For splice sites, ALWAYS use local_dir (build-specific)
    splice_sites_file_path = os.path.join(local_dir, "splice_sites.tsv")
    
    # Only use shared_dir for truly shared resources
    if use_shared_db:
        shared_dir = Analyzer.shared_dir
        shared_db_file = os.path.join(shared_dir, 'annotations.db')
        # But NOT for splice_sites!
```

### Step 2: Verify Registry Consistency

Ensure all path resolution goes through Registry:
- `registry.resolve('splice_sites')` → build-specific path
- `registry.resolve('gene_features')` → build-specific path
- `registry.get_annotations_db_path()` → can be shared

### Step 3: Update Workflow to Pass Build Info

Ensure `local_dir` is always build-specific:
```python
registry = Registry(build='GRCh37', release='87')
local_dir = str(registry.get_local_dir())  # = "data/ensembl/GRCh37"

prepare_splice_site_annotations(
    local_dir=local_dir,  # Build-specific!
    # ...
)
```

### Step 4: Clean Up Old GRCh38 Files

Move old GRCh38 files to `data/ensembl/GRCh38/`:
```bash
mkdir -p data/ensembl/GRCh38
mv data/ensembl/splice_sites.tsv data/ensembl/GRCh38/
mv data/ensembl/gene_sequence_*.parquet data/ensembl/GRCh38/
```

---

## 🧪 Testing Plan

After fix:

1. **Re-run GRCh37 workflow** (chr21-22)
   - Verify splice_sites loaded from `data/ensembl/GRCh37/`
   - Verify no coordinate mismatch errors

2. **Test GRCh38 workflow** (ensure not broken)
   - Verify splice_sites loaded from `data/ensembl/GRCh38/` or `data/ensembl/`
   - Verify backward compatibility

3. **Test Registry resolution**
   - `Registry(build='GRCh37').resolve('splice_sites')` → GRCh37 path
   - `Registry(build='GRCh38').resolve('splice_sites')` → GRCh38 path

4. **Test cross-build isolation**
   - Run GRCh37 and GRCh38 workflows in sequence
   - Verify no cross-contamination

---

## 📚 Lessons Learned

1. **Coordinate-based data is ALWAYS build-specific**
   - Never assume it can be shared

2. **"Shared resources" must be carefully defined**
   - Only truly build-agnostic data can be shared

3. **Registry should be the single source of truth**
   - All path resolution should go through Registry
   - No hardcoded `Analyzer.shared_dir` assumptions

4. **Test with multiple builds early**
   - Multi-build support is not just about paths
   - It's about understanding which data is build-specific

---

## 🚀 Next Steps

1. ✅ Document bug (this file)
2. 🔧 Fix `data_preparation.py`
3. 🧪 Re-run GRCh37 test
4. ✅ Verify fix works
5. 📝 Update documentation
6. 🎯 Prepare for OpenSpliceAI integration

---

**Status**: Ready to implement fix



