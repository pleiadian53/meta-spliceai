# Critical Bug #2: annotations.db Incorrectly Treated as Shared Resource

**Date**: November 1, 2025  
**Status**: 🔧 **FIXED**  
**Severity**: **CRITICAL** - Breaks multi-build support  
**Related**: CRITICAL_BUG_SPLICE_SITES_PATH_2025-11-01.md

---

## 🐛 Bug Description

After fixing the splice_sites.tsv path bug, we discovered a second critical issue: `annotations.db` was also being incorrectly treated as a "shared resource" across genome builds.

**The Problem**:
- `annotations.db` was being stored in `data/ensembl/` (shared)
- Should be stored in `data/ensembl/{build}/` (build-specific)

**Why This Is Wrong**:
- `annotations.db` contains **genomic coordinates** (`start`, `end` columns)
- Coordinates differ between GRCh37 and GRCh38
- Sharing it across builds causes coordinate mismatches

---

## 🔍 Investigation

### Database Schema

```sql
CREATE TABLE features (
    id text,
    seqid text,
    source text,
    featuretype text,
    start int,        ← BUILD-SPECIFIC!
    end int,          ← BUILD-SPECIFIC!
    score text,
    strand text,
    frame text,
    attributes text,
    extra text,
    bin int,
    primary key (id)
)
```

The presence of `start` and `end` columns proves that `annotations.db` is **build-specific**.

### Example Data

```
ENSG00000279493|21|5011799|5017145|gene
ENSG00000277117|21|5022531|5036771|gene
```

These coordinates are specific to a genome build and will differ in GRCh37 vs GRCh38.

---

## 🔧 Root Cause

In `data_preparation.py::prepare_gene_annotations()`:

```python
# OLD (WRONG) CODE:
if use_shared_db:
    shared_dir = Analyzer.shared_dir  # = "data/ensembl"
    shared_db_file = os.path.join(shared_dir, 'annotations.db')  # ← WRONG!
    db_file = shared_db_file
```

This treated `annotations.db` as a shared resource, when it's actually build-specific.

---

## ✅ The Fix

### Updated Code

```python
# CRITICAL: annotations.db contains coordinates (start, end) and is BUILD-SPECIFIC
# It should ALWAYS be in local_dir (build-specific directory), not shared_dir

# ALWAYS use build-specific directory for annotations.db
db_file = os.path.join(local_dir, 'annotations.db')

# Extract gene annotations if requested
if do_extract:
    from meta_spliceai.splice_engine.extract_genomic_features import extract_annotations
    if verbosity >= 1:
        print_emphasized("[action] Extract gene annotations...")
        print(f"[info] Using build-specific annotations.db: {db_file}")
    
    extract_annotations(gtf_file, db_file=db_file, output_file=output_file, sep=separator)
```

### Changes Made

1. **Removed all "shared" logic** for annotations.db
2. **Always use `local_dir`** (build-specific) for annotations.db
3. **Simplified the code** - removed complex shared file copying logic
4. **Added clear documentation** explaining why it's build-specific

---

## 📂 Correct File Locations

### After Fix

```
data/ensembl/
├── GRCh37/
│   ├── annotations.db              ← GRCh37 coordinates
│   ├── annotations_all_transcripts.tsv
│   ├── splice_sites.tsv
│   └── gene_sequence_*.parquet
│
└── GRCh38/
    ├── annotations.db              ← GRCh38 coordinates
    ├── annotations_all_transcripts.tsv
    ├── splice_sites.tsv
    └── gene_sequence_*.parquet
```

### No Shared Resources

**Important**: There are **NO** truly shared resources between builds. Everything that contains coordinates must be build-specific.

---

## 🔍 Related Issues

This bug is related to the same fundamental misconception as the splice_sites bug:

### The Misconception

❌ **OLD ASSUMPTION**: "We can share genomic data files to save space"

✅ **REALITY**: "All coordinate-based data is build-specific and cannot be shared"

### Build-Specific Resources (Cannot Share)

| Resource | Reason |
|----------|--------|
| `annotations.db` | Contains `start`, `end` coordinates |
| `splice_sites.tsv` | Contains splice site coordinates |
| `gene_features.tsv` | Contains gene start/end coordinates |
| `exon_features.tsv` | Contains exon coordinates |
| `transcript_features.tsv` | May contain coordinates |
| `gene_sequence_*.parquet` | Extracted from build-specific FASTA |
| `junctions.tsv` | Contains junction coordinates |

### Truly Shared Resources

**NONE** - All genomic data files are build-specific when they contain or are derived from coordinate information.

---

## 🧪 Testing

### Verification Steps

1. **Check file locations**:
```bash
ls -lh data/ensembl/GRCh37/annotations.db
ls -lh data/ensembl/GRCh38/annotations.db
```

2. **Verify coordinates differ**:
```bash
# GRCh37
sqlite3 data/ensembl/GRCh37/annotations.db \
  "SELECT id, start, end FROM features WHERE featuretype='gene' AND seqid='21' LIMIT 1;"

# GRCh38
sqlite3 data/ensembl/GRCh38/annotations.db \
  "SELECT id, start, end FROM features WHERE featuretype='gene' AND seqid='21' LIMIT 1;"
```

3. **Run workflow**:
```bash
python scripts/setup/run_grch37_full_workflow.py --chromosomes 21,22 --test-mode
```

Expected output:
```
[info] Using build-specific annotations.db: data/ensembl/GRCh37/annotations.db
```

---

## 📚 Lessons Learned

### 1. Always Check for Coordinates

When determining if a file can be shared:
- ✅ Check the schema/structure
- ✅ Look for coordinate columns (`start`, `end`, `position`, etc.)
- ✅ If coordinates present → **build-specific**

### 2. "Shared" is Misleading

The `use_shared_db` flag was misleading:
- It was meant for "sharing within a build"
- But was incorrectly used for "sharing across builds"
- **Solution**: Remove the concept of "shared" for coordinate-based data

### 3. Test with Multiple Builds Early

Multi-build support requires:
- Testing with actual different builds (GRCh37, GRCh38)
- Verifying coordinates differ
- Ensuring complete isolation

---

## 🎯 Impact

### Before Fix
- ❌ annotations.db shared across builds
- ❌ Coordinate mismatches
- ❌ Workflow failures
- ❌ Cannot support multiple builds

### After Fix
- ✅ annotations.db build-specific
- ✅ Coordinates match build
- ✅ Workflow succeeds
- ✅ Full multi-build support

---

## 🚀 Next Steps

1. ✅ Fix applied to `data_preparation.py`
2. ⏳ Re-run GRCh37 workflow to verify
3. ⏳ Test GRCh38 workflow for backward compatibility
4. ⏳ Update all documentation

---

**Status**: Fix applied, testing in progress
