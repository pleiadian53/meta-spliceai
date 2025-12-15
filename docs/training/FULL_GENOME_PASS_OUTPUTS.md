# Full Genome Base Model Pass - Output Locations

**Date**: 2025-11-12  
**Status**: ✅ **RUNNING**

## 1. Full Coverage Mode Status

✅ **DISABLED** - Nucleotide-level scores are **NOT** being saved

**Confirmation**:
- Log shows: `Nucleotide Scores: Disabled`
- Script default: `save_nucleotide_scores=False` (unless `--save-nucleotide-scores` flag is used)
- **Rationale**: Full coverage mode generates massive data volumes (GBs per gene)

**What This Means**:
- ✅ Only splice site positions are saved (not every nucleotide)
- ✅ Analysis sequences contain ±250bp windows around positions
- ✅ Much smaller data volume (100s of MB to GBs total, not GBs per gene)
- ✅ Suitable for meta-learning training dataset

## 2. Output Directory Structure

### Base Directory

```
data/mane/GRCh38/openspliceai_eval/
```

### Production Mode Artifacts Directory

For **production mode** with **full_genome** coverage:

```
data/mane/GRCh38/openspliceai_eval/meta_models/
```

**Note**: In production mode, artifacts go directly to `meta_models/` (no test subdirectory).

### Expected Output Files

All artifacts will be saved in:

```
data/mane/GRCh38/openspliceai_eval/meta_models/
├── analysis_sequences_1_chunk_1_500.tsv
├── analysis_sequences_1_chunk_501_1000.tsv
├── analysis_sequences_1_chunk_1001_1500.tsv
├── ... (many chunk files, one per chromosome chunk)
├── analysis_sequences_2_chunk_1_500.tsv
├── ... (continues for all chromosomes)
├── full_splice_positions_enhanced.tsv
├── full_splice_errors.tsv
├── gene_manifest.tsv
├── evaluation_metrics.json
└── full_genome_pass_summary.json
```

### File Descriptions

1. **`analysis_sequences_*_chunk_*.tsv`**
   - **Purpose**: Training data for meta-learning model
   - **Format**: Per-chunk TSV files (memory efficient)
   - **Content**: Sequences ±250bp around each position with all features
   - **Count**: Many files (one per chromosome chunk)
   - **Size**: 100s of MB to GBs total

2. **`full_splice_positions_enhanced.tsv`**
   - **Purpose**: All analyzed positions with predictions
   - **Content**: Position-level features and predictions
   - **Size**: ~100s of MB

3. **`full_splice_errors.tsv`**
   - **Purpose**: False positives and false negatives
   - **Content**: Error analysis with TP/FP/FN classifications
   - **Size**: Smaller (only errors)

4. **`gene_manifest.tsv`**
   - **Purpose**: Track which genes were processed
   - **Content**: Gene ID, status, counts, processing time
   - **Size**: Small (~MB)

5. **`evaluation_metrics.json`**
   - **Purpose**: Comprehensive performance metrics
   - **Content**: F1, ROC-AUC, AP, top-k accuracy
   - **Size**: Very small (KB)

6. **`full_genome_pass_summary.json`**
   - **Purpose**: Complete summary of the run
   - **Content**: Configuration, metrics, paths, manifest summary
   - **Size**: Very small (KB)

## 3. Current Run Details

**Test Name**: `full_genome_openspliceai_20251112_004644`  
**Base Model**: OpenSpliceAI  
**Mode**: Production  
**Coverage**: Full Genome  

**Output Location**:
```
/Users/pleiadian53/work/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/
```

**Note**: The `test_name` is used for logging/identification but does **NOT** affect the output path in production mode. All production artifacts go to the same `meta_models/` directory.

## 4. Verification Commands

### Check Output Directory

```bash
# Check if directory exists
ls -ld data/mane/GRCh38/openspliceai_eval/meta_models/

# List all artifacts
ls -lh data/mane/GRCh38/openspliceai_eval/meta_models/

# Count analysis sequence files
ls data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*.tsv | wc -l

# Check file sizes
du -sh data/mane/GRCh38/openspliceai_eval/meta_models/
```

### Check Specific Files

```bash
# Check if main files exist
ls -lh data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv
ls -lh data/mane/GRCh38/openspliceai_eval/meta_models/gene_manifest.tsv
ls -lh data/mane/GRCh38/openspliceai_eval/meta_models/evaluation_metrics.json

# Verify no nucleotide scores (should not exist)
ls data/mane/GRCh38/openspliceai_eval/meta_models/nucleotide_scores.tsv 2>&1
# Expected: "No such file or directory"
```

## 5. Data Volume Estimates

### With Full Coverage Disabled (Current Run)

- **Analysis sequences**: ~100s of MB to GBs (depends on number of positions)
- **Positions**: ~100s of MB
- **Errors**: ~10s of MB
- **Manifest**: ~MB
- **Total**: ~1-10 GB (reasonable for training)

### If Full Coverage Were Enabled (NOT the case)

- **Nucleotide scores**: ~GBs per gene (would be massive!)
- **Total**: Could be 100s of GBs or TBs
- **Not recommended** for full genome pass

## 6. Summary

✅ **Full Coverage Mode**: **DISABLED** (correct for training data generation)  
✅ **Output Location**: `data/mane/GRCh38/openspliceai_eval/meta_models/`  
✅ **File Structure**: Per-chunk analysis sequences + aggregated positions/errors  
✅ **Data Volume**: Reasonable (~1-10 GB total)  

---

**Last Updated**: 2025-11-12  
**Status**: ✅ Running with correct configuration

