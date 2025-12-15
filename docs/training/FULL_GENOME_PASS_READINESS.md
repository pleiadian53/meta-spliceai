# Full Genome Base Model Pass - Readiness Checklist

**Date**: 2025-11-11  
**Status**: ✅ **READY**

## Overview

This document verifies readiness for running a full-genome base model pass with OpenSpliceAI to generate training data for the meta-learning layer.

## Readiness Checklist

### ✅ Workflow Components

- [x] **Enhanced Splice Prediction Workflow** (`splice_prediction_workflow.py`)
  - ✅ Supports OpenSpliceAI as base model
  - ✅ Processes all genes when `target_genes=None`
  - ✅ Generates gene manifest
  - ✅ Saves nucleotide-level scores (optional)
  - ✅ Handles per-chromosome processing for memory efficiency

- [x] **Artifact Management**
  - ✅ Artifact manager configured correctly
  - ✅ Analysis sequences saved per-chunk (`analysis_sequences_*_chunk_*.tsv`)
  - ✅ Positions saved (`full_splice_positions_enhanced.tsv`)
  - ✅ Errors saved (`full_splice_errors.tsv`)
  - ✅ Gene manifest saved (`gene_manifest.tsv`)
  - ✅ Nucleotide scores saved (if enabled)

- [x] **Base Model Support**
  - ✅ OpenSpliceAI model loading
  - ✅ PyTorch model inference
  - ✅ Correct tensor shape handling
  - ✅ Device management (CPU/MPS/CUDA)

- [x] **Gene Mapping**
  - ✅ Enhanced Gene Mapper for cross-build mapping
  - ✅ Standardized build naming (`GRCh38_MANE`)
  - ✅ External ID mapping (MANE→Ensembl)

### ✅ Evaluation Components

- [x] **Comprehensive Evaluation Module** (`comprehensive_evaluation.py`)
  - ✅ F1 Score calculation
  - ✅ ROC-AUC calculation
  - ✅ Average Precision (AP) calculation
  - ✅ Top-k accuracy calculation
  - ✅ Per-splice-type metrics (donor/acceptor)
  - ✅ JSON output for metrics

- [x] **Full Genome Pass Script** (`run_full_genome_base_model_pass.py`)
  - ✅ Command-line interface
  - ✅ OpenSpliceAI support
  - ✅ Comprehensive evaluation integration
  - ✅ Artifact verification
  - ✅ Summary generation

### ✅ Data Requirements

- [x] **Genomic Resources**
  - ✅ MANE/GRCh38 gene features (`gene_features.tsv`)
  - ✅ MANE/GRCh38 splice sites (`splice_sites_enhanced.tsv`)
  - ✅ GRCh38 reference genome (FASTA)
  - ✅ MANE GTF file

- [x] **Model Files**
  - ✅ OpenSpliceAI model weights
  - ✅ Model loading utilities

### ⚠️ Considerations

1. **Memory Management**
   - ✅ Per-chromosome processing (memory efficient)
   - ✅ Chunk-based processing
   - ✅ Analysis sequences saved per-chunk (not aggregated)
   - ⚠️ Nucleotide scores can be large (disabled by default)

2. **Runtime**
   - ⚠️ Full genome pass will take several hours
   - ⚠️ Recommend running in background or on compute cluster
   - ✅ Progress tracking with tqdm

3. **Storage**
   - ⚠️ Analysis sequences: ~100s of MB to GBs (per-chunk files)
   - ⚠️ Positions: ~100s of MB
   - ⚠️ Nucleotide scores: ~GBs (if enabled)
   - ✅ Artifacts organized by test_name

## Usage

### Basic Usage

```bash
# Run full genome pass with OpenSpliceAI
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome
```

### With Options

```bash
# Full genome pass with all options
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome \
    --threshold 0.5 \
    --no-tn-sampling \
    --verbosity 1
```

### Output Structure

```
data/mane/GRCh38/openspliceai_eval/
└── production/
    └── full_genome_openspliceai_YYYYMMDD_HHMMSS/
        ├── predictions/
        │   ├── analysis_sequences_1_chunk_1_500.tsv
        │   ├── analysis_sequences_1_chunk_501_1000.tsv
        │   ├── ...
        │   ├── full_splice_positions_enhanced.tsv
        │   ├── full_splice_errors.tsv
        │   ├── gene_manifest.tsv
        │   └── nucleotide_scores.tsv (if enabled)
        ├── evaluation_metrics.json
        └── full_genome_pass_summary.json
```

## Expected Outputs

### 1. Analysis Sequences (`analysis_sequences_*_chunk_*.tsv`)

**Purpose**: Training data for meta-learning model

**Columns**:
- `gene_id`, `transcript_id`, `strand`
- `position`, `splice_type`, `pred_type`
- `donor_score`, `acceptor_score`, `neither_score`
- `sequence` (±250bp window)
- Derived features (context scores, probability features, etc.)

**Format**: Per-chunk TSV files (memory efficient)

### 2. Positions (`full_splice_positions_enhanced.tsv`)

**Purpose**: All analyzed positions with predictions

**Columns**: All position-level features and predictions

### 3. Errors (`full_splice_errors.tsv`)

**Purpose**: False positives and false negatives for analysis

**Columns**: Error analysis with TP/FP/FN classifications

### 4. Gene Manifest (`gene_manifest.tsv`)

**Purpose**: Track which genes were processed successfully

**Columns**: Gene ID, status, counts, processing time

### 5. Evaluation Metrics (`evaluation_metrics.json`)

**Purpose**: Comprehensive performance metrics

**Metrics**:
- F1 Score
- Precision, Recall, Accuracy
- ROC-AUC
- Average Precision (AP)
- Top-K Accuracy
- Per-splice-type metrics

### 6. Summary (`full_genome_pass_summary.json`)

**Purpose**: Complete summary of the run

**Contents**: Configuration, metrics, paths, manifest summary

## Verification Steps

After running, verify:

1. **Artifacts Exist**
   ```bash
   ls -lh data/mane/GRCh38/openspliceai_eval/production/full_genome_openspliceai_*/predictions/
   ```

2. **Analysis Sequences Generated**
   ```bash
   ls analysis_sequences_*.tsv | wc -l  # Should have many chunk files
   ```

3. **Gene Manifest Complete**
   ```bash
   # Check how many genes processed
   tail -n +2 gene_manifest.tsv | wc -l
   ```

4. **Metrics Calculated**
   ```bash
   cat evaluation_metrics.json | jq '.f1, .roc_auc, .average_precision'
   ```

## Next Steps After Completion

1. **Review Metrics**
   - Check F1, ROC-AUC, AP scores
   - Verify reasonable performance
   - Check per-splice-type metrics

2. **Verify Data Quality**
   - Check gene manifest for processing success rate
   - Verify analysis sequences have expected columns
   - Check for any missing chunks

3. **Prepare for Meta-Learning**
   - Use analysis_sequences_* files as training data
   - Load positions for feature engineering
   - Use gene manifest for train/test splits

4. **Proceed to Training**
   - Run meta-learning training pipeline
   - Use generated artifacts as input

## Troubleshooting

### Issue: Out of Memory

**Solution**: 
- Ensure per-chromosome processing is enabled
- Disable nucleotide scores if not needed
- Process chromosomes sequentially

### Issue: Missing Analysis Sequences

**Solution**:
- Check artifact manager paths
- Verify `save_analysis_sequences` is called
- Check disk space

### Issue: Low Performance Metrics

**Solution**:
- Check threshold value
- Verify splice site annotations are correct
- Check position adjustments

## Status

✅ **READY FOR FULL GENOME PASS**

All components are in place:
- ✅ Workflow supports OpenSpliceAI
- ✅ Artifact management configured
- ✅ Comprehensive evaluation implemented
- ✅ Script ready to run
- ✅ Documentation complete

**Recommendation**: Proceed with full genome pass using OpenSpliceAI.

---

**Date**: 2025-11-11  
**Status**: ✅ READY  
**Next Action**: Run full genome pass




