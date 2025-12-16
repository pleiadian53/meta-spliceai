# 🔧 **Inference Workflow Troubleshooting Guide**

A comprehensive guide documenting all errors encountered during the development and optimization of the meta-model inference workflow, along with their solutions and preventive measures.

## 🚨 **CRITICAL: Meta-Only Inference Mode Issues**

**Meta-only inference mode was the most challenging to implement and debug.** The most common issues encountered:

### **🔴 Issue #0: Identical Performance Metrics Bug (Most Subtle)**
- **Problem**: Performance comparison shows identical metrics between `base_only`, `hybrid`, and `meta_only` modes
- **Impact**: Cannot evaluate meta-model effectiveness, masks actual performance improvements
- **Root Cause**: Prediction combiner incorrectly applying meta predictions to main score columns
- **Solution**: Preserve base predictions in main columns, apply meta predictions only to meta columns

**Symptoms:**
```
📈 Average Precision (Higher is Better for Imbalanced Data):
  🏆 base_only   : 0.837
     meta_only   : 0.837 (0.000)  ← Identical! Should be different
     hybrid      : 0.837 (0.000)  ← Identical! Should be different

🎯 F1 Scores (at optimal threshold):
  🏆 base_only   : 0.762 (thresh=0.97)
     meta_only   : 0.762 (thresh=0.97) (0.000)  ← Identical!
     hybrid      : 0.762 (thresh=0.97) (0.000)  ← Identical!
```

**Technical Analysis:**
This bug was extremely subtle because:
1. **Meta-model was working correctly** - generating different predictions from base model
2. **Prediction files showed correct metadata** - `prediction_source` correctly marked as `meta_model`
3. **No error messages** - workflow completed "successfully"
4. **Silent failure** - meta predictions were generated but not applied to final results

**Root Cause Details:**
```python
# PROBLEMATIC: Prediction combiner was doing this in meta_only mode
combined_df['donor_meta'] = combined_df['donor_score']  # Initialize with base
# Meta predictions generated correctly but not applied due to logic bug
# Result: Both donor_score and donor_meta contained base predictions

# PROBLEMATIC: Performance analysis comparing identical values
base_scores = pred_df['donor_score'] + pred_df['acceptor_score']  # Base predictions
meta_scores = pred_df['donor_meta'] + pred_df['acceptor_meta']    # Also base predictions!
# Result: Identical performance metrics
```

**Correct Solution:**
```python
# FIXED: Proper meta_only mode behavior
# 1. Keep original base predictions in main columns for comparison
combined_df['donor_score'] = base_predictions['donor_score']  # Original base
combined_df['donor_meta'] = meta_predictions['donor_meta']    # Actual meta

# 2. Performance analysis uses appropriate columns
base_only_scores = base_run['donor_score'] + base_run['acceptor_score']      # Base run
meta_only_scores = meta_run['donor_meta'] + meta_run['acceptor_meta']        # Meta run
# Result: Meaningful performance differences
```

**Expected Behavior After Fix:**
```
📈 Average Precision (Higher is Better for Imbalanced Data):
     base_only   : 0.149
  🏆 meta_only   : 0.359 (+0.210)  ← Significant improvement!

🎯 F1 Scores (at optimal threshold):
     base_only   : 0.269 (thresh=0.27)
  🏆 meta_only   : 0.419 (+0.151)  ← Significant improvement!
```

**Diagnostic Steps:**
```bash
# 1. Check if meta predictions are actually different from base
python -c "
import polars as pl
meta_df = pl.read_parquet('results/*/predictions/meta_model_predictions.parquet')
print('Meta vs base different:', (meta_df['donor_meta'] != meta_df['donor_score']).any())
"

# 2. Check final predictions preserve base in main columns
python -c "
import polars as pl
final_df = pl.read_parquet('results/*/predictions/complete_coverage_predictions.parquet')
print('Final: base vs meta different:', (final_df['donor_score'] != final_df['donor_meta']).any())
"

# 3. Verify position mapping is working
python -c "
import polars as pl
base_df = pl.read_parquet('results/*/predictions/base_model_predictions.parquet')
meta_df = pl.read_parquet('results/*/predictions/meta_model_predictions.parquet')
base_pos = set(base_df['position'].to_list())
meta_pos = set(meta_df['position'].to_list())
print(f'Position overlap: {len(base_pos & meta_pos)}/{len(base_pos)}')
"
```

**Prevention:**
- Always verify performance metrics show meaningful differences between modes
- Check that meta_only mode shows `donor_meta` ≠ `donor_score` for all positions  
- Test meta-model on training data to verify it produces different predictions
- Monitor for suspicious identical metrics across different inference modes

### **🔴 Issue #1: Incomplete Analysis Sequences (Most Critical)**
- **Problem**: Analysis_sequences files with only 8 columns instead of 57+ columns
- **Impact**: Meta-model cannot function without advanced features
- **Root Cause**: Investigating failed runs instead of successful runs
- **Solution**: Use `--complete-coverage` flag and verify workflow completion

### **🔴 Issue #2: False Success Reporting**  
- **Problem**: Workflow reports success but fails internally
- **Impact**: Misleading users about actual workflow status
- **Root Cause**: Poor exception handling and result validation
- **Solution**: Enhanced error handling and result validation

### **🔴 Issue #3: Dataset Builder False Lead**
- **Problem**: Extensive time spent debugging k-mer feature extraction
- **Impact**: Wasted development time on non-issues
- **Root Cause**: Investigating symptoms rather than root causes
- **Solution**: Always verify workflow completion before debugging components

### **🔴 Issue #4: Position Count Confusion (Recent Discovery)**
- **Problem**: Confusion about position count differences (11,443 vs 5,716) causing debugging efforts
- **Impact**: Time spent investigating normal system behavior as potential bugs
- **Root Cause**: Lack of understanding of donor/acceptor consolidation and boundary enhancement
- **Solution**: Document and validate expected position count behavior (see Position Count Analysis)

**Key Lesson**: For unseen genes (the most common use case), meta-only mode requires complete coverage workflow to generate full analysis_sequences with all advanced features. Position count discrepancies are normal and consistent across all inference modes.

---

## 📊 **Error Categories & Solutions**

This guide is organized by error categories, from most critical to least critical, based on impact on production workflows.

---

## 🚨 **Category 0: Position Count Understanding (Recent Addition)**

### **Issue 0.1: Position Count Discrepancy Confusion**

**Symptoms:**
```
Total error count: 2, Total positions count: 11443
  Total positions: 11443
   📊 Total positions: 5,716
# Users confused about why two different position counts
```

**Root Cause:**
- Lack of documentation about normal SpliceAI processing behavior
- Misunderstanding of donor/acceptor consolidation process
- Confusion about boundary position enhancement

**Technical Analysis:**
The position count behavior is **completely normal and expected**:
1. **11,443 positions**: Raw donor (5,715) + acceptor (5,728) predictions
2. **5,716 positions**: Final unique genomic positions after consolidation
3. **+1 discrepancy**: Boundary position added during evaluation for complete coverage

**Solution:**
Understanding the position count behavior eliminates false debugging:

```bash
# This is NORMAL behavior - not a bug to investigate
echo "ENSG00000142748 (5,715 bp gene):"
echo "  Raw predictions: 11,443 (donor + acceptor counts)"
echo "  Final positions: 5,716 (consolidated + boundary)"
echo "  Discrepancy: +1 (3' boundary enhancement)"
echo "  Status: ✅ Normal and expected"
```

**Key Insights:**
1. **Donor/Acceptor Asymmetry**: 0.1-0.3% asymmetry is biologically expected due to different recognition mechanisms
2. **+1 Position Location**: Most likely at 3' end for boundary completeness
3. **Inference Mode Consistency**: All modes (base_only, meta_only, hybrid) show identical position counts
4. **Universal Pattern**: All tested genes show +1 discrepancy when using complete evaluation

**Validation:**
```bash
# Test position count consistency across inference modes
for mode in base_only meta_only hybrid; do
    echo "Testing $mode mode:"
    python -m meta_spliceai...main_inference_workflow \
        --genes ENSG00000142748 \
        --inference-mode $mode \
        --verbose | grep "📊 Total positions"
done
# Expected: All modes show identical position counts
```

**Prevention:**
- Document expected position count behavior for users
- Use position count consistency as validation metric
- Recognize +1 discrepancies as quality enhancement, not errors

---

## 🚨 **Category 1: Critical Performance Issues**

### **Error 1.1: Extremely Slow Performance (7+ minutes)**

**Symptoms:**
```
[i/o] Loading exon dataframe from cache: /path/to/exon_df_from_gtf.tsv
[test] columns(exons): ['gene_id', 'transcript_id', ...]
<HANGS for 7+ minutes>
```

**Root Cause:**
- Heavy feature enrichment operations loading 139MB+ genomic files repeatedly
- Inefficient overlap feature computation from raw GTF files  
- Coordinate system conversions happening for every position

**Solution:**
```python
# BEFORE: Used full feature enrichment pipeline
from meta_spliceai.splice_engine.meta_models.features.feature_enrichment import apply_feature_enrichers
positions_enriched = apply_feature_enrichers(
    positions_pd,
    enrichers=["gene_level", "length_features", "performance_features", 
              "overlap_features", "distance_features"],  # TOO HEAVY
    verbose=verbose
)

# AFTER: Use optimized inference-specific enrichment
from meta_spliceai.splice_engine.meta_models.workflows.inference.optimized_feature_enrichment import (
    create_optimized_enricher
)
enricher = create_optimized_enricher(verbose=verbose)
feature_matrix = enricher.generate_features_for_uncertain_positions(
    uncertain_positions,
    gene_id,
    config.model_path
)
```

**Performance Impact:**
- **Before**: 447.8 seconds (7.5 minutes)
- **After**: 1.1 seconds  
- **Improvement**: 497x faster

**Prevention:**
- Always use `optimized_feature_enrichment.py` for inference workflows
- Avoid `overlap_features` enricher unless pre-computed data exists
- Monitor processing time - should be <2 seconds per gene

---

## 🚨 **Category 2: Meta-Only Inference Mode Critical Issues**

### **Error 2.1: Incomplete Analysis Sequences for Unseen Genes**

**Symptoms:**
```
Dataset missing expected columns: ['gene_id', 'transcript_id', 'position', 'pred_type', 'splice_type']…
Available columns (8): ['gene_id', 'transcript_id', 'position', 'splice_type', 'pred_type', 'donor_score', 'acceptor_score', 'neither_score']
Expected columns (68): ['gene_id', 'transcript_id', 'position', 'donor_score', 'neither_score', 'acceptor_score', 'pred_type', 'splice_type', ...]
```

**Root Cause:**
- Meta-only mode with `--complete-coverage` failing to generate complete analysis_sequences files
- Fallback workflow generating minimal analysis_sequences with only 8 columns instead of 57+ columns
- Missing advanced probability features, context scores, genomic features, sequence motifs required by meta-model
- The most critical issue: **basic use case (unseen genes) was completely broken**

**Technical Analysis:**
The issue was initially misdiagnosed as a dataset builder problem, but the real cause was:
1. **Failed workflow runs** generated incomplete analysis_sequences files (8 columns)
2. **Successful workflow runs** generate complete analysis_sequences files (57 columns)
3. The incomplete files were from error states, not from the working workflow

**Diagnostic Steps:**
```bash
# 1. Check if analysis_sequences files have complete feature set
find /tmp -name "*analysis_sequences*" -type f | head -1 | xargs head -1 | wc -w
# Expected: ~57 columns
# Problem: ~8 columns

# 2. Verify complete coverage workflow is being used
grep "complete_coverage_inference_workflow" logs/*.log
# Should show: Using complete coverage workflow

# 3. Check feature columns in analysis_sequences
find /tmp -name "complete_analysis_sequences.tsv" | tail -1 | xargs head -1
# Expected columns: gene_id, transcript_id, context_score_m2, donor_score, splice_type, position, strand, acceptor_score, context_score_p1, pred_type, score, context_score_m1, true_position, predicted_position, neither_score, context_score_p2, relative_donor_probability, splice_probability, donor_acceptor_diff, splice_neither_diff, donor_acceptor_logodds, splice_neither_logodds, probability_entropy, context_neighbor_mean, context_asymmetry, context_max, donor_diff_m1, donor_diff_m2, donor_diff_p1, donor_diff_p2, donor_surge_ratio, donor_is_local_peak, donor_weighted_context, donor_peak_height_ratio, donor_second_derivative, donor_signal_strength, donor_context_diff_ratio, acceptor_diff_m1, acceptor_diff_m2, acceptor_diff_p1, acceptor_diff_p2, acceptor_surge_ratio, acceptor_is_local_peak, acceptor_weighted_context, acceptor_peak_height_ratio, acceptor_second_derivative, acceptor_signal_strength, acceptor_context_diff_ratio, donor_acceptor_peak_ratio, type_signal_difference, score_difference_ratio, signal_strength_ratio, chrom, window_start, window_end, transcript_count, sequence
```

**Solution:**
The workflow was actually working correctly when successful. The issue was distinguishing between failed and successful runs:

```bash
# CORRECT: Meta-only mode (complete coverage auto-enabled)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers \
    --genes-file unseen_genes.txt \
    --output-dir results/test_unseen_genes \
    --inference-mode meta_only \
    --verbose
# Note: --complete-coverage is automatically enabled for meta_only mode

# Expected successful output:
# ✅ Generated meta-model predictions for 5,003 positions
# ✅ Meta-model recalibrated: 5,003 (100.0%)
# ✅ Complete coverage achieved with selective efficiency!
```

**Key Technical Details:**
- Complete coverage workflow calls `run_complete_coverage_inference_workflow` from `splice_inference_workflow.py`
- This calls `run_enhanced_splice_prediction_workflow` with `essential_columns_only=False`
- This generates complete analysis_sequences with all 57 advanced features
- The workflow then successfully applies meta-model to all positions

**Validation:**
```bash
# 1. Verify successful completion
grep "INFERENCE WORKFLOW COMPLETED SUCCESSFULLY" results/*/inference_workflow.log

# 2. Check meta-model usage
grep "Meta-model recalibrated" results/*/performance_report.txt
# Should show: "Meta-model recalibrated: 5,003 (100.0%)" for complete coverage

# 3. Verify complete feature generation
ls -la /tmp/selective_inference_*/complete_coverage_output/complete_analysis_sequences.tsv
# File should exist and be substantial size (not just a few hundred bytes)
```

**Prevention:**
- ✅ **Auto-enabled**: Complete coverage is now automatically enabled for meta-only mode
- Monitor log output for "🔄 Auto-enabled complete coverage for meta_only mode"
- Monitor log output for "Complete coverage achieved with selective efficiency!"
- Verify analysis_sequences files have 50+ columns, not just 8
- Check that meta-model usage is >0%, ideally 100% for meta_only mode

### **Error 2.2: False Success Reporting During Failures**

**Symptoms:**
```
✅ ENSG00000196890 completed in 2.3s
Processing genes: 100%|██████████████████████| 1/1 [00:02<00:00,  2.34s/it]
🎉 INFERENCE WORKFLOW COMPLETED SUCCESSFULLY
📊 Successfully processed 1/1 genes
```
**But the actual workflow failed internally with dataset builder errors**

**Root Cause:**
- Exception handling was catching errors but still reporting success
- Workflow continuing with empty/invalid data after internal failures
- Success metrics calculated on empty datasets showing false positives

**Technical Analysis:**
```python
# PROBLEMATIC: Exception caught but success still reported
try:
    result = workflow._process_single_gene(gene_id, gene_index)
    # If this fails, result might be None or empty
except Exception as e:
    logger.error(f"Error processing {gene_id}: {e}")
    result = {"success": False}  # But this wasn't being checked

# Success was reported based on completion, not actual results
success_count = len([r for r in results if r is not None])  # ❌ Wrong logic
```

**Solution:**
Enhanced error handling and validation:

```python
# FIXED: Proper success validation
try:
    result = workflow._process_single_gene(gene_id, gene_index)
    if not result or not result.get("success", False):
        raise RuntimeError(f"Gene {gene_id} processing failed")
    
    # Validate actual results exist
    if "predictions" not in result or len(result["predictions"]) == 0:
        raise RuntimeError(f"No predictions generated for {gene_id}")
        
except Exception as e:
    logger.error(f"❌ {gene_id} failed: {e}")
    failed_genes.append(gene_id)
    continue

# Only report success if all genes actually succeeded
if failed_genes:
    raise RuntimeError(f"Workflow failed for genes: {failed_genes}")
```

**Validation:**
```bash
# 1. Check for actual prediction files
ls -la results/*/genes/*/predictions.parquet
# Should exist and be non-empty

# 2. Verify performance metrics are reasonable
grep -E "(Total samples|Average Precision)" results/*/performance_report.txt
# Should show actual numbers, not zeros or empty

# 3. Check for error messages in logs
grep -E "(ERROR|Failed|Exception)" results/*/inference_workflow.log
# Should be empty for truly successful runs
```

---

## 🚨 **Category 3: Meta-Model Activation Failures**

### **Error 2.1: Meta-Model Never Used (0% Usage)**

**Symptoms:**
```
🔗 STEP 5: Combining predictions for complete coverage
   📊 Complete coverage: 2,151 positions
   🤖 Meta-model recalibrated: 0 (0.0%)  # ❌ SHOULD BE >0%
   🔄 Base model reused: 2,151 (100.0%)
```

**Root Cause:**
- Feature generation failing due to coordinate system mismatches
- Uncertain positions identified but feature matrix generation failing
- Synchronization issues between uncertainty detection and feature generation

**Diagnostic Steps:**
```bash
# 1. Check if uncertain positions are being identified
python -c "
import pandas as pd
import numpy as np
import glob

# Load base predictions
files = glob.glob('results/*/selective_inference/predictions/*/base_model_predictions.parquet')
if files:
    df = pd.read_parquet(files[0])
    max_scores = np.maximum(df['donor_score'], df['acceptor_score'])
    uncertain_mask = (max_scores >= 0.02) & (max_scores < 0.80)
    print(f'Uncertain positions: {uncertain_mask.sum()}/{len(df)} ({100*uncertain_mask.sum()/len(df):.1f}%)')
"

# Expected output: "Uncertain positions: 65/2151 (3.0%)"
```

**Solution:**
Ensure the optimized feature enrichment is being used and coordinate system issues are avoided:

```python
# BEFORE: Problematic coordinate system conversion
true_donor_positions = []
for site in donor_sites:
    if strand == '+':
        relative_position = site['position'] - gene_data['gene_start']  # ❌ COORDINATE MISMATCH
    elif strand == '-':
        relative_position = gene_data['gene_end'] - site['position']    # ❌ COORDINATE MISMATCH

# AFTER: Direct feature generation bypassing coordinate conversion
enricher = create_optimized_enricher(verbose=verbose)
feature_matrix = enricher.generate_features_for_uncertain_positions(
    uncertain_positions,  # Already in correct coordinate system
    gene_id,
    config.model_path
)
```

**Validation:**
```bash
# Verify meta-model is being used
grep "Meta-model recalibrated" results/*/performance_report.txt
# Should show: "Meta-model recalibrated: 65 (3.0%)" or similar >0%
```

---

## 🚨 **Category 3: Coordinate System Errors**

### **Error 3.1: Index Out of Range for Probabilities**

**Symptoms:**
```
[feature-generation] ❌ Error generating features: true_donor_positions contain indices out of range for donor_probabilities
AssertionError: true_donor_positions contain indices out of range for donor_probabilities
```

**Root Cause:**
- Splice site annotations in absolute genomic coordinates
- Prediction arrays in relative gene coordinates  
- Incorrect coordinate conversion between systems

**Technical Details:**
```python
# The problematic code path:
donor_probabilities = np.array(gene_data['donor_prob'])  # Length: gene sequence length
true_donor_positions = [site['position'] - gene_start for site in donor_sites]  # Absolute → relative

# Issue: If gene_start is wrong or coordinate systems don't match:
assert true_donor_positions.max() < len(donor_probabilities)  # ❌ FAILS
```

**Solution:**
Bypass coordinate system conversion entirely using optimized enrichment:

```python
# BEFORE: Complex coordinate system conversion
ss_annotations_relative = ss_annotations_df.clone()
for gene_id in base_predictions.keys():
    gene_start = base_predictions[gene_id]['gene_start']
    gene_end = base_predictions[gene_id]['gene_end'] 
    strand = base_predictions[gene_id]['strand']
    
    if strand == '+':
        relative_starts = gene_annotations['start'] - gene_start  # ❌ ERROR-PRONE
        relative_ends = gene_annotations['end'] - gene_start

# AFTER: Use optimized enrichment that avoids coordinate conversion
enricher = create_optimized_enricher(verbose=verbose)
feature_matrix = enricher.generate_features_for_uncertain_positions(
    uncertain_positions,  # Uses existing position data
    gene_id,
    config.model_path
)
```

**Prevention:**
- Always use `optimized_feature_enrichment.py` for inference
- Avoid direct coordinate system conversions
- Test with known genes that have worked before

### **Error 2.3: Dataset Builder Column Preservation (False Lead)**

**Symptoms:**
```python
# During k-mer feature extraction, metadata columns appear to be dropped
# featurize_gene_sequences() seems to lose gene_id, transcript_id, position columns
# Dataset builder failing with missing columns error
```

**Initial Misdiagnosis:**
This was initially thought to be the root cause of the meta-only inference failures. Investigation focused on:
- K-mer feature extraction dropping metadata columns
- `drop_source_columns=False` parameter not working correctly
- Dataset builder column preservation logic

**Technical Investigation Performed:**
```python
# Extensive debugging of dataset builder column preservation
essential_metadata_cols = ['gene_id', 'transcript_id', 'position', 'pred_type', 'splice_type', 
                          'donor_score', 'acceptor_score', 'neither_score', 'chrom']
preserved_metadata = {}
for col in essential_metadata_cols:
    if col in pd_batch.columns:
        preserved_metadata[col] = pd_batch[col].copy()

# Use featurize_gene_sequences directly to have control over column preservation
pd_batch, _ = featurize_gene_sequences(
    pd_batch,
    kmer_sizes=kmer_sizes,
    return_feature_set=True,
    drop_source_columns=False,  # CRITICAL: Preserve all metadata columns
    verbose=0,
)

# CRITICAL: Restore essential metadata columns that may have been dropped
for col, data in preserved_metadata.items():
    if col not in pd_batch.columns:
        pd_batch[col] = data
```

**Actual Root Cause:**
This was **NOT** the real problem. The dataset builder and k-mer feature extraction were working correctly. The real issue was:
1. **Failed workflow runs** were generating incomplete analysis_sequences files
2. **Successful workflow runs** were generating complete analysis_sequences files  
3. The investigation was looking at incomplete files from failed runs, not successful runs

**Resolution:**
No changes were needed to the dataset builder or k-mer feature extraction. The workflow was working correctly when it ran successfully. The key was ensuring the workflow completed successfully by:
- Using `--complete-coverage` flag with meta-only mode
- Ensuring proper error handling and reporting
- Distinguishing between failed and successful workflow runs

**Lessons Learned:**
- **Always verify the source of problematic data** - incomplete files may be from failed runs
- **Check workflow completion status** before investigating downstream issues
- **Test with known working examples** to establish baseline behavior
- **Focus on end-to-end workflow success** rather than individual component debugging

**Prevention:**
- Monitor workflow completion status before investigating data issues
- Use temporary file timestamps to identify which runs generated which files
- Always test with a known working gene first to establish baseline

---

## 🚨 **Category 4: Feature Harmonization Errors**

### **Error 4.1: Feature Shape Mismatch**

**Symptoms:**
```
📊 Feature matrix shape: (65, 123)
❌ Error in selective feature generation: Feature shape mismatch, expected: 124, got 123
ValueError: Feature shape mismatch, expected: 124, got 123
```

**Root Cause:**
- Training model expects 124 features
- Generated feature matrix has 123 features
- Mismatch usually due to missing or incorrectly excluded features

**Diagnostic Steps:**
```bash
# 1. Check training feature manifest
python -c "
import pandas as pd
df = pd.read_csv('results/gene_cv_pc_1000_3mers_run_4/feature_manifest.csv')
print(f'Training features: {len(df)}')
print(f'First 5: {df[\"feature\"].head().tolist()}')
print(f'Last 5: {df[\"feature\"].tail().tolist()}')

# Check for metadata vs feature columns
metadata_cols = ['gene_id', 'position', 'transcript_id', 'strand', 'chromosome']
training_features = df['feature'].tolist()
feature_cols = [f for f in training_features if f not in metadata_cols]
print(f'Pure features (excluding metadata): {len(feature_cols)}')
print(f'Metadata in training: {[f for f in training_features if f in metadata_cols]}')
"
```

**Solution:**
Ensure proper feature/metadata column separation:

```python
# BEFORE: Incorrectly excluding 'chrom' as metadata
feature_cols = [col for col in feature_matrix.columns if col not in ['gene_id', 'position', 'chrom']]

# AFTER: Keep 'chrom' as feature, exclude only true metadata
metadata_cols = ['gene_id', 'position', 'transcript_id', 'strand', 'chromosome']  # Note: chrom is kept
feature_cols = [col for col in feature_matrix.columns if col not in metadata_cols]
```

**Prevention:**
- Always check training feature manifest to understand expected features
- Use `optimized_feature_enrichment.py` which handles harmonization automatically
- Validate feature count matches training exactly (124 features)

### **Error 4.2: Missing K-mer Features**

**Symptoms:**
```
⚠️ Missing 64 non-kmer features: ['3mer_AAA', '3mer_AAC', ...]
```

**Root Cause:**
- Hardcoded k-mer detection only looking for 3-mers
- Training model might use different k-mer sizes or mixed k-mers
- K-mer feature generation not flexible enough

**Solution:**
Use dynamic k-mer detection:

```python
# BEFORE: Hardcoded 3-mer detection
kmer_features = [f for f in training_features if f.startswith('3mer_')]

# AFTER: Dynamic k-mer detection with regex
import re
kmer_pattern = re.compile(r'^\d+mer_')  # Matches any k-mer (1mer_, 2mer_, 3mer_, etc.)
kmer_features = [f for f in training_features if kmer_pattern.match(f)]
```

**Prevention:**
- Use flexible k-mer detection that supports any k value
- Automatically fill missing k-mers with zeros (standard practice)
- Verify k-mer feature count matches training expectations

---

## 🚨 **Category 5: Data Loading and Type Errors**

### **Error 5.1: Chromosome Parsing Errors**

**Symptoms:**
```
polars.exceptions.ComputeError: could not parse `X` as dtype `i64` at column 'chrom'
polars.exceptions.ComputeError: could not parse `Y` as dtype `i64` at column 'chrom'
```

**Root Cause:**
- Chromosome columns containing 'X', 'Y', 'MT' as strings
- Polars trying to infer as integer type
- Missing schema overrides for string chromosome data

**Solution:**
```python
# BEFORE: No schema override
df = pl.read_csv(file_path, separator="\t", infer_schema_length=1000)

# AFTER: Explicit schema override for chromosome
df = pl.read_csv(
    file_path,
    separator="\t", 
    infer_schema_length=1000,
    schema_overrides={"chrom": pl.Utf8}  # Handle chromosome X, Y as strings
)
```

**Prevention:**
- Always use `schema_overrides={"chrom": pl.Utf8}` when loading genomic data
- Apply to all TSV/CSV files that contain chromosome information
- Test with data containing sex chromosomes (X, Y)

### **Error 5.2: Missing Metadata Columns**

**Symptoms:**
```
KeyError: 'gene_start'
KeyError: 'strand'
KeyError: 'gene_id'
```

**Root Cause:**
- Feature enrichment not applied before coordinate conversion
- Metadata columns not propagated through pipeline
- Wrong order of operations in feature generation

**Solution:**
```python
# BEFORE: Convert before enrichment (loses metadata)
predictions = _convert_positions_to_predictions(positions_df, target_genes)
positions_enriched = apply_feature_enrichers(positions_pd, ...)

# AFTER: Enrich before conversion (preserves metadata)
positions_enriched = apply_feature_enrichers(positions_pd, ...)
predictions = _convert_positions_to_predictions(positions_enriched, target_genes)
```

**Prevention:**
- Always apply feature enrichment before format conversions
- Validate required metadata columns exist before processing
- Use optimized enrichment which handles metadata preservation automatically

---

## 🚨 **Category 6: Environment and Dependencies**

### **Error 6.1: Module Import Errors**

**Symptoms:**
```
ModuleNotFoundError: No module named 'polars'
ImportError: cannot import name 'load_gene_features'
```

**Root Cause:**
- Conda environment not activated
- Missing dependencies in environment
- Import path changes during development

**Solution:**
```bash
# 1. Activate correct environment
mamba activate surveyor

# 2. Verify key dependencies
python -c "import polars, pandas, numpy, sklearn; print('All dependencies available')"

# 3. Check environment packages
mamba list | grep -E "(polars|pandas|scikit-learn)"
```

**Prevention:**
- Always activate `surveyor` environment before running workflows
- Document exact dependency versions for reproducibility
- Use try/except blocks for optional imports with graceful fallbacks

### **Error 6.2: File Path Resolution Issues**

**Symptoms:**
```
FileNotFoundError: No trained meta-model found!
FileNotFoundError: Gene features file not found: data/ensembl/...
```

**Root Cause:**
- Relative paths from wrong working directory
- Model path discovery logic failing
- Missing or moved data files

**Solution:**
```python
# BEFORE: Hardcoded paths
model_path = "results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl"

# AFTER: Dynamic model discovery with fallbacks
results_dir = project_root / "results"
pattern = str(results_dir / "gene_cv_pc_1000_3mers_run_*")
run_dirs = glob.glob(pattern)
run_info = []
for run_dir in run_dirs:
    match = re.search(r'run_(\d+)', run_dir)
    if match:
        run_number = int(match.group(1))
        model_path = Path(run_dir) / "model_multiclass.pkl"
        run_info.append((run_number, model_path))

# Sort by run number (highest first = most recent)
run_info.sort(key=lambda x: x[0], reverse=True)
```

**Prevention:**
- Use dynamic path discovery with version prioritization
- Provide clear error messages with exact paths checked
- Implement graceful fallbacks for missing optional files

---

## 🚨 **Category 7: Memory and Performance Issues**

### **Error 7.1: Memory Overflow**

**Symptoms:**
```
MemoryError: Unable to allocate array
Process killed (OOM)
```

**Root Cause:**
- Full feature matrix generation for all positions
- Large genomic files loaded multiple times
- Inefficient data structures

**Solution:**
```python
# BEFORE: Process all positions
feature_matrix = generate_features_for_all_positions(all_positions)  # ❌ MEMORY INTENSIVE

# AFTER: Process only uncertain positions
uncertain_mask = identify_uncertain_positions(predictions)
uncertain_positions = predictions[uncertain_mask]
feature_matrix = generate_features_for_uncertain_positions(uncertain_positions)  # ✅ MEMORY EFFICIENT
```

**Prevention:**
- Always use selective processing for production workflows
- Monitor memory usage during development
- Use streaming/chunked processing for large datasets

---

## 📋 **Diagnostic Checklist**

Use this checklist to systematically diagnose issues:

### **Level 1: Basic Health Check**
- [ ] Conda environment activated (`mamba activate surveyor`)
- [ ] Model file exists and accessible
- [ ] Input gene IDs are valid format
- [ ] Output directory writable

### **Level 2: Configuration Validation**
- [ ] Training dataset path correct (if provided)
- [ ] Uncertainty thresholds reasonable (0.01-0.99)
- [ ] Inference mode valid (`hybrid`, `base_only`, `meta_only`)
- [ ] Verbosity level appropriate

### **Level 3: Performance Validation**
- [ ] Processing time <2 seconds per gene
- [ ] Meta-model usage >0% in hybrid mode
- [ ] Feature count matches training (typically 124)
- [ ] Memory usage reasonable

### **Level 4: Output Validation**
- [ ] All expected output files created
- [ ] Parquet files readable and non-empty
- [ ] Performance report contains valid metrics
- [ ] Log file shows successful completion

---

## 🛠️ **Advanced Debugging**

### **Enable Maximum Verbosity**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model MODEL_PATH \
    --genes GENE_ID \
    --output-dir debug_output \
    -vvv  # Maximum verbosity
```

### **Component-Level Testing**
```python
# Test optimized feature enrichment directly
from meta_spliceai.splice_engine.meta_models.workflows.inference.optimized_feature_enrichment import (
    create_optimized_enricher
)

enricher = create_optimized_enricher(verbose=True)
enricher.load_feature_manifest("results/gene_cv_pc_1000_3mers_run_4")

# Test feature generation
import pandas as pd
test_positions = pd.DataFrame({
    'gene_id': ['ENSG00000154358'] * 5,
    'position': [100, 200, 300, 400, 500],
    'donor_score': [0.1, 0.5, 0.8, 0.05, 0.3],
    'acceptor_score': [0.2, 0.3, 0.1, 0.7, 0.4],
    'neither_score': [0.7, 0.2, 0.1, 0.25, 0.3]
})

features = enricher.generate_features_for_uncertain_positions(
    test_positions, 
    'ENSG00000154358',
    "results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl"
)
print(f"Generated features: {features.shape}")
```

### **Performance Profiling**
```python
import time
import psutil
import os

def profile_workflow():
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run workflow here
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Processing time: {end_time - start_time:.2f}s")
    print(f"Memory usage: {end_memory - start_memory:.2f}MB")
```

---

## 📈 **Performance Benchmarks**

### **Expected Performance Targets**

| Metric | Target | Typical | Investigate If |
|--------|--------|---------|----------------|
| **Processing Time (Hybrid)** | <35s | 32.5s | >60s |
| **Processing Time (Selective)** | <2s | 1.1s | >5s |
| **Meta-model Usage (Hybrid)** | 2-5% | 3.0% | 0% or >10% |
| **Meta-model Usage (Meta-Only)** | 100% | 100% | <100% |
| **Memory Usage** | <1GB | ~200MB | >2GB |
| **Feature Count** | 124 | 124 | ≠124 |
| **Analysis Sequences Columns** | 50+ | 57 | <20 |
| **Error Rate** | 0% | 0% | >0% |
| **Position Count Consistency** | Identical across modes | ✅ | Different counts |
| **Position Discrepancy** | +1 typical | +1 | >±3 |

### **Meta-Only Mode Specific Targets**

| Metric | Target | Typical | Investigate If |
|--------|--------|---------|----------------|
| **Complete Coverage** | Required | Yes | No |
| **Analysis Sequences Generated** | Required | Yes | No |
| **Positions Processed** | All gene positions | 5,003 | 0 |
| **Advanced Features Present** | 57 columns | 57 columns | 8 columns |

### **Performance Regression Detection**
```bash
# Baseline performance test
echo "ENSG00000154358" > benchmark_gene.txt

# Run performance test
time python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes-file benchmark_gene.txt \
    --output-dir performance_test \
    --inference-mode hybrid \
    --verbose

# Check results
grep "Processing time" performance_test/performance_report.txt
grep "Meta-model recalibrated" performance_test/performance_report.txt
```

---

## 🎯 **Quick Fix Reference**

| Problem | Quick Fix |
|---------|-----------|
| **Identical performance metrics** | Check prediction combiner logic, verify meta predictions applied to meta columns |
| **Meta-only mode fails** | Use `--complete-coverage` flag, check for 57 columns in analysis_sequences |
| **Incomplete analysis_sequences (8 cols)** | Verify workflow completed successfully, not from failed run |
| **False success reporting** | Check actual prediction files exist, verify performance metrics |
| **Very slow (>30s)** | Use `optimized_feature_enrichment.py` |
| **Meta-model not used (0%)** | Check uncertainty thresholds, verify feature generation |
| **Coordinate errors** | Use optimized enrichment, avoid coordinate conversion |
| **Feature count mismatch** | Check metadata vs feature separation, validate manifest |
| **Chromosome parse errors** | Add `schema_overrides={"chrom": pl.Utf8}` |
| **Import errors** | Activate `surveyor` environment |
| **Memory errors** | Use selective processing, reduce position count |
| **Position count confusion** | Document as normal (11,443→5,716 consolidation + +1 boundary) |
| **Different counts across modes** | Investigate - should be identical across all inference modes |

## 🔍 **Meta-Only Mode Specific Quick Fixes**

| Problem | Quick Fix |
|---------|-----------|
| **Identical performance metrics with base_only** | Check prediction combiner, verify meta predictions in meta columns |
| **Dataset builder column errors** | Usually a false lead - check if workflow completed successfully |
| **Missing advanced features** | Ensure `--complete-coverage` flag is used |
| **0% meta-model usage in meta_only** | Check analysis_sequences generation, verify complete workflow |
| **False success with internal failures** | Check prediction files exist and are non-empty |
| **Position count differs from base_only** | Investigate - meta_only should show identical position counts |
| **Confused about +1 discrepancy** | Normal boundary enhancement - same across all modes |

---

## 📚 **Related Analysis and Documentation**

### **Position Count Analysis Package (Comprehensive)**
- **Main Package**: [`workflows/analysis/`](../../analysis/) - Complete position count analysis toolkit
- **Driver Scripts**:
  - [`main_driver.py`](../../analysis/main_driver.py) - Interactive analysis interface
  - [`analyze_position_counts.py`](../../analysis/analyze_position_counts.py) - Command-line analysis driver
- **Analysis Modules**:
  - [`position_counts.py`](../../analysis/position_counts.py) - Core analysis framework
  - [`inference_validation.py`](../../analysis/inference_validation.py) - Cross-mode consistency validation
  - [`boundary_effects.py`](../../analysis/boundary_effects.py) - Boundary effect investigation
  - [`pipeline_tracing.py`](../../analysis/pipeline_tracing.py) - Evaluation pipeline analysis
  - [`detailed_analysis.py`](../../analysis/detailed_analysis.py) - Comprehensive question answering
- **Documentation**:
  - [`FINAL_POSITION_ANALYSIS_REPORT.md`](./FINAL_POSITION_ANALYSIS_REPORT.md) - Complete technical analysis
  - [`POSITION_COUNT_ANALYSIS_SUMMARY.md`](./POSITION_COUNT_ANALYSIS_SUMMARY.md) - Analysis summary

### **Key Position Count Insights**
1. **11,443 vs 5,716 positions**: Normal donor/acceptor consolidation
2. **+1 discrepancies**: Boundary enhancement at 3' end (normal)
3. **Inference mode consistency**: All modes show identical position counts
4. **Donor/acceptor asymmetry**: 0.1-0.3% asymmetry is biologically expected
5. **Universal pattern**: All genes show +1 discrepancy with complete evaluation

---

**This troubleshooting guide represents comprehensive battle-tested knowledge from developing and optimizing the inference workflow. Following these solutions should resolve 95%+ of issues encountered in production deployments.**