# Base Model Comprehensive Test - Running

## Status: ✅ RUNNING (SCHEMA STANDARDIZATION APPLIED)

**Current Run**:
- **Started**: November 4, 2025, 16:58:17
- **PID**: 17870
- **Log file**: `logs/base_model_comprehensive_20251104_165817.log`

**Previous Runs**:
1. **Run 1** (PID 11579, 16:27:04): ❌ Failed - `ValueError: Splice site annotations must have a 'site_type' column`
   - **Fix**: Applied schema standardization in `enhanced_workflow.py`

**Critical Bug Fixed**: Column mismatch (`site_type` → `splice_type`) - See [CRITICAL_BUG_FIX_SITE_TYPE_COLUMN.md](CRITICAL_BUG_FIX_SITE_TYPE_COLUMN.md)  
**Solution**: Formal schema standardization system - See [SCHEMA_STANDARDIZATION_COMPLETE.md](../development/SCHEMA_STANDARDIZATION_COMPLETE.md)

## Test Configuration

### Genes Sampled
- **15 protein-coding genes** (5kb-500kb size range)
- **5 lncRNA genes** (500bp-500kb size range)
- **Total**: 20 genes from 12 chromosomes

### Chromosomes
1, 2, 5, 6, 7, 10, 11, 12, 14, 16, 19, 20

### Parameters
- **Build**: GRCh37 (Ensembl release 87)
- **Threshold**: 0.5
- **Consensus window**: 2bp (±2bp tolerance for matching)
- **Error window**: 500bp (contextual sequence extraction)
- **Auto adjustments**: Enabled (automatic coordinate alignment detection)

### Data Files Verified
- ✅ GTF: `data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf` (1.1 GB)
- ✅ FASTA: `data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa` (3.0 GB)
- ✅ Gene features: `data/ensembl/GRCh37/gene_features.tsv` (57,905 genes)
- ✅ Splice sites: `data/ensembl/GRCh37/splice_sites_enhanced.tsv` (35,306 genes)

## Monitoring

### Check if still running
```bash
ps aux | grep 17870
```

### View live log output
```bash
tail -f logs/base_model_comprehensive_20251104_165817.log
```

### Check last 50 lines
```bash
tail -50 logs/base_model_comprehensive_20251104_165817.log
```

### Check progress
```bash
grep -E "Processing chromosomes|Completed chromosome|Total genes processed" logs/base_model_comprehensive_20251104_165817.log
```

## Expected Timeline

Based on previous tests with small gene sets:

1. **Annotation extraction**: ~2-3 minutes (all 196K GTF entries)
2. **Gene features derivation**: Automatic (if not already exists)
3. **Splice site loading**: < 1 minute (using existing file)
4. **Sequence extraction**: ~1-2 minutes (20 genes)
5. **Overlapping genes**: Skipped
6. **Chromosome determination**: < 1 second
7. **Model loading**: ~10 seconds (GPU initialization)
8. **Predictions**: ~5-15 minutes (20 genes, depends on gene size)
9. **Evaluation & feature extraction**: ~2-5 minutes

**Total estimated time**: 15-30 minutes

## What's Being Tested

### Workflow Steps
1. ✅ Data preparation (annotations, gene features, splice sites, sequences)
2. ⏳ Chromosome-level processing (currently running)
3. ⏳ Gene-level predictions with SpliceAI
4. ⏳ Enhanced evaluation with all probability scores
5. ⏳ Coordinate alignment analysis
6. ⏳ Error analysis (TP/FP/FN classification)
7. ⏳ Feature extraction for meta-learning

### Key Validations
- **Coordinate alignment**: With consensus_window=2, verify predictions align within ±2bp
- **Biotype performance**: Compare protein-coding vs lncRNA prediction accuracy
- **Auto-adjustment detection**: Verify automatic coordinate adjustment works
- **Full coverage mode**: Ensure workflow handles diverse genes from multiple chromosomes

## Output Files

### Chunk-level files (per chromosome/chunk)
- `predictions/error_analysis_chr{N}_{start}_{end}.parquet`
- `predictions/splice_positions_enhanced_chr{N}_{start}_{end}.parquet`
- `predictions/analysis_sequences_chr{N}_{start}_{end}.parquet`

### Aggregated files
- `predictions/error_analysis_aggregated.parquet`
- `predictions/splice_positions_enhanced_aggregated.parquet`

### Gene list
- `sampled_genes.tsv` (20 genes with biotype labels)

## Next Steps (After Completion)

1. **Verify completion**:
   ```bash
   grep "WORKFLOW COMPLETED SUCCESSFULLY" logs/base_model_comprehensive_20251104_150826.log
   ```

2. **Check for errors**:
   ```bash
   grep -i "error\|failed\|traceback" logs/base_model_comprehensive_20251104_150826.log
   ```

3. **Analyze results**:
   ```bash
   python scripts/testing/analyze_base_model_comprehensive.py
   ```

4. **Review metrics**:
   - Overall prediction accuracy (precision, recall, F1)
   - Coordinate alignment statistics
   - Biotype-specific performance
   - True Positive / False Positive / False Negative distributions

## Previous Issues Resolved

### Issue 1: No genes processed (0 genes)
**Cause**: `gene_features.tsv` missing or incomplete  
**Resolution**: ✅ Workflow now auto-generates `gene_features.tsv` during data preparation

### Issue 2: Only chr21 and chr22 in splice_sites_enhanced.tsv
**Cause**: Initial extraction limited to test chromosomes  
**Resolution**: ✅ Regenerated complete `splice_sites_enhanced.tsv` with all chromosomes (35,306 genes)

### Issue 3: 0 lncRNA genes found
**Cause**: GRCh37 uses 'lincRNA' instead of 'lncRNA' as gene_type  
**Resolution**: ✅ Updated sampling to match 'lncRNA|lincRNA|long_noncoding' (found 5,972 genes)

### Issue 4: Schema errors reading gene_features.tsv
**Cause**: Polars inferring 'X' chromosome as integer  
**Resolution**: ✅ Added `schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}`

### Issue 5: nohup "python: No such file or directory"
**Cause**: Conda environment not activated in nohup context  
**Resolution**: ✅ Use `conda run -n surveyor --no-capture-output python ...`

## Connection to Full Coverage Mode Goal

This test is a critical step toward the full coverage inference workflow from previous sessions:

1. **Base model validation**: Ensure SpliceAI predictions work correctly on GRCh37
2. **Multi-chromosome support**: Validate processing genes from multiple chromosomes
3. **Biotype diversity**: Test on both protein-coding and lncRNA genes
4. **Automatic data preparation**: Verify auto-generation of derived datasets
5. **Coordinate alignment**: Confirm automatic adjustment detection works

Once this test completes successfully, we can confidently:
- ✅ Run inference on large gene sets
- ✅ Process entire chromosomes or genome-wide
- ✅ Handle diverse gene biotypes
- ✅ Trust automatic coordinate alignment
- ✅ Move forward with meta-model training on enriched features

## Troubleshooting

### If test hangs or fails

1. **Check if process is still running**:
   ```bash
   ps aux | grep 1866
   ```

2. **Check for GPU issues**:
   ```bash
   grep -i "gpu\|cuda\|out of memory" logs/base_model_comprehensive_20251104_150826.log
   ```

3. **Check for data issues**:
   ```bash
   grep -i "file not found\|missing\|corrupt" logs/base_model_comprehensive_20251104_150826.log
   ```

4. **Kill and restart if needed**:
   ```bash
   kill 1866
   bash scripts/testing/run_comprehensive_test.sh
   ```

## References

- [Base Model Data Mapping](../base_models/BASE_MODEL_DATA_MAPPING.md)
- [Build-Specific Datasets](../data/BUILD_SPECIFIC_DATASETS.md)
- [Automatic Gene Features Generation](../development/AUTOMATIC_GENE_FEATURES_GENERATION.md)
- [Splice Prediction Workflow](../../meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py)

