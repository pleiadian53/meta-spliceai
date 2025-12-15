# Gene Category Comparison Test

**Date**: November 5, 2025  
**Test Script**: `scripts/testing/test_base_model_gene_categories.py`  
**Status**: Running

## Objective

Comprehensively test the base model (SpliceAI) across different gene categories to:
1. Compare performance on protein-coding vs lncRNA vs edge case genes
2. Validate handling of genes without splice sites
3. Assess coordinate alignment across diverse gene types
4. Provide performance benchmarks by gene category

## Test Design

### Gene Categories

**Category 1: Protein-Coding Genes** (n=20)
- **Expected**: High performance, many splice sites
- **Criteria**: 
  - `gene_type == 'protein_coding'`
  - Length: 5kb - 500kb
  - ≥4 splice sites (at least 2 exons)
- **Sampled**: 20 genes
  - Splice sites: 8-224 per gene (mean: 93.1)

**Category 2: lncRNA Genes** (n=10)
- **Expected**: Variable performance, fewer splice sites
- **Criteria**:
  - `gene_type` in ['lincRNA', 'antisense', 'processed_transcript', 'sense_intronic']
  - Length: 1kb - 200kb
  - ≥2 splice sites (at least 1 intron)
- **Sampled**: 10 genes
  - Splice sites: 2-18 per gene (mean: 6.8)
  - Biotypes: processed_transcript, sense_intronic, antisense, lincRNA

**Category 3: Edge Cases** (n=5)
- **Expected**: Low/no splice sites, low prediction scores
- **Criteria**:
  - `gene_type` in ['tRNA', 'rRNA', 'snoRNA', 'snRNA', 'miRNA', 'misc_RNA']
  - OR: 0 splice sites
  - Length: 50bp - 10kb
- **Sampled**: 5 genes
  - Splice sites: 0 per gene (mean: 0.0)
  - Biotypes: pseudogene, rRNA

### Total Test Set

- **Total genes**: 35
- **Chromosomes**: 17 (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15, 16, 17, 18, 20, X)
- **Genome build**: GRCh37 (release 87)

## Configuration

```python
SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    test_name='gene_categories_test',
    threshold=0.5,
    consensus_window=2,
    error_window=500,
    use_auto_position_adjustments=True,
    do_extract_annotations=True,
    do_extract_splice_sites=False,
    do_extract_sequences=True,
)
```

## Expected Outcomes

### Protein-Coding Genes
- **Precision**: >95% (few false positives)
- **Recall**: >90% (most true splice sites detected)
- **F1 Score**: >92%
- **Avg splice site score**: >0.90

### lncRNA Genes
- **Precision**: 85-95% (more variable)
- **Recall**: 75-90% (some sites may be missed)
- **F1 Score**: 80-92%
- **Avg splice site score**: 0.70-0.90

### Edge Case Genes
- **True splice sites**: 0 (by design)
- **False positives**: Should be minimal
- **Max prediction score**: Should be <0.5 (below threshold)
- **Interpretation**: Model correctly identifies lack of splice sites

## Analysis Plan

### 1. Category-Specific Metrics

For each category, calculate:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
- Average score at true splice sites
- Distribution of prediction scores

### 2. Comparative Analysis

Compare across categories:
- Precision differences
- Recall differences
- F1 score differences
- Score distributions
- Error patterns (FP vs FN)

### 3. Edge Case Validation

For genes without splice sites:
- Verify no true positives (as expected)
- Count false positives (should be minimal)
- Examine score distribution (should be low)
- Validate model behavior on non-spliced genes

## Output Files

### Results Directory
```
results/base_model_gene_categories_test/
├── sampled_genes_by_category.tsv          # Gene list with categories
├── category_performance_summary.tsv        # Performance metrics by category
└── predictions/
    ├── full_splice_positions_enhanced.tsv  # All positions with predictions
    ├── full_splice_errors.tsv              # FP and FN errors
    └── analysis_sequences_*.tsv            # Per-chromosome sequences
```

### Artifacts Directory
```
data/ensembl/GRCh37/spliceai_eval/tests/gene_categories_test/
└── meta_models/predictions/
    └── (same as above - managed by artifact manager)
```

## Monitoring

### Check Progress
```bash
# Monitor log file
tail -f logs/gene_categories_test_*.log

# Check for completion
grep "WORKFLOW COMPLETED" logs/gene_categories_test_*.log

# Check for errors
grep -i "error\|failed" logs/gene_categories_test_*.log
```

### Expected Runtime
- **Protein-coding genes** (20): ~10-15 minutes (depends on gene size)
- **lncRNA genes** (10): ~3-5 minutes
- **Edge case genes** (5): ~1-2 minutes
- **Total estimated**: 15-25 minutes

## Success Criteria

### Must Pass
- [x] All 35 genes sampled successfully
- [ ] Workflow completes without errors
- [ ] All categories have predictions
- [ ] Protein-coding genes: F1 > 90%
- [ ] lncRNA genes: F1 > 75%
- [ ] Edge case genes: FP < 10 per gene

### Should Pass
- [ ] Protein-coding genes: F1 > 95%
- [ ] lncRNA genes: F1 > 85%
- [ ] Edge case genes: FP < 5 per gene
- [ ] No coordinate misalignments
- [ ] Consistent performance across chromosomes

## Interpretation Guide

### High Performance (F1 > 90%)
- Model accurately predicts splice sites
- Good generalization to diverse genes
- Reliable for production use

### Moderate Performance (F1: 75-90%)
- Model works but with some errors
- May need meta-model correction
- Acceptable for research use

### Low Performance (F1 < 75%)
- Model struggles with this gene type
- Investigate error patterns
- May need specialized handling

### Edge Cases (No splice sites)
- **Good**: FP < 5 per gene, max score < 0.3
- **Acceptable**: FP < 10 per gene, max score < 0.5
- **Poor**: FP > 10 per gene, max score > 0.5

## Related Tests

- [Base Model Comprehensive Test](BASE_MODEL_TEST_RUNNING.md) - 20 genes, protein-coding + lncRNA
- [Multi-Gene Test Plan](MULTI_GENE_TEST_PLAN.md) - Detailed test scenarios
- [Artifact Manager Tests](../development/ARTIFACT_MANAGER_TESTED_COMPLETE.md) - System tests

## Next Steps After Completion

1. **Analyze Results**
   ```bash
   # View performance summary
   cat results/base_model_gene_categories_test/category_performance_summary.tsv
   
   # Check for errors
   wc -l results/base_model_gene_categories_test/predictions/full_splice_errors.tsv
   ```

2. **Compare Categories**
   - Plot F1 scores by category
   - Analyze error patterns
   - Identify challenging gene types

3. **Document Findings**
   - Update this document with results
   - Create performance report
   - Identify areas for improvement

4. **Production Readiness Assessment**
   - If all categories pass: System is production-ready
   - If some categories fail: Document limitations and proceed with caution
   - If edge cases fail: Investigate false positive patterns

---

**Test Started**: November 5, 2025  
**Expected Completion**: ~20 minutes  
**Status**: ⏳ Running

