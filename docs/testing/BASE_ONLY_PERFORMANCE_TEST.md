# Base-Only Mode Performance Test

**Date**: October 29, 2025  
**Status**: 🔄 In Progress  
**Objective**: Validate SpliceAI performance on protein-coding genes in base-only mode

---

## Test Design

### Rationale
SpliceAI was trained primarily on **protein-coding genes**, so we expect:
- **High precision** (>90%): Few false positive predictions
- **High recall** (>90%): Most true splice sites detected
- **High F1 score** (>90%): Overall excellent performance

This test validates that:
1. Full coverage implementation is working correctly
2. SpliceAI base model performs as expected
3. Coordinate systems are correctly aligned
4. Performance metrics calculation is accurate

### Test Genes

| Gene ID | Gene Name | Type | Reason |
|---------|-----------|------|--------|
| ENSG00000134202 | GSTM3 | Protein-coding | Previously tested, 7,107 bp, 5 transcripts |
| ENSG00000157764 | BRAF | Protein-coding | Well-studied oncogene, complex splicing |
| ENSG00000141510 | TP53 | Protein-coding | Tumor suppressor, highly conserved |

### Metrics Calculated

For each gene:
- **Donor sites**: TP, FP, FN, Precision, Recall, F1
- **Acceptor sites**: TP, FP, FN, Precision, Recall, F1
- **Overall**: Combined metrics across both site types

**Threshold**: 0.5 (splice site score > 0.5 considered a prediction)

---

## Expected Results

### High Performance Scenario (Best Case)
```
Overall F1: >90%
Precision: >90%
Recall: >90%
```

**Interpretation**: SpliceAI is performing as expected on its training distribution.

### Good Performance Scenario (Acceptable)
```
Overall F1: 70-90%
Precision: 75-95%
Recall: 70-90%
```

**Interpretation**: SpliceAI is performing well but may have:
- Edge effects (positions near gene boundaries)
- Ambiguous splice sites
- Coordinate system misalignments (small offsets)

### Poor Performance Scenario (Problem)
```
Overall F1: <70%
Precision or Recall: <70%
```

**Interpretation**: Indicates a problem:
- Full coverage not implemented correctly
- Coordinate system bugs
- Incorrect metric calculation
- Wrong annotations or predictions

---

## Test Script

**Location**: `scripts/testing/test_base_only_protein_coding.py`

**Features**:
1. Tests multiple protein-coding genes
2. Loads annotated splice sites from `splice_sites_enhanced.tsv`
3. Runs inference in base-only mode
4. Calculates precision, recall, F1 for donor/acceptor/overall
5. Provides summary statistics and pass/fail assessment

**Usage**:
```bash
cd /Users/pleiadian53/work/meta-spliceai
PYTHONPATH=$PWD:$PYTHONPATH conda run -n surveyor python scripts/testing/test_base_only_protein_coding.py
```

---

## Results

### Current Status
🔄 **Test Running** (Started: 2025-10-29 12:30 PM PST)

Output: `/tmp/base_only_test.log`

### Expected Runtime
- ~2-3 minutes per gene (SpliceAI inference)
- ~6-9 minutes total for 3 genes

---

## Key Implementation Notes

### Full Coverage Verified
- ✅ Direct SpliceAI invocation with `output_format='pandas'`
- ✅ One row per nucleotide position (N × 3 matrix)
- ✅ No row multiplication bugs
- ✅ Complete gene sequence coverage

### Coordinate System
All coordinates are **genomic** (absolute positions):
- Predictions: `position` column contains genomic coordinates
- Annotations: `position` column contains genomic coordinates
- No strand-specific adjustments needed (already applied)

### Threshold Sensitivity
Using threshold=0.5:
- **Lower threshold** (0.3): More sensitive, higher recall, more FPs
- **Higher threshold** (0.7): More specific, higher precision, more FNs
- **Current** (0.5): Balanced trade-off

---

## Troubleshooting

### If F1 < 70%

**Check 1: Coordinate Alignment**
```python
# Compare predicted vs annotated positions
pred_donors = predictions.filter(pl.col('donor_score') > 0.5)['position']
annot_donors = annotations.filter(pl.col('site_type') == 'donor')['position']

# Look for offsets
for pred_pos in pred_donors[:10]:
    nearest_annot = min(annot_donors, key=lambda x: abs(x - pred_pos))
    offset = pred_pos - nearest_annot
    print(f"Pred: {pred_pos}, Annot: {nearest_annot}, Offset: {offset}")
```

**Check 2: Coverage Verification**
```python
# Ensure predictions cover the full gene
gene_length = gene_info['end'] - gene_info['start'] + 1
assert predictions.height == gene_length, "Incomplete coverage"
```

**Check 3: Annotation Quality**
```python
# Check if annotations exist for the gene
ss_df = pl.read_csv('data/ensembl/splice_sites_enhanced.tsv', separator='\t')
gene_ss = ss_df.filter(pl.col('gene_id') == gene_id)
print(f"Annotated splice sites: {gene_ss.height}")
```

### If Tests Fail to Run

**Problem**: Missing annotations
**Solution**: Ensure `splice_sites_enhanced.tsv` exists and contains the test genes

**Problem**: Out of memory
**Solution**: Test one gene at a time, clear predictions directory between runs

**Problem**: Config parameter mismatch
**Solution**: Verify `EnhancedSelectiveInferenceConfig` parameters match the dataclass definition

---

## Follow-Up Tests

After base-only mode validation:

1. **Hybrid Mode**: Test selective meta-model application
2. **Meta-Only Mode**: Test full meta-model recalibration
3. **Cross-Mode Comparison**: Verify score differences
4. **LncRNA Genes**: Test on non-coding genes (expect lower performance)
5. **Multi-Gene Batch**: Test on 10+ genes for scalability

---

## Success Criteria

✅ **Pass**:
- Average F1 > 85% across all test genes
- All genes have F1 > 70%
- No coordinate system issues detected
- Full coverage verified (N positions = gene length)

⚠️ **Acceptable**:
- Average F1 = 70-85%
- Minor coordinate offsets (<3 bp)
- Some genes with F1 = 60-70%

❌ **Fail**:
- Average F1 < 70%
- Major coordinate misalignments (>5 bp)
- Coverage gaps or duplications
- Multiple genes with F1 < 50%

---

## Related Documents

- `docs/testing/FULL_COVERAGE_IMPLEMENTATION.md` - Full coverage implementation
- `docs/testing/CRITICAL_BUG_FOUND.md` - Original dimension mismatch bug
- `docs/session_summaries/SESSION_STATUS_2025-10-29_FINAL.md` - Current session status

---

**Last Updated**: 2025-10-29 12:30 PM PST


