# Comprehensive Multi-Gene Test Plan

**Date**: November 2, 2025  
**Status**: 🔄 Running  
**Purpose**: Test SpliceAI on diverse genes (protein-coding + lncRNA)

---

## Test Configuration

### Gene Selection

- **15 protein-coding genes** (randomly sampled, seed=42)
- **5 lncRNA genes** (randomly sampled, seed=42)
- **Total: 20 genes**

### Processing Strategy

**Key improvement from previous test**:
- Process **ALL chromosomes** where target genes are located
- Previous test only processed chr21 → only 1 gene analyzed
- This test will process all relevant chromosomes → all 20 genes analyzed

### Evaluation Parameters

```python
threshold = 0.5
consensus_window = 2  # Standard ±2bp tolerance
error_window = 500    # For sequence extraction
use_auto_position_adjustments = True  # Enable adjustment detection
```

---

## Questions to Answer

### 1. Is the +2bp Donor Offset Systematic?

**AGPAT3 showed**:
- Donors: -2bp offset (100% of positions)
- Acceptors: 0bp offset (100% of positions)

**Question**: Is this pattern common across genes?

**Expected outcomes**:
- **Systematic**: Most genes show -2bp donor offset
- **Gene-specific**: Only some genes have this offset
- **Random**: Offsets vary randomly by gene

### 2. How Well Does SpliceAI Perform on lncRNA Genes?

**Hypothesis**: Lower performance than protein-coding genes

**Reasons**:
- SpliceAI trained primarily on protein-coding genes
- lncRNAs have different splicing patterns
- Splice sites may be less conserved

**Metrics to compare**:
- F1 scores (protein-coding vs lncRNA)
- Precision and recall
- Coordinate alignment (exact matches)

### 3. Are There Biotype-Specific Adjustment Patterns?

**Questions**:
- Do protein-coding genes need different adjustments than lncRNAs?
- Are adjustment patterns consistent within biotypes?
- Do different biotypes have different offset distributions?

### 4. What's the Performance Distribution?

**Metrics**:
- Mean, std, min, max F1 scores
- Number of genes with F1 ≥ 0.8
- Number of genes with F1 < 0.5
- Outliers and their characteristics

---

## Test Implementation

### Script 1: test_multi_gene_comprehensive.py

**Purpose**: Run the splice prediction workflow

**Key features**:
1. Sample genes from GTF dynamically
2. Identify all chromosomes where genes are located
3. Process all relevant chromosomes
4. Enable automatic adjustment detection
5. Save gene info for analysis

**Output**:
- `full_splice_positions_enhanced.tsv`
- `full_splice_errors.tsv`
- `gene_info.json` (gene metadata)
- Per-chromosome chunk files

### Script 2: analyze_multi_gene_results.py

**Purpose**: Analyze results comprehensively

**Analyses**:
1. **By biotype**: Aggregate metrics for protein-coding vs lncRNA
2. **By gene**: Individual gene performance
3. **Coordinate adjustments**: Offset distribution by gene and splice type
4. **Summary statistics**: Mean, std, min, max across genes

**Output**: Comprehensive report with tables and statistics

---

## Expected Results

### Protein-Coding Genes

**Based on previous tests**:
- Average F1: 0.89-0.93
- Precision: ~1.0 (very few FPs)
- Recall: 0.79-0.82

**Coordinate alignment**:
- Donors: Likely need +2bp adjustment
- Acceptors: Likely perfect alignment

### lncRNA Genes

**Unknown - exploratory!**

**Possible scenarios**:

**Scenario A: Good performance (F1 ≥ 0.7)**
- SpliceAI generalizes well
- lncRNA splicing similar to protein-coding

**Scenario B: Moderate performance (F1 = 0.5-0.7)**
- Some generalization
- More false negatives (missed sites)

**Scenario C: Poor performance (F1 < 0.5)**
- Limited generalization
- lncRNA splicing very different
- May need separate model

---

## Analysis Plan

### Phase 1: Overall Metrics

**By biotype**:
```
Protein-coding (15 genes):
  Donor F1:    X.XXXX
  Acceptor F1: X.XXXX
  
lncRNA (5 genes):
  Donor F1:    X.XXXX
  Acceptor F1: X.XXXX
```

**Comparison**:
- ΔF1 = F1(protein-coding) - F1(lncRNA)
- Statistical significance?

### Phase 2: Gene-Level Analysis

**Table format**:
```
Gene      Biotype         Chr  Donor F1  Acc F1  D Exact%  A Exact%  D Off  A Off
────────────────────────────────────────────────────────────────────────────────
GENE1     protein_coding  1    0.9123    0.8856  0.0       100.0     -2     0
GENE2     protein_coding  2    0.8945    0.9012  0.0       100.0     -2     0
...
LNCRNA1   lncRNA          5    0.6234    0.5891  50.0      75.0      -1     0
LNCRNA2   lncRNA          7    0.7123    0.6945  25.0      80.0      -2     +1
```

**Insights**:
- Which genes perform best/worst?
- Are there chromosome-specific patterns?
- Do lncRNAs show consistent patterns?

### Phase 3: Adjustment Analysis

**Offset distributions**:
```
Donor offsets:
  -2bp: 12 genes (60%)
  -1bp:  3 genes (15%)
   0bp:  5 genes (25%)

Acceptor offsets:
   0bp: 18 genes (90%)
  +1bp:  2 genes (10%)
```

**Questions**:
- Is -2bp donor offset the most common?
- Are acceptors generally well-aligned?
- Are there biotype-specific patterns?

### Phase 4: Statistical Summary

**Protein-coding**:
```
Donor F1:    Mean=0.89, Std=0.05, Min=0.78, Max=0.95
Acceptor F1: Mean=0.88, Std=0.06, Min=0.75, Max=0.94
```

**lncRNA**:
```
Donor F1:    Mean=0.??, Std=0.??, Min=0.??, Max=0.??
Acceptor F1: Mean=0.??, Std=0.??, Min=0.??, Max=0.??
```

---

## Success Criteria

### Must Pass ✅

1. **All genes processed**: 20/20 genes in results
2. **Workflow completes**: No errors
3. **Protein-coding performance**: Average F1 ≥ 0.80
4. **Data quality**: No missing values, consistent schema

### Nice to Have 🎯

1. **lncRNA performance**: Average F1 ≥ 0.60
2. **Consistent offsets**: Clear pattern emerges
3. **High precision**: ≥0.95 for both biotypes

### Exploratory 🔍

1. **Biotype differences**: How much do they differ?
2. **Gene variability**: How consistent is performance?
3. **Adjustment patterns**: Can we predict needed adjustments?

---

## Comparison with Previous Tests

### AGPAT3 Test (Single Gene)

```
Gene: AGPAT3 (protein-coding, chr21)
Donor F1:    0.9017
Acceptor F1: 0.8824
Donor offset: -2bp (100%)
Acc offset:   0bp (100%)
```

**Limitation**: Only 1 gene, single chromosome

### Previous 50-Gene Test

```
Genes: 50 protein-coding
Average F1: 0.9312
consensus_window: 2
```

**Limitation**: 
- Only protein-coding genes
- No lncRNA comparison
- No gene-level analysis

### This Test (20 Diverse Genes)

```
Genes: 15 protein-coding + 5 lncRNA
Chromosomes: Multiple (all where genes are located)
Analysis: By biotype, by gene, by adjustment
```

**Advantages**:
- Biotype comparison
- Gene-level insights
- Adjustment patterns
- More representative sample

---

## Timeline

### Estimated Duration

**Workflow execution**: 10-30 minutes
- Depends on number of chromosomes
- Depends on gene sizes
- Includes prediction + evaluation

**Analysis**: 1-2 minutes
- Load results
- Calculate metrics
- Generate report

**Total**: ~15-35 minutes

### Monitoring

```bash
# Check progress
tail -f logs/multi_gene_test_*.log

# Check if complete
ls -lh data/ensembl/GRCh37/spliceai_eval/meta_models/multi_gene_test/
```

---

## Post-Test Actions

### If Results Show Systematic -2bp Donor Offset

**Action**: Update default adjustments
```python
# In splice_prediction_workflow.py or config
default_adjustments = {
    'donor': {'plus': 2, 'minus': 2},
    'acceptor': {'plus': 0, 'minus': 0}
}
```

### If lncRNA Performance is Poor (F1 < 0.5)

**Actions**:
1. Document the limitation
2. Consider separate lncRNA model
3. Add biotype-specific handling

### If Adjustments Vary by Gene

**Actions**:
1. Build gene-specific adjustment database
2. Implement per-gene adjustment lookup
3. Or use more flexible consensus_window

---

## Documentation

### Files Created

1. `test_multi_gene_comprehensive.py` - Test runner
2. `analyze_multi_gene_results.py` - Results analyzer
3. `MULTI_GENE_TEST_PLAN.md` - This document
4. Results will be in: `data/ensembl/GRCh37/spliceai_eval/meta_models/multi_gene_test/`

### Reports to Generate

1. **Summary report**: Overall metrics by biotype
2. **Gene report**: Individual gene performance
3. **Adjustment report**: Offset patterns and recommendations
4. **Comparison report**: vs previous tests

---

## Conclusion

This comprehensive test will:

1. ✅ **Validate** the multi-build system works correctly
2. ✅ **Compare** protein-coding vs lncRNA performance
3. ✅ **Identify** systematic adjustment patterns
4. ✅ **Provide** gene-level performance insights
5. ✅ **Inform** future model improvements

**Expected outcome**: Clear understanding of SpliceAI's performance across gene biotypes and identification of needed coordinate adjustments.

---

**Date**: November 2, 2025  
**Status**: Test running, awaiting results  
**Next**: Analyze results and generate comprehensive report



