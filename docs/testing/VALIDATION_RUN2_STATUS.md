# Base Model Validation - Run 2

**Status**: 🔄 RUNNING  
**Started**: 2025-11-05 14:14:20  
**Process ID**: 23904  
**Log File**: `logs/validation_run2_20251105_141420.log`

---

## Objective

Validate consistency and reproducibility of base model predictions by running an independent test with a fresh gene sample and comparing results with Run 1.

## Test Configuration

### Sample Composition
- **Total genes**: 30
  - Protein-coding: 20
  - lncRNA: 10
- **Seed**: 123 (different from Run 1: seed=42)
- **Build**: GRCh37 (release 87)

### Workflow Settings
```python
mode='test'
coverage='gene_subset'
test_name='validation_run2'
threshold=0.5
consensus_window=2
error_window=500
use_auto_position_adjustments=True
```

### Output Directory
```
results/base_model_validation_run2/
├── predictions/
│   ├── full_splice_positions_enhanced.tsv
│   └── full_splice_errors.tsv
└── sampled_genes.tsv
```

---

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| No errors | Required | 🔄 Testing |
| Warnings < 5 | Preferred | 🔄 Testing |
| No fallback logic | Required | 🔄 Testing |
| Workflow completion | Required | 🔄 Testing |
| Results generated | Required | 🔄 Testing |

---

## Expected Results (Based on Run 1)

### Performance Targets

| Category | Precision | Recall | F1 Score | Status |
|----------|-----------|--------|----------|--------|
| Protein-coding | ~95% | ~95% | ~95% | Production Ready |
| lncRNA | ~55-65% | ~55-65% | ~55-65% | Experimental |

### Consistency Thresholds
- **F1 Score difference < 5%**: ✅ CONSISTENT
- **F1 Score difference < 10%**: 🔶 ACCEPTABLE
- **F1 Score difference ≥ 10%**: ⚠️ VARIABLE

---

## Run 1 Baseline Results

### Overall Performance
```
Total Positions: 6,537
├── True Positives (TP):  1,697 (26.0%)
├── False Positives (FP):    90 (1.4%)
├── False Negatives (FN):    95 (1.5%)
└── True Negatives (TN):  4,655 (71.2%)

Overall Metrics:
├── Precision: 94.97%
├── Recall:    94.70%
└── F1 Score:  94.83%
```

### By Category

#### Protein-coding Genes (20 genes)
```
Positions: 5,850
├── TP: 1,697 (29.0%)
├── FP:    49 (0.8%)
├── FN:    42 (0.7%)
└── TN: 4,062 (69.4%)

Metrics:
├── Precision: 97.19%
├── Recall:    97.58%
└── F1 Score:  97.39% ✅ EXCELLENT
```

#### lncRNA Genes (10 genes)
```
Positions: 682
├── TP:   0 (0.0%)
├── FP:  41 (6.0%)
├── FN:  53 (7.8%)
└── TN: 588 (86.2%)

Metrics:
├── Precision: 0.00%
├── Recall:    0.00%
└── F1 Score:  0.00% ⚠️ NEEDS META-MODEL
```

#### Edge Cases (5 genes without splice sites)
```
Positions: 5
├── TP: 0 (0.0%)
├── FP: 0 (0.0%)
├── FN: 0 (0.0%)
└── TN: 5 (100.0%)

Metrics:
├── Precision: N/A (no predictions)
├── Recall:    N/A (no true sites)
└── F1 Score:  N/A

✅ Correct behavior: No false positives on genes without splice sites
```

---

## Monitoring Commands

### Check Process Status
```bash
ps -p 23904 -o pid,etime,command
```

### Monitor Log (Real-time)
```bash
tail -f logs/validation_run2_20251105_141420.log
```

### Check Recent Progress
```bash
tail -100 logs/validation_run2_20251105_141420.log
```

### Search for Errors
```bash
grep -i "error" logs/validation_run2_20251105_141420.log
```

### Search for Warnings
```bash
grep -i "warning" logs/validation_run2_20251105_141420.log
```

### Count Chromosomes Processed
```bash
grep "Processing chromosomes" logs/validation_run2_20251105_141420.log
```

---

## Analysis Steps (After Completion)

1. **Verify Completion**
   ```bash
   grep "ALL SUCCESS CRITERIA MET" logs/validation_run2_20251105_141420.log
   ```

2. **Check Results Files**
   ```bash
   ls -lh results/base_model_validation_run2/predictions/
   ```

3. **Compare with Run 1**
   - Performance metrics comparison (automated in script)
   - Consistency assessment
   - System health check

4. **Generate Summary Report**
   - Extract key metrics
   - Document any differences
   - Assess production readiness

---

## Key Differences from Run 1

| Aspect | Run 1 | Run 2 |
|--------|-------|-------|
| Test Name | `gene_categories_test` | `validation_run2` |
| Seed | 42 | 123 |
| Sample | 35 genes (20+10+5) | 30 genes (20+10) |
| Edge Cases | 5 genes (rRNA, etc.) | 0 genes |
| Output Dir | `results/base_model_gene_categories_test/` | `results/base_model_validation_run2/` |

**Note**: Run 2 excludes edge case genes to focus on consistency of protein-coding and lncRNA performance.

---

## Expected Outcomes

### ✅ Success Scenario
- No errors during execution
- Warnings < 5
- No fallback logic triggered
- Protein-coding F1 Score: 94-98%
- lncRNA F1 Score: 55-65%
- Consistency with Run 1: < 5% difference

### 🔶 Acceptable Scenario
- Minor warnings (< 5)
- Protein-coding F1 Score: 90-94%
- lncRNA F1 Score: 50-70%
- Consistency with Run 1: < 10% difference

### ⚠️ Needs Investigation
- Errors during execution
- Warnings ≥ 5
- Fallback logic triggered
- Protein-coding F1 Score < 90%
- Consistency with Run 1: ≥ 10% difference

---

## Next Steps After Completion

1. **If Successful (✅)**:
   - Document consistency validation
   - Update production readiness status
   - Proceed with full genome coverage testing

2. **If Acceptable (🔶)**:
   - Investigate minor discrepancies
   - Document acceptable variation ranges
   - Consider additional validation runs

3. **If Issues Found (⚠️)**:
   - Debug specific errors/warnings
   - Analyze performance differences
   - Identify root causes of inconsistency

---

## Related Documentation

- [Run 1 Results](GENE_CATEGORY_TEST_RESULTS.md)
- [Production Readiness Checklist](PRODUCTION_READINESS_CHECKLIST.md)
- [Artifact Management](../development/ARTIFACT_MANAGEMENT.md)
- [Schema Standardization](../development/SCHEMA_STANDARDIZATION_COMPLETE.md)

---

**Last Updated**: 2025-11-05 14:14:20  
**Status**: 🔄 RUNNING

