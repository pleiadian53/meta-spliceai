# Validation Run 2 - Quick Reference

## Current Status

**Process ID**: 23904  
**Log File**: `logs/validation_run2_20251105_141420.log`  
**Status**: 🔄 RUNNING

---

## Quick Commands

### Check if Running
```bash
ps -p 23904
```

### Monitor Progress (Live)
```bash
tail -f logs/validation_run2_20251105_141420.log
```

### Check Last 100 Lines
```bash
tail -100 logs/validation_run2_20251105_141420.log
```

### Check for Errors
```bash
grep -i "error" logs/validation_run2_20251105_141420.log | tail -20
```

### Check for Warnings
```bash
grep -i "warning" logs/validation_run2_20251105_141420.log | wc -l
```

### See Chromosome Progress
```bash
grep "Processing chromosomes" logs/validation_run2_20251105_141420.log
```

### Check Completion
```bash
grep -E "(SUCCESS CRITERIA|Workflow failed)" logs/validation_run2_20251105_141420.log
```

---

## Test Configuration

- **Genes**: 30 (20 protein-coding, 10 lncRNA)
- **Seed**: 123 (independent from Run 1)
- **Build**: GRCh37
- **Mode**: test
- **Coverage**: gene_subset

---

## Expected Timeline

| Stage | Estimated Time | Status |
|-------|---------------|--------|
| Gene sampling | 1-2 min | ✅ Complete |
| Annotation loading | 2-3 min | 🔄 In Progress |
| Sequence extraction | 5-10 min | ⏳ Pending |
| Predictions | 10-20 min | ⏳ Pending |
| Analysis | 2-5 min | ⏳ Pending |
| **Total** | **20-40 min** | 🔄 Running |

---

## Success Indicators

Look for these in the log:

1. ✅ "Gene features available"
2. ✅ "Standardizing splice site annotations"
3. ✅ "Processing chromosomes"
4. ✅ "Completed chromosome-level prediction"
5. ✅ "saved aggregated positions"
6. ✅ "ALL SUCCESS CRITERIA MET"

---

## Warning Indicators

Acceptable (< 5 warnings):
- Schema standardization messages
- Minor data type conversions
- Empty chunk skips

Not Acceptable:
- File not found errors
- Schema mismatch errors
- Prediction failures

---

## After Completion

### 1. Verify Success
```bash
tail -50 logs/validation_run2_20251105_141420.log | grep "SUCCESS CRITERIA"
```

### 2. Check Output Files
```bash
ls -lh results/base_model_validation_run2/predictions/
```

### 3. View Results Summary
```bash
cat results/base_model_validation_run2/category_performance_summary.tsv
```

### 4. Compare with Run 1
The test script automatically compares results and prints:
- Performance metrics by category
- Consistency assessment
- Success criteria evaluation

---

## Comparison with Run 1

### Run 1 Results (Baseline)

**Protein-coding** (20 genes):
- Precision: 97.19%
- Recall: 97.58%
- F1 Score: **97.39%** ✅

**lncRNA** (10 genes):
- Precision: 0.00%
- Recall: 0.00%
- F1 Score: **0.00%** ⚠️

### Expected Run 2 Results

**Protein-coding**:
- F1 Score: 94-98% (consistent with Run 1)
- Difference: < 5% (CONSISTENT)

**lncRNA**:
- F1 Score: 0-10% (similar to Run 1)
- Difference: < 10% (ACCEPTABLE)

---

## Troubleshooting

### If Process Stops
```bash
# Check if still running
ps -p 23904

# Check last error
tail -100 logs/validation_run2_20251105_141420.log | grep -i error

# Restart if needed
./scripts/testing/run_validation_run2.sh
```

### If Taking Too Long (> 60 min)
```bash
# Check what stage it's at
tail -20 logs/validation_run2_20251105_141420.log

# Check system resources
top -pid 23904
```

### If Errors Found
1. Check the specific error message
2. Verify conda environment is active
3. Check file permissions
4. Review data file availability

---

## Next Steps

### If Test Passes ✅
1. Document consistency validation
2. Update production readiness assessment
3. Proceed with full genome testing

### If Test Fails ❌
1. Analyze specific failure points
2. Compare with Run 1 logs
3. Debug and rerun

---

**Created**: 2025-11-05 14:14:20  
**Last Updated**: 2025-11-05 14:14:20

