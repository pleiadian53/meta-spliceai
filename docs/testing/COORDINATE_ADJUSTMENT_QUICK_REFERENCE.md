# Coordinate Adjustment Quick Reference

**Last Updated**: 2025-10-31  

---

## TL;DR

✅ **Use Score-Shifting** (new implementation) for inference mode  
⚠️ **Position-Shifting** (old v0) is deprecated but preserved for reference  

---

## Quick Comparison

| Feature | Position-Shifting (v0) | Score-Shifting (NEW) |
|---------|------------------------|----------------------|
| **Coverage** | 83-88% ❌ | 100% ✅ |
| **Collisions** | Yes (many) ❌ | None ✅ |
| **Status** | Deprecated | Recommended |
| **Use Case** | Legacy/Training | Inference/Production |

---

## What Each Does

### Position-Shifting (OLD)
```python
# Moves POSITION COORDINATES
position_100_donor → adjusted_position_98
position_98_acceptor → adjusted_position_98
# Result: COLLISION at position 98!
```

### Score-Shifting (NEW)
```python
# Moves SCORE VALUES
position_98: donor_score = score_from_position_100
position_98: acceptor_score = score_from_position_98
# Result: NO COLLISION, position 98 has one unique score set
```

---

## Code Usage

### Inference (Use Score-Shifting)

```python
from meta_spliceai.splice_engine.meta_models.workflows.inference import EnhancedSelectiveInferenceWorkflow

workflow = EnhancedSelectiveInferenceWorkflow()
predictions = workflow.predict_for_genes(
    gene_ids=['ENSG00000141510'],
    mode='base_only'
)
# Automatically uses _apply_coordinate_adjustments() (score-shifting)
```

### If You Need Old Behavior (Not Recommended)

```python
# Old method is preserved as _apply_coordinate_adjustments_v0()
# But should not be used for new code
predictions_df = workflow._apply_coordinate_adjustments_v0(predictions_df)
```

---

## Test Results

### Coverage Comparison

| Gene | Length | Position-Shift | Score-Shift |
|------|--------|----------------|-------------|
| GSTM3 | 7,107 bp | 6,300 (88.6%) | 7,107 (100%) ✅ |
| BRAF | 205,603 bp | 171,413 (83.4%) | 205,603 (100%) ✅ |
| TP53 | 25,768 bp | 21,646 (84.0%) | 25,768 (100%) ✅ |

### Position Collisions

| Gene | Position-Shift | Score-Shift |
|------|----------------|-------------|
| GSTM3 | 807 collisions | 0 ✅ |
| BRAF | 34,190 collisions | 0 ✅ |
| TP53 | 4,122 collisions | 0 ✅ |

---

## When to Use Each

### Score-Shifting ✅ (Recommended)

Use for:
- ✅ Inference mode (predicting ALL positions)
- ✅ Full coverage requirements
- ✅ Position-level analysis
- ✅ New code/production systems

### Position-Shifting ⚠️ (Deprecated)

Only acceptable for:
- ⚠️ Legacy training pipelines (already implemented)
- ⚠️ Evaluation at known splice sites only
- ⚠️ Historical compatibility

**Do NOT use for new code!**

---

## Key Concepts

### Adjustment Values

| Splice Type | Strand | Adjustment | Meaning |
|-------------|--------|------------|---------|
| Donor | + | +2 | Score at pos+2 belongs to pos |
| Donor | - | +1 | Score at pos+1 belongs to pos |
| Acceptor | + | 0 | Score at pos belongs to pos |
| Acceptor | - | -1 | Score at pos-1 belongs to pos |

### Score-Shifting Logic

```python
# For donor adjustment +2 on + strand:
# Position 98 gets donor score from position 100

donor_scores = raw_donor_scores.shift(-2)
# shift(-2) moves values DOWN: index i gets value from index i+2
```

---

## Troubleshooting

### "Coverage less than 100%"

**Cause**: Using old position-shifting approach  
**Solution**: Verify using `_apply_coordinate_adjustments()` not `_apply_coordinate_adjustments_v0()`

### "Position collisions detected"

**Cause**: Using old position-shifting approach  
**Solution**: Switch to score-shifting implementation

### "Scores at boundaries are zero"

**Cause**: Normal behavior - positions near boundaries lack context  
**Solution**: This is expected and correct (use `fill_null(0)`)

---

## Files Reference

### Implementation
- **New**: `enhanced_selective_inference.py::_apply_coordinate_adjustments()` (line 618)
- **Old**: `enhanced_selective_inference.py::_apply_coordinate_adjustments_v0()` (line 547)

### Documentation
- **Detailed Comparison**: `COORDINATE_ADJUSTMENT_COMPARISON.md`
- **Implementation Guide**: `SCORE_SHIFTING_IMPLEMENTATION.md`
- **Test Results**: `SCORE_SHIFTING_SUCCESS.md`
- **This File**: `COORDINATE_ADJUSTMENT_QUICK_REFERENCE.md`

---

## Decision Tree

```
Need coordinate adjustment?
│
├─ For inference (predict ALL positions)?
│  └─ ✅ Use score-shifting
│     └─ _apply_coordinate_adjustments()
│
├─ For training/evaluation (known sites only)?
│  ├─ New code?
│  │  └─ ✅ Use score-shifting (recommended)
│  └─ Legacy code?
│     └─ ⚠️ Position-shifting acceptable (but consider migrating)
│
└─ Unsure?
   └─ ✅ Default to score-shifting (always correct)
```

---

## FAQ

**Q: Why was position-shifting used originally?**  
A: It was simpler to implement and worked fine for training/evaluation where we only care about annotated sites.

**Q: Will changing to score-shifting affect my trained models?**  
A: No, models are already trained. This only affects how we apply predictions during inference.

**Q: Should I retrain models with score-shifting?**  
A: Not necessary, but would be more principled. Current models work fine.

**Q: What if I see position collisions with score-shifting?**  
A: This should NOT happen. If it does, it's a bug - please investigate.

**Q: Can I use position-shifting for inference?**  
A: Technically yes, but you'll lose 12-17% coverage. Not recommended.

---

## Summary

**Bottom Line**: Use score-shifting for all new code. It's correct, faster, and maintains 100% coverage.

**Migration**: Old position-shifting code is preserved as `_v0()` for reference but should not be used for new implementations.

**Validation**: Score-shifting has been tested and verified to achieve 100% coverage with zero position collisions on genes of all sizes.

