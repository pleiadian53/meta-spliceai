# Score-Shifting Coordinate Adjustment Implementation

**Date**: 2025-10-30  
**Status**: ✅ **IMPLEMENTED**  

---

## Overview

We've implemented a **score-shifting** approach for coordinate adjustments that maintains 100% coverage by shifting SCORE VECTORS instead of POSITION COORDINATES.

---

## The Problem with Position-Shifting (Old Approach)

### What It Did
```python
# OLD: Shift position coordinates based on predicted splice type
if predicted_type == 'donor' and strand == '+':
    adjusted_position = position - 2
elif predicted_type == 'acceptor' and strand == '+':
    adjusted_position = position - 0
```

### Issues
1. **Position Collisions**: Multiple positions map to same coordinate
   ```
   Position 103 (donor) → 103 - 2 = 101
   Position 101 (acceptor) → 101 - 0 = 101
   → COLLISION at position 101!
   ```

2. **Coverage Loss**: 7,107 positions → 6,300 unique (88.6% coverage)

3. **Data Loss**: Had to average scores to resolve collisions

---

## The Solution: Score-Shifting (New Approach)

### What It Does
```python
# NEW: Shift score vectors, keep positions unchanged
if strand == '+':
    donor_scores = raw_donor_scores.shift(-2)      # Get scores from position+2
    acceptor_scores = raw_acceptor_scores.shift(0)  # Get scores from position+0
```

### Benefits
1. **No Position Collisions**: All N positions preserved for N-bp gene
2. **100% Coverage**: Every nucleotide has a score
3. **No Data Loss**: No averaging needed
4. **Correct Alignment**: Scores still align with GTF annotations

---

## Technical Details

### Conceptual Model

Think of it as having **different views** of the score vectors:

```
Original raw scores (from SpliceAI):
Position:     98    99   100   101   102   103
Donor:       0.2   0.3   0.9   0.1   0.2   0.8
Acceptor:    0.7   0.6   0.1   0.8   0.7   0.2

After adjustment (+ strand, donor_adj=+2, acceptor_adj=0):
Position:     98    99   100   101   102   103
Donor:       0.9   0.1   0.2   0.8   0.0   0.0  ← shifted from positions +2
Acceptor:    0.7   0.6   0.1   0.8   0.7   0.2  ← no shift
```

**Key Insight**: Position 98 gets donor score from position 100, but **keeps coordinate 98**!

### Why This Works

SpliceAI predicts at intronic positions:
- Donor predictions are ~2 bp upstream of actual splice site (+ strand)
- Acceptor predictions are at the exact splice site (+ strand)

By shifting the score vectors, we're saying:
- "The donor score at position 100 actually belongs to position 98"
- "The acceptor score at position 98 stays at position 98"

This aligns the scores with GTF annotation coordinates without moving positions.

---

## Implementation

### File Modified
`meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`

### Changes Made

#### 1. Backed Up Old Implementation
```python
def _apply_coordinate_adjustments_v0(self, predictions_df):
    """OLD VERSION: Shifts positions (creates collisions)."""
    # ... original position-shifting code ...
```

#### 2. New Implementation
```python
def _apply_coordinate_adjustments(self, predictions_df):
    """NEW VERSION: Shifts scores (maintains coverage)."""
    
    # Get adjustment values
    donor_plus_adj = self.adjustment_dict['donor']['plus']      # e.g., +2
    acceptor_plus_adj = self.adjustment_dict['acceptor']['plus']  # e.g., 0
    
    if strand == '+':
        predictions_df = predictions_df.with_columns([
            # Shift donor scores by +2: position 100 gets score from 102
            pl.col('donor_prob').shift(-donor_plus_adj).fill_null(0),
            
            # No shift for acceptor scores: position 100 gets score from 100
            pl.col('acceptor_prob').shift(-acceptor_plus_adj).fill_null(0),
            
            # Neither scores unchanged
            pl.col('neither_prob')
        ])
    
    return predictions_df  # Positions unchanged!
```

#### 3. Updated Collision Handling
```python
# OLD: Expected collisions, averaged scores
if predictions_df.height > n_positions:
    # Average scores for colliding positions
    predictions_df = predictions_df.group_by(['position']).agg([...])

# NEW: No collisions expected!
if predictions_df.height > n_positions:
    # This should NOT happen!
    self.logger.warning("UNEXPECTED: Position duplicates detected")
elif predictions_df.height == n_positions:
    self.logger.info("✅ Full coverage maintained: All positions unique")
```

#### 4. Updated Coverage Reporting
```python
# OLD: Expected ~88% coverage
if complete_predictions.height != gene_length:
    self.logger.info(f"Position collisions expected...")

# NEW: Expect 100% coverage
if complete_predictions.height == gene_length:
    self.logger.info(f"✅ Full coverage achieved: 100%")
else:
    self.logger.warning(f"⚠️ Coverage issue: {coverage_pct:.1f}%")
```

---

## Polars `.shift()` Semantics

Important to understand:
```python
# shift(-n): Move values DOWN (later → earlier)
# shift(+n): Move values UP (earlier → later)

scores = [0.1, 0.2, 0.3, 0.4, 0.5]

scores.shift(-2)  # [0.3, 0.4, 0.5, null, null]
# Position 0 gets value from position 2

scores.shift(+2)  # [null, null, 0.1, 0.2, 0.3]
# Position 2 gets value from position 0
```

For adjustment +2 (get score from position+2), we use `shift(-2)`.

---

## Testing Strategy

### Test 1: Coverage Verification
```python
# Test genes of different lengths
genes = [
    ('ENSG00000134202', 'GSTM3', 7107 bp),
    ('ENSG00000141510', 'TP53', 25768 bp),
    ('ENSG00000157764', 'BRAF', 205603 bp)
]

# Expected: actual_positions == expected_length for all genes
```

### Test 2: No Collisions
```python
# Verify all positions are unique
assert predictions_df.height == predictions_df['position'].n_unique()
```

### Test 3: Alignment with GTF
```python
# Compare predicted splice sites with GTF annotations
# Expected: F1 scores >= previous approach (ideally improved)
```

### Test 4: Score Continuity
```python
# Verify scores at boundaries are reasonable
# Check that shifted scores make biological sense
```

---

## Expected Results

### Before (Position-Shifting)
```
ENSG00000134202 (7,107 bp):
  Raw predictions: 7,107 positions
  After adjustment: 6,300 positions (88.6% coverage)
  Position collisions: 807
  Status: ❌ Incomplete coverage
```

### After (Score-Shifting)
```
ENSG00000134202 (7,107 bp):
  Raw predictions: 7,107 positions
  After adjustment: 7,107 positions (100% coverage)
  Position collisions: 0
  Status: ✅ Full coverage maintained
```

---

## Edge Cases

### 1. Boundary Positions
Positions near gene start/end may not have scores to shift from:
```python
# Position 0 with donor_adj=+2 needs score from position 2
# Position N-1 with donor_adj=+2 needs score from position N+1 (doesn't exist!)

# Solution: fill_null(0) for missing values
pl.col('donor_prob').shift(-2).fill_null(0)
```

### 2. Negative Strand
Adjustments may be different for negative strand:
```python
if strand == '+':
    donor_adj = +2, acceptor_adj = 0
elif strand == '-':
    donor_adj = +1, acceptor_adj = -1
```

### 3. Different Base Models
OpenSpliceAI may have different adjustments than SpliceAI:
```python
# SpliceAI: donor +2/+1, acceptor 0/-1
# OpenSpliceAI: donor +1, acceptor 0
# Auto-detected from training data
```

---

## Validation Checklist

- [ ] Test on small gene (GSTM3, 7,107 bp)
- [ ] Test on medium gene (TP53, 25,768 bp)
- [ ] Test on large gene (BRAF, 205,603 bp)
- [ ] Verify 100% coverage for all genes
- [ ] Verify no position collisions
- [ ] Compare F1 scores with old approach
- [ ] Test on + and - strand genes
- [ ] Verify alignment with GTF annotations
- [ ] Test with different base models (SpliceAI, OpenSpliceAI)

---

## Next Steps

1. ✅ Implement score-shifting approach
2. 🔄 Run coverage test (in progress)
3. ⏳ Verify alignment with GTF annotations
4. ⏳ Compare performance with old approach
5. ⏳ Update training workflow if needed

---

## References

- Original position-shifting: `_apply_coordinate_adjustments_v0()`
- New score-shifting: `_apply_coordinate_adjustments()`
- Training adjustment detection: `splice_prediction_workflow.py` lines 287-364
- Polars shift documentation: https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.shift.html

---

## Conclusion

The score-shifting approach is the **correct** implementation for maintaining full coverage while aligning predictions with GTF annotations. It treats the coordinate adjustment as a **score transformation** rather than a **position transformation**, which preserves all N positions for an N-bp gene.

This is exactly what the user described: "different views of the score vector depending on the strand and splice type."

