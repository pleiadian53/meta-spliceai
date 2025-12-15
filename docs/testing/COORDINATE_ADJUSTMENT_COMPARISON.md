# Coordinate Adjustment: Position-Shifting vs Score-Shifting

**Date**: 2025-10-31  
**Status**: 📊 **COMPREHENSIVE ANALYSIS**  

---

## Executive Summary

We implemented two approaches for aligning base model predictions with GTF annotations:
1. **Position-Shifting** (v0): Adjusts position coordinates → Creates collisions, loses coverage
2. **Score-Shifting** (new): Adjusts score vectors → Maintains coverage, no collisions

**Verdict**: Score-shifting is the **correct** approach for inference mode where we need predictions for ALL positions.

---

## The Problem: Coordinate Convention Mismatch

### Background

Different splice site prediction models use different coordinate conventions:

**SpliceAI Convention** (intronic positions):
- Donor sites: Predicts at positions ~2 bp upstream (+ strand) or ~1 bp upstream (- strand) of actual junction
- Acceptor sites: Predicts at exact junction (+ strand) or ~1 bp downstream (- strand)

**GTF Annotation Convention** (exact junctions):
- Donor sites: Last base of exon (exact junction)
- Acceptor sites: First base of exon (exact junction)

**Example**:
```
Gene sequence:  ...EXON1...GT|AG...EXON2...
                       ↑     ↑
GTF annotation:        98   102  (exact junctions)
SpliceAI prediction:   100  102  (donor at +2, acceptor at exact)
```

To align predictions with annotations, we need adjustments:
- Donor (+ strand): shift by +2
- Acceptor (+ strand): shift by 0

---

## Approach 1: Position-Shifting (v0) - OLD

### Concept

Adjust the **position coordinates** based on the predicted splice type.

### Implementation

```python
def _apply_coordinate_adjustments_v0(self, predictions_df):
    """Shift POSITION COORDINATES based on predicted splice type."""
    
    # Determine predicted type for each position
    predictions_df = predictions_df.with_columns([
        pl.when(pl.col('donor_prob') > pl.col('acceptor_prob'))
          .then(pl.lit('donor'))
          .otherwise(pl.lit('acceptor'))
          .alias('predicted_type')
    ])
    
    # Adjust positions based on type and strand
    if strand == '+':
        predictions_df = predictions_df.with_columns([
            pl.when(pl.col('predicted_type') == 'donor')
              .then(pl.col('position') - 2)      # Donor: shift left by 2
              .when(pl.col('predicted_type') == 'acceptor')
              .then(pl.col('position') - 0)      # Acceptor: no shift
              .otherwise(pl.col('position'))
              .alias('position')
        ])
    
    return predictions_df
```

### Example

```
Original predictions (+ strand):
Position  Donor_Prob  Acceptor_Prob  Predicted_Type
98        0.2         0.7            acceptor
99        0.3         0.6            acceptor
100       0.9         0.1            donor
101       0.1         0.8            acceptor
102       0.2         0.7            acceptor
103       0.8         0.2            donor

After position adjustment:
Position  Donor_Prob  Acceptor_Prob  (original position → adjusted)
98        0.9         0.1            (100 → 98, donor)
98        0.2         0.7            (98 → 98, acceptor)  ← COLLISION!
99        0.3         0.6            (99 → 99, acceptor)
101       0.8         0.2            (103 → 101, donor)
101       0.1         0.8            (101 → 101, acceptor) ← COLLISION!
102       0.2         0.7            (102 → 102, acceptor)
```

### Issues

1. **Position Collisions**: Multiple original positions map to same adjusted coordinate
   - Position 100 (donor) → 98
   - Position 98 (acceptor) → 98
   - Result: Two rows with position=98

2. **Coverage Loss**: After deduplication/averaging, we lose positions
   - Original: 6 positions (98-103)
   - After adjustment + deduplication: 5 unique positions
   - Coverage: 83% (1 position lost)

3. **Data Loss**: Must average or drop duplicate positions
   - Averaging: Loses information about individual predictions
   - Dropping: Loses entire positions

4. **Ambiguity**: Which score belongs to which position?
   - Position 98 has two different score sets
   - Unclear which represents the "true" splice signal

### When This Approach Works

- **Training/Evaluation Mode**: When we only care about annotated splice sites
  - We filter to known splice sites, so collisions don't matter
  - We're comparing predictions at specific coordinates
  - Coverage of all positions is not required

- **Sparse Prediction Mode**: When we only want high-confidence predictions
  - Filter by threshold first, then adjust positions
  - Fewer predictions = fewer collisions

---

## Approach 2: Score-Shifting (NEW) - CORRECT

### Concept

Keep **position coordinates unchanged**, but shift the **score vectors** to align with GTF convention.

### Implementation

```python
def _apply_coordinate_adjustments(self, predictions_df):
    """Shift SCORE VECTORS, keep positions unchanged."""
    
    # Get adjustment values
    donor_adj = 2      # For + strand
    acceptor_adj = 0   # For + strand
    
    # Shift score vectors (not positions!)
    if strand == '+':
        predictions_df = predictions_df.with_columns([
            # Donor scores: position i gets score from position i+2
            pl.col('donor_prob').shift(-donor_adj).fill_null(0).alias('donor_prob'),
            
            # Acceptor scores: position i gets score from position i+0 (no shift)
            pl.col('acceptor_prob').shift(-acceptor_adj).fill_null(0).alias('acceptor_prob'),
            
            # Neither scores: unchanged
            pl.col('neither_prob').alias('neither_prob')
        ])
    
    return predictions_df  # Positions unchanged!
```

### Example

```
Original predictions (+ strand):
Position  Donor_Prob  Acceptor_Prob  Neither_Prob
98        0.2         0.7            0.1
99        0.3         0.6            0.1
100       0.9         0.1            0.0
101       0.1         0.8            0.1
102       0.2         0.7            0.1
103       0.8         0.2            0.0

After score adjustment (donor_adj=+2, acceptor_adj=0):
Position  Donor_Prob  Acceptor_Prob  Neither_Prob
98        0.9         0.7            0.1  ← donor from 100, acceptor from 98
99        0.1         0.6            0.1  ← donor from 101, acceptor from 99
100       0.2         0.1            0.0  ← donor from 102, acceptor from 100
101       0.8         0.8            0.1  ← donor from 103, acceptor from 101
102       0.0         0.7            0.1  ← donor from 104 (null→0), acceptor from 102
103       0.0         0.2            0.0  ← donor from 105 (null→0), acceptor from 103
```

### Benefits

1. **No Position Collisions**: Each position remains unique
   - 6 original positions → 6 adjusted positions
   - All positions preserved

2. **100% Coverage**: Every nucleotide has a score
   - N-bp gene → N positions with scores
   - No data loss

3. **Clear Interpretation**: Each position has one set of adjusted scores
   - Position 98 has adjusted donor score (from 100) and original acceptor score (from 98)
   - No ambiguity

4. **Biologically Correct**: Represents the actual splice signal location
   - The donor signal at position 100 (intronic) actually corresponds to junction at 98
   - We're saying "the splice signal we detected at 100 belongs to position 98"

### Interpretation

Think of it as having **different views** of the raw scores:

```
Raw scores (what SpliceAI outputs):
Position:     98    99   100   101   102   103
Donor:       0.2   0.3   0.9   0.1   0.2   0.8  ← Raw donor predictions
Acceptor:    0.7   0.6   0.1   0.8   0.7   0.2  ← Raw acceptor predictions

Adjusted scores (aligned with GTF):
Position:     98    99   100   101   102   103
Donor:       0.9   0.1   0.2   0.8   0.0   0.0  ← Shifted by +2 (from 100, 101, 102, 103, -, -)
Acceptor:    0.7   0.6   0.1   0.8   0.7   0.2  ← No shift (from 98, 99, 100, 101, 102, 103)
```

**Key Insight**: Position 98 keeps coordinate 98, but gets donor score from position 100.

---

## Technical Comparison

### Coverage

| Approach | GSTM3 (7,107 bp) | BRAF (205,603 bp) | TP53 (25,768 bp) |
|----------|------------------|-------------------|------------------|
| Position-Shifting | 6,300 (88.6%) | 171,413 (83.4%) | 21,646 (84.0%) |
| Score-Shifting | 7,107 (100%) ✅ | 205,603 (100%) ✅ | 25,768 (100%) ✅ |

### Position Collisions

| Approach | GSTM3 | BRAF | TP53 |
|----------|-------|------|------|
| Position-Shifting | 807 | 34,190 | 4,122 |
| Score-Shifting | 0 ✅ | 0 ✅ | 0 ✅ |

### Data Integrity

| Aspect | Position-Shifting | Score-Shifting |
|--------|-------------------|----------------|
| Positions preserved | ❌ No (collisions) | ✅ Yes (all N) |
| Scores preserved | ⚠️ Averaged/dropped | ✅ Yes (shifted) |
| Coverage | ❌ 83-88% | ✅ 100% |
| Ambiguity | ⚠️ Multiple scores per position | ✅ One score per position |

---

## Which Approach is Correct?

### For Inference Mode (Predicting ALL Positions)

**Score-Shifting is CORRECT** ✅

**Reasons**:
1. **Requirement**: Generate predictions for ALL nucleotide positions
   - Score-shifting: ✅ Achieves 100% coverage
   - Position-shifting: ❌ Loses 12-17% of positions

2. **Data Integrity**: Preserve all information
   - Score-shifting: ✅ No data loss
   - Position-shifting: ❌ Must average/drop collisions

3. **Biological Interpretation**: Clear signal-to-position mapping
   - Score-shifting: ✅ Each position has one adjusted score set
   - Position-shifting: ❌ Ambiguous (multiple scores per position)

4. **Downstream Analysis**: Enable position-level analysis
   - Score-shifting: ✅ Can analyze any position
   - Position-shifting: ❌ Missing positions can't be analyzed

### For Training/Evaluation Mode (Comparing to Annotations)

**Position-Shifting is ACCEPTABLE** ⚠️

**Reasons**:
1. **Focus**: Only care about annotated splice sites
   - We filter to known sites, so coverage doesn't matter
   - Collisions at non-annotated positions are irrelevant

2. **Simplicity**: Easier to implement and understand
   - Direct position adjustment
   - Clear correspondence to annotations

3. **Historical**: This is how it was originally implemented
   - Training pipeline uses this approach
   - Works fine for evaluation purposes

**However**, score-shifting would also work and is more principled!

---

## Polars `.shift()` Mechanics

Understanding how Polars shifts work is crucial:

```python
# shift(-n): Move values DOWN (from later to earlier indices)
# shift(+n): Move values UP (from earlier to later indices)

scores = [0.1, 0.2, 0.3, 0.4, 0.5]

# shift(-2): Each position gets value from position+2
scores.shift(-2)  # [0.3, 0.4, 0.5, null, null]
# Index 0 gets value from index 2 (0.3)
# Index 1 gets value from index 3 (0.4)
# Index 2 gets value from index 4 (0.5)
# Index 3 gets null (no index 5)
# Index 4 gets null (no index 6)

# shift(+2): Each position gets value from position-2
scores.shift(+2)  # [null, null, 0.1, 0.2, 0.3]
# Index 0 gets null (no index -2)
# Index 1 gets null (no index -1)
# Index 2 gets value from index 0 (0.1)
# Index 3 gets value from index 1 (0.2)
# Index 4 gets value from index 2 (0.3)
```

For adjustment +2 (get score from position+2), we use `shift(-2)`.

---

## Edge Cases and Boundary Handling

### Boundary Positions

**Problem**: Positions near gene boundaries may not have scores to shift from.

```
Gene: positions 0-99 (100 bp)
Donor adjustment: +2 (get score from position+2)

Position 98: needs score from position 100 ✓ (exists)
Position 99: needs score from position 101 ✗ (doesn't exist!)
```

**Solution**: Use `fill_null(0)` for missing values
```python
pl.col('donor_prob').shift(-2).fill_null(0)
```

**Interpretation**: Positions near boundaries have no upstream/downstream context, so score=0 is appropriate (no splice signal).

### Negative Strand

**Different Adjustments**: Negative strand may have different offsets.

```python
if strand == '+':
    donor_adj = 2
    acceptor_adj = 0
elif strand == '-':
    donor_adj = 1
    acceptor_adj = -1
```

**Why Different**: Coordinate systems are reversed on negative strand.

### Very Short Genes

**Problem**: Genes shorter than adjustment offset.

```
Gene: 5 bp (positions 0-4)
Donor adjustment: +2

All positions need scores from beyond gene boundary!
```

**Solution**: `fill_null(0)` handles this gracefully. Short genes will have mostly zero scores, which is biologically correct (insufficient context for splice prediction).

---

## Performance Implications

### Computational Cost

| Aspect | Position-Shifting | Score-Shifting |
|--------|-------------------|----------------|
| Adjustment operation | O(N) | O(N) |
| Collision resolution | O(N log N) | O(1) (none) |
| Memory | Same | Same |
| Total | O(N log N) | O(N) ✅ |

Score-shifting is actually **faster** because no collision resolution needed!

### Memory Usage

Both approaches use similar memory:
- Position-shifting: N rows → M unique rows (after deduplication)
- Score-shifting: N rows → N rows (no deduplication)

For large genes, score-shifting uses slightly more memory (keeps all N rows), but the difference is negligible.

---

## Validation Strategy

### Test 1: Coverage Verification ✅ PASSED

```
GSTM3 (7,107 bp):   7,107 positions (100%)
BRAF (205,603 bp):  205,603 positions (100%)
TP53 (25,768 bp):   25,768 positions (100%)
```

### Test 2: No Collisions ✅ PASSED

```
All genes: 0 position collisions
```

### Test 3: Alignment with GTF ⏳ PENDING

Compare predicted splice sites with GTF annotations:
- Load annotated splice sites
- Apply threshold to adjusted scores
- Calculate precision, recall, F1
- Expected: F1 >= position-shifting approach

### Test 4: Score Continuity ⏳ PENDING

Verify adjusted scores are biologically reasonable:
- Check score distributions
- Verify high scores at annotated sites
- Check for artifacts at boundaries

---

## Recommendations

### For Inference Mode (Production)

**Use Score-Shifting** ✅

```python
workflow = EnhancedSelectiveInferenceWorkflow()
predictions = workflow.predict_for_genes(
    gene_ids=['ENSG00000141510'],
    mode='base_only'
)
# Uses _apply_coordinate_adjustments() (score-shifting)
```

**Reasons**:
- 100% coverage guaranteed
- No position collisions
- Clear interpretation
- Faster (no collision resolution)

### For Training/Evaluation Mode

**Either Approach Works** ⚠️

Current training pipeline uses position-shifting, which is fine because:
- We only evaluate at annotated sites
- Coverage of all positions not required
- Historical compatibility

**However**, migrating to score-shifting would be beneficial:
- More consistent with inference
- More principled approach
- Would enable position-level training analysis

---

## Migration Path

### Phase 1: Inference (DONE) ✅

- ✅ Implement score-shifting in inference workflow
- ✅ Verify 100% coverage
- ⏳ Validate alignment with GTF
- ⏳ Compare performance with old approach

### Phase 2: Training (OPTIONAL)

- Update training workflow to use score-shifting
- Verify training metrics unchanged
- Update evaluation code
- Regenerate trained models

### Phase 3: Deprecation

- Mark `_apply_coordinate_adjustments_v0()` as deprecated
- Add warnings if old approach is used
- Eventually remove old implementation

---

## Conclusion

### Summary

| Criterion | Position-Shifting | Score-Shifting |
|-----------|-------------------|----------------|
| Coverage | ❌ 83-88% | ✅ 100% |
| Collisions | ❌ Yes (many) | ✅ No |
| Data Loss | ❌ Yes (averaging) | ✅ No |
| Inference Mode | ❌ Incorrect | ✅ Correct |
| Training Mode | ⚠️ Acceptable | ✅ Better |
| Performance | ⚠️ Slower | ✅ Faster |
| Clarity | ⚠️ Ambiguous | ✅ Clear |

### Final Verdict

**Score-shifting is the correct approach** for inference mode where we need predictions for ALL positions. It:
- Maintains 100% coverage (all N positions for N-bp gene)
- Eliminates position collisions
- Preserves all information
- Has clear biological interpretation
- Is computationally more efficient

**Position-shifting** is acceptable for training/evaluation but inferior. It should be considered deprecated for new code.

### User's Insight

The user correctly identified that we should have "different views of the score vector depending on the strand and splice type" - this is exactly what score-shifting implements, and it's the principled, correct approach.

---

## References

- Implementation: `enhanced_selective_inference.py`
  - Old: `_apply_coordinate_adjustments_v0()` (lines 547-616)
  - New: `_apply_coordinate_adjustments()` (lines 618-695)
- Test results: `SCORE_SHIFTING_SUCCESS.md`
- Technical details: `SCORE_SHIFTING_IMPLEMENTATION.md`
- Training workflow: `splice_prediction_workflow.py` (lines 287-364)

