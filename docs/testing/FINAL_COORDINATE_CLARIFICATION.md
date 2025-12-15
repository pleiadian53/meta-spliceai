# Final Coordinate Clarification - Corrected

**Date**: November 2, 2025  
**Purpose**: Correct understanding of evaluation parameters

---

## My Mistake

I incorrectly stated that `error_window = 500` was used for splice site matching tolerance. **This was wrong!**

You correctly pointed out:
> "`error_window` is for extracting contextual sequences surrounding the queried position, similar to how we extract contextual sequences to derive k-mers for meta learning."

---

## The Correct Understanding

### Two Different Windows

**1. `consensus_window` (default = 2)**
- **Purpose**: Tolerance for matching predicted positions to true splice sites
- **Usage**: ±2bp window around true splice site
- **Code**: `if window_start <= i <= window_end` where `window_start = true_pos - consensus_window`
- **This is what determines TP/FP/FN classification**

**2. `error_window` (default = 500)**
- **Purpose**: Define boundary of contextual sequences for feature extraction
- **Usage**: Extract ±500bp sequences around errors for analysis
- **Future use**: Input for deep-learning sequence modules
- **This does NOT affect TP/FP/FN classification**

---

## What This Means for Our Tests

### Previous Test (50 genes)

**Configuration**:
```python
consensus_window = 2   # ±2bp matching tolerance
error_window = 500     # ±500bp for sequence extraction
```

**Result**: F1 = 0.9312

**Interpretation**:
- Predictions within ±2bp of true sites counted as correct
- This is the **standard evaluation protocol**
- F1 = 0.9312 is the **true performance**

### Current Test (AGPAT3)

**Configuration**:
```python
consensus_window = 2   # ±2bp matching tolerance
error_window = 500     # ±500bp for sequence extraction
```

**Result**: F1 = 0.0000

**The Real Question**: Why did AGPAT3 fail when it should have the same ±2bp tolerance?

---

## The Real Issue with AGPAT3

### The -2bp Offset

**Finding**: All predictions are offset by -2bp
```
Predicted: 106304
True:      106306
Offset:    -2bp
```

### Why F1 = 0?

**With consensus_window = 2**:
```python
window_start = true_pos - consensus_window  # 106306 - 2 = 106304
window_end = true_pos + consensus_window    # 106306 + 2 = 106308

# Check if predicted position (106304) is in window
if 106304 <= 106304 <= 106308:  # TRUE!
    # Should be counted as TP
```

**The prediction SHOULD be within the window!**

So why F1 = 0? Let me check the actual matching logic more carefully...

---

## Investigating the True Cause

### Hypothesis 1: Window Boundary Issue

**The -2bp offset is at the exact edge of the window:**
- Window: [106304, 106308]
- Predicted: 106304
- This is **at the lower boundary**

**Possible issue**: Off-by-one error in boundary checking?

### Hypothesis 2: Threshold Issue

**All predictions need score >= threshold (0.5)**

From AGPAT3 analysis:
- 78 donor positions with score ≥ 0.5
- But 0 True Positives

**Possible issue**: Scores at predicted positions might be below threshold, even though nearby positions have high scores?

### Hypothesis 3: Evaluation Bug

**The evaluation might not be working correctly for this gene**

Evidence:
- High scores (0.95-0.998)
- Predictions within ±2bp
- But still 0 TPs

---

## What We Need to Check

### 1. Verify the Matching Logic

Check if the evaluation code is correctly identifying TPs for AGPAT3:

```python
# For each predicted position with score >= threshold
# Check if it's within consensus_window of any true site
# If yes → TP
# If no → FP
```

### 2. Check Score Thresholding

Verify that scores at the **predicted positions** (not nearby) are >= 0.5:

```python
# We know max score is 0.998
# But what's the score at the exact predicted position?
```

### 3. Examine the Positions DataFrame

Look at the actual `pred_type` assignments:
- How many positions have `pred_type = 'TP'`?
- How many have `pred_type = 'FP'`?
- What are their scores?

---

## Next Steps

### Immediate Action

**Re-analyze AGPAT3 with detailed logging**:

1. Check scores at predicted positions (not just max scores)
2. Verify window matching logic
3. Examine why 0 TPs despite high scores and -2bp offset

### Possible Outcomes

**Outcome A**: Evaluation bug
- Fix the bug
- Re-run test
- Expect F1 ~1.0

**Outcome B**: Score threshold issue
- Scores at predicted positions < 0.5
- But nearby positions have high scores
- This would explain 0 TPs

**Outcome C**: Something else
- Need more investigation

---

## Apology and Correction

I apologize for the confusion about `error_window`. You were absolutely correct:

- **`consensus_window = 2`**: Matching tolerance (±2bp)
- **`error_window = 500`**: Contextual sequence boundary (±500bp)

The F1 = 0 for AGPAT3 is **not** explained by evaluation windows. There's something else going on that we need to investigate.

---

## Summary

### What I Got Wrong

❌ Said `error_window = 500` was for matching tolerance  
❌ Thought previous test used ±500bp matching  
❌ Concluded that's why F1 was high

### What's Actually True

✅ `consensus_window = 2` is for matching (both tests)  
✅ `error_window = 500` is for sequence extraction  
✅ Both tests should use same ±2bp matching

### The Real Mystery

**Why does AGPAT3 have F1 = 0 when:**
- Predictions have high scores (0.95-0.998)
- Offset is -2bp (within ±2bp tolerance)
- Same evaluation parameters as previous test (F1 = 0.9312)

**This needs further investigation!**

---

**Date**: November 2, 2025  
**Status**: Corrected understanding, but AGPAT3 failure still unexplained



