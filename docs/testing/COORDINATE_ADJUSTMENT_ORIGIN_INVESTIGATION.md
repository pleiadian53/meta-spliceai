# Investigation: Origin of SpliceAI Coordinate Adjustment Values

## Date: 2025-10-31

## Background

The codebase contains documented coordinate adjustments for SpliceAI:
```python
{
    'donor': {'plus': 2, 'minus': 1},
    'acceptor': {'plus': 0, 'minus': -1}
}
```

These are claimed to be "empirically determined," but our current tests show they make predictions **WORSE** for our workflow.

## Documentation Trail

### 1. `BASE_MODEL_SPLICE_SITE_DEFINITIONS.md`
**Location**: `meta_spliceai/system/docs/`

**Claims**:
- "From extensive empirical analysis"
- "Offsets determined through systematic empirical analysis"
- Source: `splice_utils.py`

**Missing**:
- No methodology described
- No reference to which dataset was used
- No test results showing these values improve performance

### 2. `coordinate_reconciliation.py`
**Location**: `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/`

**Comment** (line 107):
```python
# SpliceAI model adjustments (from your analysis)
```

**Interpretation**: The comment suggests these values were **user-provided**, not empirically determined by the system!

### 3. `SPLICE_SITE_DEFINITION_ANALYSIS.md`
**Location**: `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/`

**Context**:
- Documents OpenSpliceAI offsets: `donor: +1, acceptor: 0`
- Shows "combined" offsets: OpenSpliceAI (+1) + SpliceAI (+2/+1)
- Describes as "Your Current Adjustments"

**Hypothesis**: The SpliceAI adjustments may have been:
1. Meant for use WITH OpenSpliceAI preprocessing (which has +1 offset)
2. Incorrectly separated from the OpenSpliceAI context

## Training Code Investigation

### Evaluation Tolerance Window

**Found in**: `eval_meta_splice.py` (line 87, 226-227)
```python
def _vectorised_site_metrics(..., window: int = 2, ...)
```

**Key Finding**: The evaluation uses a **±2bp tolerance window** for matching predictions to truth!

This means:
- A prediction at position X matches truth at position X-2, X-1, X, X+1, or X+2
- This 2bp window MASKS coordinate misalignment
- Performance looks good even with wrong adjustments!

### Position Debug Code

**Found in**: `eval_meta_splice.py` (lines 207-227)

There's extensive debug code for detecting coordinate offsets:
```python
# Calculate all pairwise distances to identify systematic offsets
min_distances = []
for pred_pos in pred_abs:
    distances = [abs(pred_pos - truth_pos) for truth_pos in truth_abs]
    min_distances.append(min(distances))

median_offset = np.median(min_distances)
if median_offset > 100:
    print(f"⚠️  LARGE SYSTEMATIC OFFSET DETECTED: {median_offset}bp")
```

**Implication**: This code was written to DETECT coordinate problems, suggesting they were aware of the issue!

## Hypothesis: The 2bp Window Theory

### The Problem

1. **Evaluation uses ±2bp window**
2. **SpliceAI predictions may be off by +2bp** (as we observed)
3. **Within the 2bp window, they still count as correct!**
4. **Adjustment values (+2/+1) were derived from this analysis**
5. **But the adjustment DIRECTION may be backwards**

### Example Scenario

**True donor site**: Position 1000

**SpliceAI predicts**: Position 1002 (2bp upstream)

**With ±2bp window**: This counts as a match! ✅

**Documented adjustment**: `donor: {plus: 2}`
- Interpretation: "Model predicts 2nt upstream"
- Intended fix: Shift positions by -2

**But in our test**:
- Base predictions at 1000 match truth perfectly
- After shifting by -2, predictions move to 998
- Now they're 2bp off in the WRONG direction!

## OpenSpliceAI Connection

### Known OpenSpliceAI Offsets

From OpenSpliceAI source code analysis:
```python
{
    'donor': {'plus': 1, 'minus': 1},  # Both strands +1
    'acceptor': {'plus': 0, 'minus': 0}  # Both strands 0
}
```

### Combined Workflow Theory

**If using OpenSpliceAI preprocessing + SpliceAI model**:
- OpenSpliceAI defines donors at +1 from GTF
- SpliceAI trained on standard GTF
- Total offset: +1 (OpenSpliceAI) + need correction
- Maybe the +2/+1 were meant for THIS combined workflow?

## Our Current Evidence

### Test Results (VCAM1, + strand)

| Configuration | Overall F1 | Donor F1 | Evidence |
|--------------|------------|----------|----------|
| Zero adjustments | 0.756 | 0.696 | Base model aligned with GTF ✅ |
| With +2/+1 adjustments | 0.400 | 0.000 | Adjustments destroy alignment ❌ |

### Direct Observation

**Manual inspection**:
- Base SpliceAI predictions: Match true donor positions exactly
- After +2 adjustment: Off by +2 from true positions

**Conclusion**: For our workflow (SpliceAI → GTF annotations), zero adjustments are optimal.

## Questions for User

1. **Was the codebase previously using OpenSpliceAI preprocessing?**
   - If yes, the +2/+1 adjustments may have been correct for that workflow
   - But not for direct SpliceAI → GTF

2. **Were the adjustments determined on a different GTF version?**
   - Different Ensembl releases (e.g., GRCh37 vs GRCh38)
   - Different splice site definitions

3. **Was the ±2bp evaluation window always used?**
   - If so, it would hide coordinate misalignment
   - Making it hard to detect incorrect adjustments

4. **Are there test results showing the +2/+1 values improve performance?**
   - We need to find the original empirical analysis
   - Compare their methodology to ours

## Recommendations

### 1. Document Current Findings
Update `BASE_MODEL_SPLICE_SITE_DEFINITIONS.md` with:
- Our empirical evidence (VCAM1 test)
- Note that adjustments are workflow-dependent
- Separate sections for "SpliceAI alone" vs "OpenSpliceAI + SpliceAI"

### 2. Validate Across Multiple Genes
Complete the 20-gene test to confirm:
- Zero adjustments work consistently
- Both + and - strands
- Different gene characteristics

### 3. Test OpenSpliceAI Workflow
If we have OpenSpliceAI:
- Test with its documented +1 offset
- See if combined +3/+2 adjustments are needed
- Document the correct workflow

### 4. Re-evaluate Training Code
The ±2bp window may need adjustment:
- Consider using ±0bp or ±1bp for stricter evaluation
- Would reveal coordinate misalignment more clearly
- But may lower reported performance

## Conclusion

The documented SpliceAI adjustments (`+2/+1, 0/-1`) appear to be:

1. **Wrong for our current workflow** (SpliceAI → GTF directly)
2. **Possibly correct for a different workflow** (with OpenSpliceAI preprocessing)
3. **Derived using a ±2bp tolerance window** that masked the actual alignment
4. **Not properly documented** in terms of methodology and applicability

**Current recommendation**: Use zero adjustments for SpliceAI with our GTF annotations.

**The multi-view adjustment system we built is correct** and ready for workflows that DO need adjustments.

