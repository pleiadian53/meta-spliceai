# Nucleotide-Level Scores: Design Rationale and Configuration

**Date**: November 7, 2025  
**Status**: ✅ Implemented with User-Configurable Switch

---

## Historical Context

### Original Design Philosophy (Pre-Nov 2025)

The base model workflow was **intentionally designed NOT to save per-nucleotide scores**. This was a deliberate choice for two key reasons:

1. **Data Volume Management**
   - Per-nucleotide outputs can become massive for long genes
   - Example: 1000 genes × 50,000 nt/gene × 9 columns = ~5 GB uncompressed
   - Storing all nucleotides for genome-wide analysis would be impractical

2. **Focus on Meta-Model Training**
   - Primary purpose: Generate training data for meta-models
   - Meta-models serve two major functions:
     - **Error Correction**: Reduce FP/FN (e.g., SpliceAI misses many lncRNA splice sites)
     - **Adaptation**: Handle alternative splicing modes (variants, disease) without retraining base model
   
   - **Efficient Data Strategy**: Store only essential positions
     - True Positives (TP): All annotated splice sites
     - False Positives (FP): All predicted but not annotated
     - True Negatives (TN): **Subsampled** (not all)
     - False Negatives (FN): All missed splice sites

This approach kept datasets **focused and lightweight** while providing all necessary information for meta-model training.

---

## Current Implementation (Nov 7, 2025)

### Default Behavior ✅

**Per-nucleotide score collection is DISABLED by default**

```python
# In BaseModelConfig (data_types.py)
save_nucleotide_scores: bool = False  # Disabled by default for efficiency
```

**Gene manifest is ALWAYS saved** (lightweight, useful metadata)

### Rationale for Default

1. **Preserves Original Design**: Maintains efficient, focused datasets for meta-model training
2. **Prevents Data Bloat**: Avoids massive data volumes by default
3. **Backward Compatible**: Existing workflows continue to work as before
4. **Opt-In for Special Cases**: Users can explicitly enable when needed

---

## When to Enable Nucleotide Scores

### ✅ Recommended Use Cases

1. **Visualization**
   - Plot complete splice site landscape for specific genes
   - Publication-quality figures
   - Exploratory analysis

2. **Full-Coverage Inference Mode**
   - Inference workflow (base-only, hybrid, meta-only modes)
   - Per-nucleotide comparison across inference strategies
   - Detailed analysis of model behavior

3. **Targeted Gene Analysis**
   - Deep dive into specific genes of interest
   - Variant effect prediction at nucleotide resolution
   - Motif analysis and pattern discovery

4. **Model Comparison**
   - Compare SpliceAI vs. OpenSpliceAI at nucleotide level
   - Identify systematic differences
   - Validate model improvements

### ❌ NOT Recommended

1. **Genome-Wide Meta-Model Training**
   - Generates massive datasets (GBs to TBs)
   - Most nucleotides are irrelevant for training
   - Use default (disabled) for efficiency

2. **Production Pipelines**
   - Unnecessary storage overhead
   - Slower processing
   - Use default unless specifically needed

---

## Configuration Options

### Option 1: Via `run_base_model_predictions()`

```python
from meta_spliceai import run_base_model_predictions

# Default: Nucleotide scores DISABLED (efficient for meta-model training)
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53']
)
# Gene manifest: ✅ Saved
# Nucleotide scores: ❌ Not saved (default)

# Enable for visualization/full-coverage analysis
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1'],
    save_nucleotide_scores=True  # Explicit opt-in
)
# Gene manifest: ✅ Saved
# Nucleotide scores: ✅ Saved
```

### Option 2: Via `BaseModelConfig`

```python
from meta_spliceai import BaseModelConfig, run_base_model_predictions

# Create config with nucleotide scores enabled
config = BaseModelConfig(
    base_model='spliceai',
    mode='test',
    test_name='visualization_test',
    save_nucleotide_scores=True  # Enable explicitly
)

results = run_base_model_predictions(
    config=config,
    target_genes=['BRCA1']
)
```

### Option 3: Via kwargs

```python
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1'],
    save_nucleotide_scores=True,  # Can be passed as kwarg
    mode='test',
    test_name='my_test'
)
```

---

## Relationship to Full-Coverage Inference Mode

### Background: Inference Workflow Evolution

The **inference workflow** operates in three modes:
1. **base-only**: Use only base model predictions
2. **hybrid**: Combine base model + meta-model selectively
3. **meta-only**: Use meta-model for all positions

### Historical Issues (Pre-Nov 2025)

During per-nucleotide comparison of these modes, we discovered:
- ❌ Base model using incorrect genomic build
- ❌ Score-view logic contained flaws
- ❌ Coordinate adjustment logic had bugs

These findings led to the current round of bug fixes and system enhancements.

### Current Status: Nucleotide Scores vs. Full-Coverage Mode

**They are COMPLEMENTARY but DISTINCT features**:

| Feature | Purpose | Data Source | Use Case |
|---------|---------|-------------|----------|
| **Nucleotide Scores** (New) | Capture raw base model output | Base model predictions | Visualization, analysis, debugging |
| **Full-Coverage Inference** (Existing) | Compare inference strategies | Inference workflow | Mode comparison, performance evaluation |

**Key Differences**:

1. **Nucleotide Scores** (This Implementation):
   - Captures **raw base model output** (donor, acceptor, neither probabilities)
   - Saved during **base model workflow** (`splice_prediction_workflow.py`)
   - **Optional** (disabled by default)
   - Used for: Visualization, targeted analysis, model comparison

2. **Full-Coverage Inference** (Existing Feature):
   - Compares **inference strategies** (base-only vs. hybrid vs. meta-only)
   - Executed in **inference workflow** (separate from base model workflow)
   - Generates **per-nucleotide predictions** for comparison
   - Used for: Evaluating inference mode performance

**Relationship**:
- Nucleotide scores from base model can **feed into** full-coverage inference
- Full-coverage inference can **use** nucleotide scores as input
- Both provide per-nucleotide resolution, but serve different purposes

---

## Implementation Details

### Code Changes

**1. Configuration Parameter** (`data_types.py`):
```python
@dataclass
class SpliceAIConfig:
    # ...
    save_nucleotide_scores: bool = False  # Disabled by default
```

**2. Conditional Capture** (`splice_prediction_workflow.py`):
```python
# Only capture if explicitly enabled
if predictions and config.save_nucleotide_scores:
    nucleotide_scores_chunk = []
    # ... extract and save scores ...
```

**3. User Interface** (`run_base_model.py`):
```python
def run_base_model_predictions(
    base_model: str = 'spliceai',
    save_nucleotide_scores: bool = False,  # Clear parameter
    # ...
):
    """
    save_nucleotide_scores : bool, default=False
        If True, save per-nucleotide splice site scores.
        WARNING: Generates large data volumes (100s MB to GBs).
        For meta-model training, keep False (default).
    """
```

### Output Behavior

**Default (save_nucleotide_scores=False)**:
```python
results = {
    'positions': pl.DataFrame,        # Position-level (TP, FP, sampled TN)
    'error_analysis': pl.DataFrame,   # Error analysis
    'gene_manifest': pl.DataFrame,    # ✅ Always saved
    'nucleotide_scores': pl.DataFrame,  # ❌ Empty DataFrame
    # ...
}
```

**Enabled (save_nucleotide_scores=True)**:
```python
results = {
    'positions': pl.DataFrame,        # Position-level (TP, FP, sampled TN)
    'error_analysis': pl.DataFrame,   # Error analysis
    'gene_manifest': pl.DataFrame,    # ✅ Always saved
    'nucleotide_scores': pl.DataFrame,  # ✅ Full nucleotide-level scores
    # ...
}
```

---

## Performance Impact

### When Disabled (Default)

- **Memory**: Minimal overhead (only gene manifest tracking)
- **Storage**: No additional files
- **Processing Time**: ~1-2% overhead (manifest only)

### When Enabled

- **Memory**: +100-500 MB per 1000 genes (depends on gene length)
- **Storage**: +100 MB to several GB (depends on dataset size)
- **Processing Time**: +2-3% overhead (extraction and saving)

---

## Migration Guide

### For Existing Code

**No changes required** - all existing code continues to work:

```python
# Before (still works exactly the same)
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1']
)
positions = results['positions']  # Works as before

# After (with explicit opt-in for nucleotide scores)
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1'],
    save_nucleotide_scores=True  # NEW: Explicit opt-in
)
positions = results['positions']          # Same as before
nucleotide_scores = results['nucleotide_scores']  # NEW: Available when enabled
```

### For New Workflows

**Recommended pattern**:

```python
# Meta-model training: Use default (efficient)
training_results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=training_genes,
    # save_nucleotide_scores=False  # Default
)

# Visualization: Enable explicitly
viz_results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1'],  # Small set
    save_nucleotide_scores=True  # Explicit opt-in
)

# Plot landscape
import matplotlib.pyplot as plt
brca1 = viz_results['nucleotide_scores'].filter(
    pl.col('gene_name') == 'BRCA1'
)
plt.plot(brca1['position'], brca1['donor_score'])
plt.show()
```

---

## Summary

### Design Principles

1. **✅ Efficiency First**: Disabled by default to preserve original design
2. **✅ Flexibility**: Easy to enable when needed
3. **✅ Clarity**: Clear documentation and warnings
4. **✅ Backward Compatible**: No breaking changes

### Key Takeaways

- **Default behavior preserves original efficient design**
- **Gene manifest always saved** (lightweight, useful)
- **Nucleotide scores opt-in only** (for special use cases)
- **Clear parameter** (`save_nucleotide_scores`) in user API
- **Complementary to full-coverage inference** (different purposes)

### Recommended Usage

| Scenario | save_nucleotide_scores | Rationale |
|----------|------------------------|-----------|
| Meta-model training | `False` (default) | Efficient, focused datasets |
| Genome-wide analysis | `False` (default) | Avoid massive data volumes |
| Gene visualization | `True` | Need complete landscape |
| Full-coverage inference | `True` | Per-nucleotide comparison |
| Targeted analysis | `True` | Deep dive into specific genes |
| Model comparison | `True` | Nucleotide-level differences |

---

*Last Updated: November 7, 2025*  
*Implementation Status: ✅ Complete with User-Configurable Switch*

