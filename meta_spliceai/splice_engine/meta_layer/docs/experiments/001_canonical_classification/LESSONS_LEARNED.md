# Lessons Learned: Experiment 001

**Experiment**: Canonical Splice Site Classification  
**Date**: December 14, 2025

---

## Key Insights

### 1. Classification Accuracy ≠ Variant Detection Ability

**Lesson**: High accuracy on canonical splice site classification does not imply the ability to detect variant effects.

**Evidence**:
- Meta-layer achieved 99.11% classification accuracy (+1.5% over base model)
- But only detected 17% of splice-altering variants (vs 67% for base model)

**Why**: The model learns "what IS a splice site" but not "what CHANGES a splice site."

**Recommendation**: Always evaluate on the target task, not a proxy task.

---

### 2. Training Objective Must Match Evaluation Objective

**Lesson**: The loss function and training data must align with the ultimate evaluation goal.

**This Experiment**:
```
Training:   CrossEntropyLoss(predicted_class, GTF_label)
Evaluation: Detection of splice-altering variants in SpliceVarDB
```

**Mismatch**: No variant pairs, no delta scores, no SpliceVarDB labels during training.

**Recommendation**: For variant effect detection, train with:
- (ref, alt) sequence pairs
- Delta score targets
- SpliceVarDB classifications as labels or weights

---

### 3. Position-Centric Architecture Has Limitations

**Lesson**: A model that outputs `[3]` for a single position cannot naturally capture "where in this window does the biggest change occur?"

**Base Model Advantage**:
- Outputs `[L, 3]` for all positions
- Can find MAX |delta| anywhere in sequence
- Designed for variant effect scanning

**Meta-Layer Limitation**:
- Outputs `[3]` for center position only
- Can only measure change at that exact position
- Variant effects at nearby positions are invisible

**Recommendation**: Consider sequence-to-sequence architecture for variant effect detection, or use sliding window inference.

---

### 4. Context Window Size vs. Variant Sensitivity Trade-off

**Lesson**: Large context windows (501nt) provide rich information for classification but dilute the signal from single-nucleotide variants.

**Math**: A single variant is 1/501 = 0.2% of the input.

**Observation**: Meta-layer delta scores were 88% smaller than base model deltas.

**Possible Solutions**:
1. Smaller, variant-centered context windows
2. Explicit variant position encoding
3. Attention mechanisms that focus on the variant
4. Variant-aware input representations

---

### 5. Pre-computed Artifacts Enable Rapid Experimentation

**Lesson**: Having pre-computed artifacts with all features enabled rapid model development and testing.

**Benefit**: 
- Loaded 20,000 training samples in ~4 seconds
- No need to re-run base model for each experiment
- Features already extracted and normalized

**Recommendation**: Continue using artifact-based training for meta-layer experiments. Only run base model for new genes or variants.

---

### 6. Path Management Prevents Production Data Corruption

**Lesson**: Separating read (production) and write (development) paths is essential during active development.

**Implementation**:
```python
pm = MetaLayerPathManager(base_model='openspliceai')
pm.get_artifacts_read_dir()   # → .../meta_models/  (READ-ONLY)
pm.get_output_write_dir()     # → .../meta_layer_dev/{timestamp}/  (SAFE)
```

**Benefit**: Model checkpoints and experimental outputs are isolated from production artifacts.

---

### 7. SpliceVarDB is a Critical Evaluation Resource

**Lesson**: SpliceVarDB provides ground-truth variant classifications that are essential for evaluating variant effect prediction.

**Key Statistics** (chromosome 21):
- 492 total variants
- 152 splice-altering (31%)
- 90 normal (18%)
- 249 low-frequency (51%)

**Recommendation**: Use SpliceVarDB as the primary benchmark for variant effect detection. Consider using it for training in Phase 2.

---

## Technical Lessons

### Data Leakage Detection

The `MetaLayerDataset` correctly detected and excluded leakage columns:
```
⚠️ LEAKAGE COLUMNS DETECTED: ['splice_type', 'pred_type', 'true_position', 'predicted_position']
These will be excluded from features.
```

**Lesson**: Automated leakage detection is valuable for preventing accidental information leakage.

### Balanced Sampling

Balanced sampling (6,666 per class) helped achieve stable training:
- Training accuracy reached 100% by epoch 15
- No class imbalance issues

**Lesson**: For 3-class classification with imbalanced data, balanced sampling is effective.

### Device Handling

The code correctly handled device selection (MPS on Mac):
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

**Lesson**: Auto-device selection simplifies cross-platform development.

---

## What Worked Well

1. ✅ Multimodal architecture (sequence + features) improves classification
2. ✅ Pre-computed artifacts enable fast iteration
3. ✅ Path manager prevents accidental production overwrites
4. ✅ Leakage detection catches potential issues
5. ✅ Training converges quickly (15 epochs)

## What Didn't Work

1. ❌ Classification training doesn't transfer to variant detection
2. ❌ Position-centric output limits variant effect analysis
3. ❌ Single variant barely changes 501nt context
4. ❌ No variant-aware training signal

---

## Recommendations for Phase 2

Based on these lessons, Phase 2 should:

1. **Train on variant pairs**: Use (ref_seq, alt_seq) inputs
2. **Predict deltas directly**: Output delta scores, not class probabilities
3. **Use SpliceVarDB for training**: Include variant classifications in the loss
4. **Consider sequence-level output**: Enable MAX delta detection in windows
5. **Focus on the variant**: Use attention or explicit position encoding

---

## Appendix: Experiment Timeline

| Time | Action | Result |
|------|--------|--------|
| 16:10 | Path manager setup | ✅ Isolated dev paths |
| 16:13 | Load training data (20k samples) | ✅ 4 seconds |
| 16:14 | Train meta-layer (15 epochs) | ✅ 99% accuracy |
| 16:15 | Test on chr21 | ✅ 99.11% accuracy |
| 16:16 | SpliceVarDB evaluation | ❌ 17% detection |
| 16:17 | Analysis complete | → Phase 2 needed |

