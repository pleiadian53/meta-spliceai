# Meta-Layer Experiments

This directory contains documentation for experiments conducted on the meta-layer architecture for splice site prediction.

---

## Experiment Index

| ID | Name | Status | Outcome | Date |
|----|------|--------|---------|------|
| [001](./001_canonical_classification/) | Canonical Classification | ✅ Completed | Partial Success | 2025-12-14 |
| [002](./002_delta_prediction/) | Delta Prediction (Paired) | ✅ Completed | r=0.38 (insufficient) | 2025-12-14 |
| [003](./003_binary_classification/) | Binary Classification (Multi-Step Step 1) | ✅ Completed | AUC=0.61, F1=0.53 | 2025-12-15 |
| [004](./004_validated_delta/) | **Validated Delta (Single-Pass)** | ✅ Completed | **r=0.41 (best!)** | 2025-12-15 |

---

## Experiment Categories

### Classification-Based Approaches

- **001_canonical_classification**: Train on GTF labels, evaluate on SpliceVarDB (FAILED for variants)
- **003_binary_classification**: Multi-Step Step 1 - "Is this variant splice-altering?" (IN PROGRESS)

### Delta-Based Approaches

- **002_delta_prediction**: Paired (Siamese) prediction (r=0.38)
- **004_validated_delta**: **Single-pass with validated targets (r=0.41) - BEST**

---

## Directory Structure

Each experiment follows this structure:

```
NNN_experiment_name/
├── README.md           # Overview, hypothesis, setup, results summary
├── RESULTS.md          # Detailed numerical results
├── ANALYSIS.md         # In-depth analysis of results
├── LESSONS_LEARNED.md  # Key insights and recommendations
└── (optional)
    ├── config.yaml     # Experiment configuration
    ├── notebook.ipynb  # Analysis notebook
    └── figures/        # Plots and visualizations
```

---

## Naming Convention

Experiments are numbered sequentially:
- `001_`, `002_`, ... `NNN_`

Names are descriptive but concise:
- `canonical_classification` - What labels/objective
- `delta_prediction` - What is being predicted
- `variant_aware` - Special training approach

---

## Key Metrics

### For Classification Experiments
- **Accuracy**: Overall classification accuracy
- **AP (Average Precision)**: Per-class ranking quality
- **PR-AUC**: Area under precision-recall curve

### For Variant Detection Experiments
- **Detection Rate**: % of splice-altering variants detected
- **Mean |Δ|**: Average absolute delta score
- **Sensitivity/Specificity**: At various thresholds

---

## Quick Reference

### Current Best Results

| Task | Best Model | Metric | Value |
|------|------------|--------|-------|
| Classification | Meta-Layer (001) | Accuracy | 99.11% |
| Variant Detection | Base Model | Detection Rate | 67% |

### Key Findings

1. **Classification ≠ Detection**: High classification accuracy doesn't translate to variant detection
2. **Training objective matters**: Must train for the evaluation task
3. **Architecture matters**: Position-centric output limits variant analysis
4. **Target quality matters**: Learning from potentially wrong base model deltas limits Approach A
5. **Binary classification is learnable**: AUC=0.61 > random, but F1=0.53 needs improvement (>0.7)

---

## How to Add a New Experiment

1. Create directory: `NNN_experiment_name/`
2. Copy template from existing experiment
3. Update `README.md` with hypothesis and setup
4. Run experiment, record results in `RESULTS.md`
5. Analyze and document in `ANALYSIS.md`
6. Summarize learnings in `LESSONS_LEARNED.md`
7. Update this index

---

## GPU Training

For experiments requiring GPU resources, see:

- [GPU_TRAINING_GUIDE.md](./GPU_TRAINING_GUIDE.md) - **Jump-start GPU training on RunPods**
- [DATA_TRANSFER_GUIDE.md](./DATA_TRANSFER_GUIDE.md) - **Transfer required data files**
- [../wishlist/GPU_EXPERIMENTS.md](../wishlist/GPU_EXPERIMENTS.md) - Experiment queue

### Priority GPU Experiments

| Priority | Experiment | Expected Improvement |
|----------|------------|---------------------|
| ⭐ 1 | HyenaDNA + ValidatedDelta | r=0.41 → r>0.55 |
| 2 | Full SpliceVarDB (50K) | Better generalization |
| 3 | Longer context (1001nt) | Capture distant effects |
| 4 | Cross-validation | Robust estimates |

---

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Meta-layer architecture
- [LABELING_STRATEGY.md](../LABELING_STRATEGY.md) - Labeling approaches
- [DELTA_SCORE_IMPLEMENTATION.md](../DELTA_SCORE_IMPLEMENTATION.md) - Delta computation
- [TRAINING_VS_INFERENCE.md](../TRAINING_VS_INFERENCE.md) - Data format differences
- [../methods/ROADMAP.md](../methods/ROADMAP.md) - Methodology roadmap
- [../methods/GPU_REQUIREMENTS.md](../methods/GPU_REQUIREMENTS.md) - Compute requirements

