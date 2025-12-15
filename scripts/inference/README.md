# Inference Workflow Scripts

Scripts for testing and running the meta-model inference workflow.

---

## Overview

This directory contains tools for testing the **inference workflow** after a meta-model has been trained. These scripts test how the trained meta-model makes predictions on new genes and positions.

> **Note:** This is separate from training! For monitoring training progress, see `scripts/monitoring/`.

---

## Inference Workflow Overview

The inference workflow supports **3 operational modes**:

### 1. **Base-Only Mode**
- Pure SpliceAI predictions without meta-model enhancement
- Establishes baseline performance
- Fastest inference mode

### 2. **Hybrid Mode** (Recommended)
- Intelligent selective recalibration
- Meta-model applied only to uncertain positions (2-5%)
- Optimal balance of speed and accuracy

### 3. **Meta-Only Mode**
- Complete meta-model recalibration for ALL positions
- Maximum enhancement coverage
- Comprehensive but slower

---

## Test Scenarios

The inference workflow should be tested on **two gene scenarios**:

### Scenario A: Training Genes (Gap Filling)
- Genes that were in the training dataset
- Meta-model predicts **unseen positions** (non-splice sites)
- Tests interpolation capability

### Scenario B: Unseen Genes (Full Prediction)
- Genes never seen during training
- Meta-model predicts **all positions**
- Tests generalization capability

---

## Planned Scripts

### `prepare_inference_gene_lists.sh`
Generate gene lists for testing both scenarios:
```bash
# Generate 3 training genes + 3 unseen genes
./scripts/inference/prepare_inference_gene_lists.sh \
    --training-genes 3 \
    --unseen-genes 3 \
    --output gene_lists/inference_test
```

### `test_inference_modes.sh`
Test all 3 operational modes on both gene scenarios:
```bash
# Test all 6 combinations (3 modes × 2 scenarios)
./scripts/inference/test_inference_modes.sh \
    --model results/meta_model_1000genes_3mers \
    --gene-lists gene_lists/inference_test \
    --output results/inference_validation
```

### `run_inference_workflow.sh`
Production inference script:
```bash
# Run inference on new genes
./scripts/inference/run_inference_workflow.sh \
    --model results/meta_model_1000genes_3mers \
    --genes-file new_genes.txt \
    --mode hybrid \
    --output results/inference_output
```

### `monitor_inference.sh`
Monitor inference progress:
```bash
# Monitor running inference
./scripts/inference/monitor_inference.sh --run-name inference_test
```

---

## Current Status

🚧 **Work in Progress**

The inference workflow testing is currently being developed. Key components:

- ✅ Core inference modules (`meta_spliceai/.../workflows/inference/`)
- ✅ Gene list preparation (`prepare_gene_lists.py`)
- ✅ Main inference driver (`splice_inference_workflow.py`)
- 🔨 Testing scripts (this directory) - IN PROGRESS
- ⏳ Monitoring tools - PLANNED

---

## Related Documentation

- **`meta_spliceai/splice_engine/meta_models/workflows/docs/COMPLETE_SPLICE_WORKFLOW.md`**
  - Comprehensive inference workflow documentation
  - Detailed explanation of all 3 modes
  - Gene selection strategies
  - Performance expectations

- **`meta_spliceai/splice_engine/meta_models/workflows/inference/`**
  - Core inference implementation
  - `main_inference_workflow.py` - Main driver
  - `prepare_gene_lists.py` - Gene selection utility
  - `data_resource_manager.py` - Resource management

---

## Workflow: Training → Inference

```
1. Dataset Generation
   └─> scripts/builder/run_builder_resumable.sh
        └─> data/train_pc_1000_3mers/

2. Meta-Model Training
   └─> python -m ...run_gene_cv_sigmoid
        ├─> Monitor: scripts/monitoring/monitor_meta_training.sh
        └─> results/meta_model_1000genes_3mers/model_multiclass.pkl

3. Inference Testing (THIS DIRECTORY)
   └─> scripts/inference/test_inference_modes.sh
        ├─> Scenario A: Training genes (gap filling)
        ├─> Scenario B: Unseen genes (generalization)
        ├─> Mode 1: Base-only
        ├─> Mode 2: Hybrid
        └─> Mode 3: Meta-only

4. Production Inference
   └─> scripts/inference/run_inference_workflow.sh
```

---

## Example Test Plan

### Quick Validation (5-10 min)
```bash
# 1. Prepare small gene lists
python -m meta_spliceai...prepare_gene_lists \
    --training 3 --unseen 3 \
    --gene-types protein_coding \
    --output gene_lists/quick_test

# 2. Test one mode on training genes
python -m meta_spliceai...main_inference_workflow \
    --model results/meta_model_1000genes_3mers \
    --genes-file gene_lists/quick_test/training_genes.txt \
    --mode hybrid \
    --output results/test_hybrid_training
```

### Comprehensive Test (30-60 min)
```bash
# Test all 6 combinations:
# - 3 modes (base_only, hybrid, meta_only)
# - 2 scenarios (training genes, unseen genes)
./scripts/inference/test_inference_modes.sh --full
```

---

## Key Differences from Training

| Aspect | Training (`scripts/monitoring/`) | Inference (`scripts/inference/`) |
|--------|----------------------------------|----------------------------------|
| **Goal** | Train meta-model on dataset | Apply trained model to new data |
| **Input** | Training dataset (parquet) | Gene lists + trained model |
| **Output** | model_multiclass.pkl | Per-position predictions |
| **Time** | Hours (1000 genes) | Minutes (small gene sets) |
| **Monitoring** | Fold progress, metrics | Gene completion, mode performance |

---

## Notes

- 🎯 Focus on **testing inference**, not training monitoring
- ✅ Use pre-trained models from `results/meta_model_*/`
- 📊 Compare performance across all 3 modes
- 🧬 Test both seen and unseen genes
- ⚡ Hybrid mode should be fastest and most accurate

---

## Future Work

- [ ] Complete `test_inference_modes.sh` implementation
- [ ] Add inference monitoring script
- [ ] Create performance comparison reports
- [ ] Add batch inference for large gene sets
- [ ] Integrate with variant analysis pipeline

