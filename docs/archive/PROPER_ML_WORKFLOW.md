# Proper Machine Learning Workflow for Meta-Model Training

## Summary

This document clarifies the **correct machine learning workflow** for the splice meta-model training system, addressing the fundamental issues identified in the current implementation.

## Current Issues

1. **Confusion about `--sample-genes`**: Currently being misunderstood as a parameter for production CV, when it should ONLY be for testing/debugging
2. **Data leakage**: Evaluating models on the same data used for training
3. **Inconsistent evaluation**: No clear strategy for post-training evaluation

## Correct ML Workflow

### 1. Cross-Validation Phase

**Purpose**: Understand model performance and variance

**Production Example** (10,000 genes dataset):
```
5-fold CV:
- Fold 1: Train on 8,000 genes, Validate on 2,000 genes
- Fold 2: Train on 8,000 genes, Validate on 2,000 genes  
- Fold 3: Train on 8,000 genes, Validate on 2,000 genes
- Fold 4: Train on 8,000 genes, Validate on 2,000 genes
- Fold 5: Train on 8,000 genes, Validate on 2,000 genes

Result: Mean performance ± std across folds
```

**Key Point**: CV uses ALL available training data, NOT a subset!

### 2. Final Model Training

**Purpose**: Create the production model for inference

```
Train on ALL 10,000 genes → model_multiclass.pkl
```

**Key Point**: Use ALL training data to maximize learning

### 3. Post-Training Evaluation

Since we've used all data for training, we need different strategies:

#### Option A: Initial Train/Test Split (Recommended)
```
Initial split: 80% train (8,000 genes), 20% test (2,000 genes)
- CV: Use only the 8,000 training genes
- Final model: Train on same 8,000 genes
- Evaluation: Test on holdout 2,000 genes
```

#### Option B: External Test Dataset
```
- CV: Use entire training dataset
- Final model: Train on entire training dataset  
- Evaluation: Separate test dataset (if available)
```

#### Option C: Repeated Holdouts for Benchmarking
```
Repeat 5 times:
  - Random 80/20 split
  - Train model on 80%
  - Evaluate on 20%
Report: Mean ± std performance
```

## Workflow Modes

### Testing Mode (`--sample-genes < 100`)
- **Purpose**: Quick development/debugging
- **CV**: Small subset (e.g., 10 genes)
- **Training**: Same small subset
- **Evaluation**: CV results only
- **USE CASE**: Development only, NOT production!

### Production Mode (default)
- **Purpose**: Real model training
- **CV**: 80% of all genes
- **Training**: Same 80% as CV
- **Evaluation**: 20% holdout test set
- **USE CASE**: Standard production training

### Research Mode (`--test-dataset` provided)
- **Purpose**: Academic evaluation
- **CV**: Entire training dataset
- **Training**: Entire training dataset
- **Evaluation**: Separate test dataset
- **USE CASE**: Paper benchmarks

### Benchmarking Mode (`--benchmark-mode`)
- **Purpose**: Rigorous evaluation
- **CV**: Multiple 80/20 splits
- **Training**: Ensemble of models
- **Evaluation**: Average across splits
- **USE CASE**: Thorough performance assessment

## Post-Training Analysis Strategies

Different analyses require different data strategies:

| Analysis | Data Strategy | Rationale |
|----------|--------------|-----------|
| **Feature Importance (SHAP)** | Sample 1-2K genes | Computationally expensive |
| **Overfitting Analysis** | Multiple small splits | Need train/val comparison |
| **Base vs Meta Comparison** | 5x random 80/20 splits | Statistical significance |
| **Threshold Optimization** | Small validation set (10%) | Prevent overfitting |

## Implementation TODO

1. **Immediate Fix**: Update help text for `--sample-genes` to clarify it's TESTING ONLY
2. **Short Term**: Implement proper train/test split at dataset loading
3. **Medium Term**: Add workflow mode selection (`--workflow-mode`)
4. **Long Term**: Support external test datasets for research

## Command Examples

### Testing (Quick Development)
```bash
python run_gene_cv_sigmoid.py \
  --dataset train_dataset \
  --out-dir test_run \
  --sample-genes 10  # TESTING ONLY!
```

### Production (Proper Training)
```bash
python run_gene_cv_sigmoid.py \
  --dataset train_dataset \
  --out-dir production_run \
  --train-test-split 0.8  # 80% train, 20% test
  # NO --sample-genes!
```

### Research (With External Test)
```bash
python run_gene_cv_sigmoid.py \
  --dataset train_dataset \
  --test-dataset test_dataset \
  --out-dir research_run
```

### Benchmarking (Rigorous Evaluation)
```bash
python run_gene_cv_sigmoid.py \
  --dataset full_dataset \
  --out-dir benchmark_run \
  --benchmark-mode \
  --n-benchmark-splits 5
```

## Key Takeaways

1. **`--sample-genes` is for TESTING ONLY**, not production
2. **CV should use ALL available training data** (or 80% if doing train/test split)
3. **Final model trains on the SAME data as CV** (consistency)
4. **Evaluation needs UNSEEN data** (via initial split, external test, or repeated holdouts)
5. **Post-training analyses can use subsets** (for computational efficiency)

## Current System Status

The system currently implements proper **final model training on all genes** but has issues with:
- Evaluation on training data (data leakage)
- Unclear purpose of `--sample-genes`
- No built-in train/test split mechanism

These issues will be addressed in the upcoming refactoring.



