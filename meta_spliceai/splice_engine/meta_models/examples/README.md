# Meta-Models Examples

This directory contains example scripts demonstrating key functionality of the MetaSpliceAI meta-models pipeline. These examples showcase how to use the enhanced workflows, feature generation, and model integration capabilities for improved splice site prediction.

## Key Example Scripts

### tri_score_and_auto_infer_splice_sites.py

**Primary Example Script** - Demonstrates the comprehensive meta-model pipeline for splice site prediction with automatic annotation adjustment detection.

#### Core Functionality

- Runs enhanced splice site prediction workflow using the tri-probability score system
- Automatically detects and applies splice site annotation adjustments
- Generates and uses derived probability features for meta-model training
- Provides detailed visualizations and comparisons of probability distributions

#### Key Features

- **Derived Probability Features** implemented in this pipeline:
  - **Relative Probabilities**:
    - `relative_donor_probability`: Donor score normalized relative to splice sites
    - `splice_probability`: Probability of any splice site vs. neither
  - **Comparative Measures**:
    - `donor_acceptor_diff`: Normalized difference between donor and acceptor scores
    - `splice_neither_diff`: Normalized difference between splice sites and neither
  - **Log-Odds Transformations**:
    - `donor_acceptor_logodds`: Log-odds of donor vs. acceptor
    - `splice_neither_logodds`: Log-odds of splice sites vs. neither
  - **Uncertainty Measurement**:
    - `probability_entropy`: Shannon entropy of the probability distribution

#### Usage

```python
# Basic execution
python -m meta_spliceai.splice_engine.meta_models.examples.tri_score_and_auto_infer_splice_sites

# Configure specific genes to analyze
python -m meta_spliceai.splice_engine.meta_models.examples.tri_score_and_auto_infer_splice_sites --gene-ids GENE1 GENE2
```

#### Implementation Notes

- Derived probability features are implemented in `meta_models/core/enhanced_workflow.py` (lines ~137-160)
- Automatic splice site adjustment detection is in `meta_models/utils/infer_splice_site_adjustments.py`
- This script subsumes and extends functionality from `tri_score_example.py`, adding robust adjustment detection

### tri_score_example.py

An earlier version focusing on the tri-probability score system without the automatic adjustment detection. Primarily used for reference, as `tri_score_and_auto_infer_splice_sites.py` supersedes this script.

### enhanced_predictions_example.py

Demonstrates basic usage of the enhanced prediction workflow without the tri-score system.

### enhanced_splice_sites_example.py

Shows how to work with enhanced splice site representations in the meta-model framework.

### als_genes_prediction_example.py

Example focusing on ALS-related gene predictions, showcasing the application of the meta-model framework to specific gene sets.

### training_data_example.py

Demonstrates how to generate training data for meta-models, including feature extraction and processing.

## Integration with MetaSpliceAI

These examples form a crucial part of the MetaSpliceAI pipeline's meta-model component, which improves on base SpliceAI predictions through:

1. Feature extraction and augmentation
2. Uncertainty quantification
3. Adjustment for systematic prediction biases
4. Enhanced visualization and analysis

The derived probability features (especially entropy and probability ratios) are essential for training meta-models that can correct errors in the base model predictions.
