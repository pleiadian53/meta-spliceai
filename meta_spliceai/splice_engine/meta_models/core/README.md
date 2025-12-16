# Splice Surveyor Meta-Models Core

This module contains enhanced evaluation and workflow functions for splice site prediction using SpliceAI and other models, with a focus on comprehensive probability analysis and automatic annotation adjustment. **Meta-learning approaches are a key application of this package, enabling error rate reduction, confidence calibration, and multi-model integration.**

## Key Features

### 1. Comprehensive Probability Analysis
- Preserves all three probability scores from SpliceAI (donor, acceptor, neither) for every position
- Enables detailed analysis of model behavior and confidence levels
- Maintains transcript-splice site relationships throughout the workflow
- **Provides essential features for meta-learning approaches that aim to reduce error rates from foundation models**

### 2. Automatic Splice Site Adjustment Detection
- Automatically identifies systematic offsets in model predictions
- Applies strand-specific and site-specific adjustments to improve prediction accuracy
- Transforms raw probability scores into biologically meaningful predictions

### 3. Enhanced Error Analysis
- Detailed classification of true/false positives/negatives
- Configurable windows for error detection and analysis
- Flexible sampling of true negatives for balanced datasets
- **Balanced true negative sampling** for both donor and acceptor sites to create representative meta-model training data
- **Robust schema validation** to ensure consistent data types across all entries

## Core Components

### Directory Structure

The meta-models framework uses a well-defined directory structure to organize genomic data, intermediate files, and analysis outputs:

```
project_root/
├── data/                      # The global data directory
│   └── ensembl/               # Source-specific data (e.g., Ensembl)
│       ├── annotations.db     # Shared genomic annotations database
│       ├── overlapping_gene_counts.tsv # Shared overlapping gene metadata
│       ├── splice_sites.tsv   # Shared splice site annotations
│       ├── spliceai_eval/     # Evaluation directories
│       │   └── meta_models/   # Meta-models specific outputs (output_subdir)
│       └── spliceai_analysis/ # Analysis directories
├── meta_spliceai/           # Source code
└── scripts/                   # Utility scripts
```

#### Key Directory Concepts

- **shared_dir**: Global directory for shared reference datasets (annotations, splice sites, etc.)
  - Typically set to `data/ensembl`, which is typically the parent directory of `eval_dir` and `analysis_dir`
  - Shared across different analysis runs to avoid redundant storage

- **local_dir**: Parent directory for intermediate files and local outputs
  - It is usually derived from `eval_dir` (its parent directory) and, by default, coincides with `shared_dir`
  - Can be customized for subject-specific analyses

- **eval_dir**: Directory for evaluation-specific outputs
  - Default: `data/ensembl/spliceai_eval`
  - Contains results from prediction evaluation runs

- **analysis_dir**: Directory for analysis-specific outputs
  - Default: `data/ensembl/spliceai_analysis`
  - Contains analysis results and visualizations

- **output_subdir**: Subject-specific subdirectory within eval_dir
  - Default: `meta_models` 
  - Segregates outputs by subject matter (e.g., meta-model results)

This hierarchical structure enables efficient data sharing across analyses while maintaining clean organization of outputs by type and subject matter.

### Enhanced Evaluation

The `enhanced_evaluation.py` module contains the core functions for evaluating splice site predictions:

- `adjust_scores(scores, strand, splice_type)`: Applies standard SpliceAI adjustments to raw prediction scores
- `apply_custom_splice_site_adjustments(scores, strand, splice_type, adjustments)`: Applies custom, data-driven adjustments to prediction scores
- `enhanced_evaluate_donor_site_errors()`: Evaluates donor site predictions with all three probability scores
- `enhanced_evaluate_acceptor_site_errors()`: Evaluates acceptor site predictions with all three probability scores
- `enhanced_evaluate_splice_site_errors()`: Combined evaluation of both donor and acceptor sites

### Enhanced Workflow

The `enhanced_workflow.py` module provides a high-level API for the prediction and evaluation workflow:

- `enhanced_process_predictions_with_all_scores()`: Main function that processes predictions with all three probability scores and evaluates accuracy

## Usage Examples

### Basic Usage

```python
from meta_spliceai.splice_engine.meta_models.core.enhanced_workflow import enhanced_process_predictions_with_all_scores

# Load annotations and predictions
annotations_df = load_annotations()
predictions = predict_splice_sites_for_genes(gene_sequences)

# Process predictions with enhanced evaluation
error_df, positions_df = enhanced_process_predictions_with_all_scores(
    annotations_df,
    predictions,
    threshold=0.5,
    collect_tn=True,
    predicted_delta_correction=True
)

# Analyze results
print(f"Found {error_df.filter(pl.col('error_type') == 'FP').shape[0]} false positives")
print(f"Found {error_df.filter(pl.col('error_type') == 'FN').shape[0]} false negatives")
```

### Automatic Splice Site Adjustment Detection

```python
from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import auto_detect_splice_site_adjustments

# Detect optimal adjustments based on annotations and predictions
adjustments = auto_detect_splice_site_adjustments(annotations_df, predictions)
# Example output: {'donor': {'plus': 2, 'minus': 1}, 'acceptor': {'plus': 0, 'minus': -1}}

# Process predictions with auto-detected adjustments
error_df, positions_df = enhanced_process_predictions_with_all_scores(
    annotations_df,
    predictions,
    threshold=0.5,
    collect_tn=True,
    predicted_delta_correction=True,
    splice_site_adjustments=adjustments
)
```

### Complete Workflow Example

The `tri_score_and_auto_infer_splice_sites.py` example demonstrates a complete workflow:

1. Load gene sequences and annotations
2. Generate SpliceAI predictions
3. Automatically detect optimal splice site adjustments
4. Process predictions with and without adjustments
5. Calculate and compare prediction statistics
6. Visualize and save results

```python
# Load gene sequences
gene_sequences = load_gene_sequences_for_targets(
    target_genes=["ENSG00000104435", "ENSG00000130477"],
    prioritized_chromosomes=["19", "8"]
)

# Generate predictions
predictions = predict_splice_sites_for_genes(gene_sequences)

# Auto-detect optimal adjustments
adjustments = auto_detect_splice_site_adjustments(annotations_df, predictions)
print(f"Detected adjustment offsets:")
print(f"  Donor sites on 'plus' strand: offset = {adjustments['donor']['plus']}")
print(f"  Donor sites on 'minus' strand: offset = {adjustments['donor']['minus']}")
print(f"  Acceptor sites on 'plus' strand: offset = {adjustments['acceptor']['plus']}")
print(f"  Acceptor sites on 'minus' strand: offset = {adjustments['acceptor']['minus']}")

# Process without adjustments
error_df_without_adj, positions_df_without_adj = enhanced_process_predictions_with_all_scores(
    annotations_df,
    predictions,
    threshold=0.3,
    collect_tn=True,
    predicted_delta_correction=False
)

# Process with adjustments
error_df_with_adj, positions_df_with_adj = enhanced_process_predictions_with_all_scores(
    annotations_df,
    predictions,
    threshold=0.3,
    collect_tn=True,
    predicted_delta_correction=True,
    splice_site_adjustments=adjustments
)

# Compare results
stats_without_adj = calculate_prediction_statistics(positions_df_without_adj)
stats_with_adj = calculate_prediction_statistics(positions_df_with_adj)
print(f"Prediction accuracy without adjustments: {stats_without_adj['accuracy']:.2f}%")
print(f"Prediction accuracy with adjustments: {stats_with_adj['accuracy']:.2f}%")
```

## Parameter Reference

### enhanced_process_predictions_with_all_scores

```python
def enhanced_process_predictions_with_all_scores(
    ss_annotations_df,
    predictions,
    threshold=0.5,
    consensus_window=2,
    error_window=500,
    analyze_position_offsets=False,
    collect_tn=True,
    predicted_delta_correction=False,
    splice_site_adjustments=None,
    verbose=1,
    **kwargs
):
    """
    Process SpliceAI predictions with enhanced evaluation approach that preserves
    all three probability scores (donor, acceptor, neither).
    
    Parameters:
    ----------
    ss_annotations_df : pl.DataFrame
        Splice site annotations with columns: chrom, start, end, strand, site_type, gene_id, transcript_id
    predictions : Dict[str, Any]
        Output from predict_splice_sites_for_genes(), containing per-nucleotide probabilities
        Must include 'donor_prob', 'acceptor_prob', and 'neither_prob' for each gene
    threshold : float, optional
        Threshold for classifying a prediction as a splice site, by default 0.5
    consensus_window : int, optional
        Window size for matching predicted sites to true sites, by default 2
    error_window : int, optional
        Window size for extracting regions around errors, by default 500
    analyze_position_offsets : bool, optional
        Whether to analyze positional offsets in predictions, by default False
    collect_tn : bool, optional
        Whether to collect true negative positions, by default True
    predicted_delta_correction : bool, optional
        Whether to apply systematic prediction adjustments, by default False
    splice_site_adjustments : Dict[str, Dict[str, int]], optional
        Custom adjustments to apply to predictions, by default None
    verbose : int, optional
        Verbosity level, by default 1
    
    Returns:
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        Tuple containing (error_df, positions_df)
        
        - error_df: DataFrame with error analysis
        - positions_df: DataFrame with all positions and their three probabilities
    """
```

#### True Negative Sampling Parameters

The enhanced evaluation functions provide advanced control over true negative sampling for balanced datasets:

- **collect_tn** (bool): When set to True, the functions will collect true negative positions from the data. This is essential for training meta-models since you need examples of non-splice site positions.

- **tn_sample_factor** (float): Controls how many true negatives to sample relative to the total count of TPs, FPs, and FNs. For example, if there are 100 combined TP+FP+FN positions and tn_sample_factor=1.2, up to 120 TN positions will be sampled.

- **tn_sampling_mode** (str): Determines the strategy for sampling true negatives:
  - "random": Selects true negatives randomly from all available TN positions
  - "proximity": Prefers TN positions that are closer to true splice sites or false predictions, which can be more informative for meta-model training

- **tn_proximity_radius** (int): When using proximity sampling, this parameter defines the preferred radius around key positions (TPs, FPs, FNs) for selecting TNs.

These parameters ensure that the resulting dataset is well-balanced and representative, which is crucial for training effective meta-models that can distinguish between splice sites and non-splice sites.

#### Schema Consistency for Meta-Model Training

The enhanced evaluation functions include robust schema validation to ensure consistent data structures, which is critical for machine learning applications:

- All positions (TP, FP, FN, TN) maintain the same schema with identical field names and data types
- Empty values and None values are handled consistently across all entry types
- Type conversion ensures compatible types (e.g., empty strings instead of None for IDs)
- Explicit schema definition prevents inference errors when combining data from multiple sources

This consistency is particularly important when:
1. Combining donor and acceptor site predictions
2. Merging results from multiple genes
3. Building training datasets for meta-models 
4. Analyzing position statistics across different splice types

## Advanced Applications

### Meta-Model Approach

The preserved probability scores enable building meta-models that combine multiple prediction models:

```python
# Example of creating a meta-feature dataset
meta_features = positions_df.select([
    'gene_id', 'position', 'splice_type',
    'donor_score', 'acceptor_score', 'neither_score',
    # Add other model scores or features
])

# Train a meta-model on these features
from sklearn.ensemble import RandomForestClassifier
meta_model = RandomForestClassifier()
X = meta_features.select(['donor_score', 'acceptor_score', 'neither_score']).to_numpy()
y = (meta_features['splice_type'] == 'donor').to_numpy().astype(int)
meta_model.fit(X, y)
```

### Custom Splice Site Adjustments

For datasets with known biases, you can specify custom adjustments:

```python
# Custom adjustments based on prior knowledge
custom_adjustments = {
    'donor': {'plus': 2, 'minus': 1},
    'acceptor': {'plus': 0, 'minus': -1}
}

# Apply custom adjustments
error_df, positions_df = enhanced_process_predictions_with_all_scores(
    annotations_df,
    predictions,
    splice_site_adjustments=custom_adjustments,
    predicted_delta_correction=True
)

```

## Meta-Learning Applications

The core functionality of this package is designed to support **meta-learning approaches for splice site prediction**. By preserving all three probability scores from foundation models like SpliceAI, we enable:

1. **Error Rate Reduction**: Meta-models can learn patterns in the probability distributions that correlate with foundation model errors
2. **Confidence Calibration**: Adjusting probability scores based on patterns learned from training data
3. **Multi-Model Integration**: Combining predictions from multiple foundation models with different strengths
4. **Feature Enrichment**: Using the probability scores as features alongside other genomic context information

A typical **meta-learning workflow** includes:

1. Extracting prediction probabilities from foundation models using the enhanced workflow
2. Identifying patterns in probability distributions that correlate with errors
3. Training a meta-model (e.g., Random Forest, Neural Network) on these features
4. Applying the meta-model to new predictions to improve accuracy

This approach has shown significant improvements in reducing both false negatives and false positives in splice site prediction tasks.
