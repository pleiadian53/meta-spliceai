# Deep Error Model Subpackage

A modular deep learning framework for analyzing and correcting splice site prediction errors using transformer-based architectures and Integrated Gradients analysis.

## Overview

This subpackage implements position-centric error analysis for splice site predictions, leveraging:

- **Position-Centric Data Representation**: Each genomic nucleotide position as an individual training instance
- **Transformer-Based Models**: Fine-tuned DNABERT-like models for binary error classification
- **Integrated Gradients Analysis**: Token-level interpretability for understanding model decisions
- **Comprehensive Visualizations**: Rich plotting tools for attribution analysis

## Architecture

```
error_model/
├── config.py                     # Configuration classes
├── dataset/                      # Data preparation and handling
│   ├── dataset_preparer.py      # Position-centric dataset preparation
│   └── data_utils.py            # PyTorch datasets and utilities
├── modeling/                     # Model training and analysis
│   ├── transformer_trainer.py   # Multi-modal transformer training
│   └── ig_analyzer.py           # Integrated Gradients analysis
├── visualization/                # Plotting and visualization
│   ├── alignment_plot.py        # Token-level attribution plots
│   └── frequency_plot.py        # Token frequency analysis plots
└── examples/                     # Usage examples
    └── ig_analysis_example.py   # Complete IG analysis workflow
```

## Key Features

### 1. Position-Centric Dataset Preparation

```python
from meta_spliceai.splice_engine.meta_models.error_model import ErrorDatasetPreparer, ErrorModelConfig

config = ErrorModelConfig(
    context_length=200,  # ±100 nt around each position
    labels=['FP_vs_TP', 'FN_vs_TP'],
    split_strategy='gene_level'
)

preparer = ErrorDatasetPreparer(config)
dataset_info = preparer.prepare_dataset(data_dir)
```

### 2. Multi-Modal Transformer Training

```python
from meta_spliceai.splice_engine.meta_models.error_model import TransformerTrainer

trainer = TransformerTrainer(config)
trainer.train(
    train_dataset=dataset_info['datasets']['train'],
    val_dataset=dataset_info['datasets']['val'],
    output_dir=output_dir
)
```

### 3. Integrated Gradients Analysis

```python
from meta_spliceai.splice_engine.meta_models.error_model import IGAnalyzer

ig_analyzer = IGAnalyzer(model, tokenizer, config)

# Compute attributions
attributions = ig_analyzer.compute_attributions(
    sequences=sequences,
    labels=labels,
    additional_features=features
)

# Analyze error patterns
analysis_results = ig_analyzer.analyze_error_patterns(attributions)
```

### 4. Comprehensive Visualizations

```python
from meta_spliceai.splice_engine.meta_models.error_model import visualization

# Token frequency comparison
plotter = visualization.FrequencyPlotter()
fig = plotter.plot_token_frequency_comparison(analysis_results)

# Sequence alignment with attributions
alignment_plotter = visualization.AlignmentPlotter()
fig = alignment_plotter.plot_sequence_attribution(sequence, tokens, attributions)
```

## Data Requirements

The error model expects position-centric training data from meta-model artifacts:

### Required Files
- `analysis_sequences_*.parquet`: Genomic sequences with context
- `splice_positions_enhanced_*.parquet`: Position annotations with features

### Expected Data Structure
```python
# analysis_sequences_*
columns = ['chrom', 'start', 'end', 'strand', 'gene_id', 'transcript_id', 'sequence']

# splice_positions_enhanced_*
columns = ['chrom', 'position', 'strand', 'gene_id', 'transcript_id', 'site_type', 
          'donor_score', 'acceptor_score', 'neither_score', 'entropy', 'max_ratio']
```

## Configuration

### ErrorModelConfig

```python
@dataclass
class ErrorModelConfig:
    # Data parameters
    context_length: int = 200           # Total sequence context (±100 nt)
    labels: List[str] = field(default_factory=lambda: ['FP_vs_TP', 'FN_vs_TP'])
    
    # Model parameters
    model_name: str = "zhihan1996/DNABERT-2-117M"
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    
    # Training parameters
    split_strategy: str = 'gene_level'   # 'gene_level' or 'position_level'
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Device configuration
    device: str = 'auto'                # 'auto', 'cpu', 'cuda'
```

### IGAnalysisConfig

```python
@dataclass
class IGAnalysisConfig:
    steps: int = 50                     # IG integration steps
    batch_size: int = 8                 # IG computation batch size
    baseline: str = "zero"              # "zero", "mask", or "random"
    top_k_tokens: int = 20              # Top tokens to analyze
```

## Usage Examples

### Complete Training and Analysis Workflow

```python
#!/usr/bin/env python3
import torch
from pathlib import Path
from meta_spliceai.splice_engine.meta_models.error_model import (
    ErrorModelConfig, ErrorDatasetPreparer, TransformerTrainer, IGAnalyzer
)

# Configuration
config = ErrorModelConfig(
    context_length=200,
    model_name="zhihan1996/DNABERT-2-117M",
    batch_size=16,
    num_epochs=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Data preparation
data_dir = Path("data/ensembl/spliceai_eval/meta_models")
preparer = ErrorDatasetPreparer(config)
dataset_info = preparer.prepare_dataset(data_dir)

# Model training
trainer = TransformerTrainer(config)
training_results = trainer.train(
    train_dataset=dataset_info['datasets']['train'],
    val_dataset=dataset_info['datasets']['val'],
    output_dir=Path("output/error_model")
)

# IG Analysis
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

ig_analyzer = IGAnalyzer(trainer.model, tokenizer, config)

# Prepare test data
test_sequences = [sample['sequence'] for sample in dataset_info['datasets']['test'][:50]]
test_labels = [sample['label'] for sample in dataset_info['datasets']['test'][:50]]

# Compute attributions
attributions = ig_analyzer.compute_attributions(test_sequences, test_labels)

# Analyze patterns
analysis_results = ig_analyzer.analyze_error_patterns(attributions)

# Save results
ig_analyzer.save_results(attributions, analysis_results, Path("output/ig_analysis"))
```

### Command-Line IG Analysis

```bash
python meta_spliceai/splice_engine/meta_models/error_model/examples/ig_analysis_example.py \
    --model_path output/error_model/best_model.pt \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/ig_analysis \
    --max_samples 100
```

## Integration with Meta-Models

The error model integrates seamlessly with the existing meta-models framework:

### Data Flow
1. **Meta-Model Training** → Position-centric artifacts (`analysis_sequences_*`, `splice_positions_enhanced_*`)
2. **Error Dataset Preparation** → Binary classification datasets (FP vs TP, FN vs TP)
3. **Transformer Training** → Fine-tuned error classification models
4. **IG Analysis** → Token-level interpretability insights
5. **Error Correction** → Improved splice site predictions

### Compatibility
- Uses existing `MetaModelDataHandler` for data loading
- Compatible with `ModelEvaluationFileHandler` for evaluation
- Integrates with meta-model training workflows
- Maintains position-centric data representation

## Visualization Gallery

The subpackage provides rich visualizations for interpretability:

### 1. Token Frequency Comparison
- Absolute and relative frequency comparisons
- Error vs correct prediction patterns
- Frequency ratio analysis

### 2. Attribution Alignment Plots
- Sequence-aligned token attributions
- Color-coded importance visualization
- Individual sequence analysis

### 3. Comparative Analysis
- Side-by-side error vs correct examples
- Statistical distribution comparisons
- Positional pattern analysis

### 4. Comprehensive Reports
- Multi-panel analysis dashboards
- Summary statistics and insights
- Exportable visualization suites

## Performance Considerations

### Memory Usage
- Use batch processing for large datasets
- Configure appropriate batch sizes for IG computation
- Consider sequence length limits for memory efficiency

### Computational Requirements
- GPU recommended for transformer training
- IG analysis is computationally intensive
- Parallel processing supported for large-scale analysis

### Scalability
- Modular design supports different model architectures
- Configurable context lengths and feature sets
- Extensible to additional error types and classifications

## Future Extensions

### Planned Features
1. **Multi-Class Error Analysis**: Support for more granular error classifications
2. **Ensemble Methods**: Integration with multiple base models
3. **Real-Time Analysis**: Streaming IG analysis for large datasets
4. **Advanced Visualizations**: Interactive plots and 3D attribution maps

### Integration Opportunities
1. **Clinical Workflows**: Integration with variant analysis pipelines
2. **Research Applications**: Support for custom splice site databases
3. **Model Comparison**: Cross-model attribution analysis
4. **Automated Reporting**: Generated insights and recommendations

## Dependencies

### Core Dependencies
- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- Captum >= 0.5.0
- Polars >= 0.18.0
- Pandas >= 1.5.0

### Visualization Dependencies
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0
- NumPy >= 1.21.0

### Optional Dependencies
- CUDA toolkit (for GPU acceleration)
- Jupyter (for interactive analysis)

## Contributing

When contributing to the error model subpackage:

1. **Maintain Modularity**: Keep components loosely coupled
2. **Follow Conventions**: Use existing naming and structure patterns
3. **Add Tests**: Include unit tests for new functionality
4. **Update Documentation**: Keep README and docstrings current
5. **Consider Performance**: Optimize for large-scale analysis

## Support

For questions and issues:

1. **Documentation**: Check this README and inline docstrings
2. **Examples**: Review example scripts and usage patterns
3. **Integration**: Consult meta-models framework documentation
4. **Performance**: Consider computational requirements and optimization

---

This subpackage provides a complete framework for deep error analysis in splice site prediction, combining state-of-the-art transformer models with interpretability analysis for actionable insights into model behavior and error patterns.
