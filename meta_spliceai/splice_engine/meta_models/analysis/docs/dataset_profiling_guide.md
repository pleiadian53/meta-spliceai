# Dataset Profiling System - Complete Guide

## Overview

The dataset profiling system provides comprehensive analysis and visualization capabilities for splice site prediction datasets. The system has been completely refactored from a monolithic script into modular, maintainable components while preserving and enhancing all functionality.

## Architecture

### Refactored Module Structure

The system is now organized into four focused modules:

```
analysis/
├── streaming_stats.py          # Statistics utilities (7KB)
├── dataset_visualizer.py       # Visualization functions (18KB)  
├── dataset_profiler_core.py    # Core profiling logic (28KB)
├── profile_dataset.py          # CLI driver (6KB)
└── profile_dataset_v0.py       # Original monolithic version (98KB)
```

### Module Responsibilities

#### 1. `streaming_stats.py`
- **StreamingStats Class**: Memory-efficient statistical calculations with median support
- **Feature Categorization**: Intelligent classification of dataset features
- **Helper Functions**: Column detection utilities (`find_splice_column`, `find_gene_column`)

#### 2. `dataset_visualizer.py`
- **Visualization Functions**: All plot creation and styling
- **Enhanced Plots**: Improved missing values, SpliceAI scores, and feature category visualizations
- **Consistent Styling**: Unified color schemes and layouts across all plots

#### 3. `dataset_profiler_core.py`
- **SpliceDatasetProfiler Class**: Main profiling orchestrator
- **Streaming Analysis**: Memory-efficient batch processing for large datasets
- **Comprehensive Reporting**: Summary statistics, recommendations, and console output

#### 4. `profile_dataset.py`
- **CLI Interface**: Clean argument parsing and user interaction
- **Driver Logic**: Orchestrates profiling workflow
- **Output Management**: JSON serialization and result organization

## Key Features

### Enhanced Statistical Analysis

#### Streaming Statistics with Median Support
- **Memory Efficiency**: Process datasets of any size without OOM errors
- **Reservoir Sampling**: Accurate median estimation for SpliceAI scores
- **Configurable Tracking**: Optional value storage for specific metrics

#### Intelligent Feature Categorization
- **Probability & Context Features**: Enhanced workflow features properly classified
- **Comprehensive Patterns**: Matches all relevant feature types from enhanced workflow
- **Clear Categories**: 9 distinct feature categories with meaningful names

### Advanced Visualizations

#### SpliceAI Scores Analysis (2x2 Subplot Layout)
```
┌─────────────────┬─────────────────┐
│ Mean by Type    │ Median by Type  │
├─────────────────┼─────────────────┤
│ Variability     │ Score Ranges    │
└─────────────────┴─────────────────┘
```

**Improvements:**
- Type-specific statistics (donor_score for donor sites, etc.)
- Median tracking for robust central tendency
- Variability measures for prediction confidence
- Full dynamic range analysis

#### Missing Values Analysis (Dual-Plot System)
```
┌─────────────────────────────────────┐
│ Missing Value Counts (Top 20)       │
├─────────────────────────────────────┤
│ Missing Percentages with Thresholds │
│ (5% and 10% threshold lines)        │
└─────────────────────────────────────┘
```

**Features:**
- Smart filtering to top 20 problematic columns
- Threshold visualization for quality assessment
- Comprehensive summary statistics
- Actionable insights in console output

## Usage Guide

### Basic Usage

```bash
# Profile a single parquet file with visualizations
python profile_dataset.py data/train_dataset.parquet --plots --output results/

# Profile a directory containing multiple parquet files
python profile_dataset.py data/train_pc_1000/master --plots --output results/

# Quick profiling without plots
python profile_dataset.py data/train_dataset.parquet --quiet

# Filter for specific genes (works with both files and directories)
python profile_dataset.py data/train_dataset.parquet \
    --genes ENSG00000139618,ENSG00000141510 \
    --plots --output results/
```

### Advanced Options

```bash
# Large dataset processing with sampling
python profile_dataset.py data/large_dataset/ \
    --max-files 10 \
    --sample-rows 50000 \
    --batch-size 100000 \
    --plots --output results/

# Memory-constrained environments
python profile_dataset.py data/dataset.parquet \
    --batch-size 50000 \
    --quiet \
    --output results/
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `dataset_path` | Path to single .parquet file OR directory containing .parquet files | Required |
| `--output, -o` | Output directory for results | `dataset_profile_results` |
| `--plots, -p` | Generate visualization plots | False |
| `--genes, -g` | Comma-separated gene IDs to filter | None |
| `--max-files` | Maximum parquet files to process | None |
| `--sample-rows` | Sample N rows per file | None |
| `--quiet, -q` | Suppress verbose output | False |
| `--batch-size` | Batch size for streaming | 100000 |

## Output Structure

```
results/
├── dataset_profile.json          # Complete profile data
└── visualizations/               # Generated plots
    ├── splice_distribution.png
    ├── spliceai_scores_statistics.png
    ├── missing_values_by_column.png
    ├── top_genes_by_occurrence.png
    └── feature_categories.png
```

## Integration Examples

### Programmatic Usage

```python
from meta_spliceai.splice_engine.meta_models.analysis.dataset_profiler_core import SpliceDatasetProfiler

# Initialize profiler
profiler = SpliceDatasetProfiler(verbose=True, batch_size=100000)

# Profile dataset
profile = profiler.profile_dataset(
    dataset_path="data/train_dataset.parquet",
    output_dir="results/",
    generate_plots=True,
    gene_filter=["ENSG00000139618", "ENSG00000141510"]
)

# Access results
print(f"Dataset size: {profile['basic_info']['sample_size']:,} samples")
print(f"Feature categories: {profile['feature_analysis']['feature_counts']}")
```

### Custom Analysis

```python
from meta_spliceai.splice_engine.meta_models.analysis.streaming_stats import StreamingStats, categorize_features

# Custom streaming statistics
stats = StreamingStats(track_values_for_median=True)
stats.update([1.0, 2.0, 3.0, 4.0, 5.0])
print(stats.get_stats())

# Feature categorization
columns = ['donor_score', 'acceptor_score', 'kmer_1', 'gene_id']
categories = categorize_features(columns)
print(categories['feature_counts'])
```

## Migration Guide

### From Original Script

The refactored system maintains full backward compatibility:

```bash
# Old usage (original v0 script)
python -m meta_spliceai.splice_engine.meta_models.analysis.profile_dataset_v0 \
    --dataset train_pc_1000/master \
    --output-dir results/ \
    --generate-plots

# New usage (refactored version)
python profile_dataset.py train_pc_1000/master \
    --output results/ \
    --plots
```

## Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce batch size
python profile_dataset.py data.parquet --batch-size 50000

# Sample data for testing
python profile_dataset.py data.parquet --sample-rows 10000
```

#### Large Dataset Processing
```bash
# Process subset for testing
python profile_dataset.py data/ --max-files 5 --sample-rows 1000

# Full processing with progress
python profile_dataset.py data/ --batch-size 50000 --verbose
```

## Development

### Adding New Visualizations

1. **Create Function**: Add to `dataset_visualizer.py`
2. **Register**: Add call in `create_visualizations_streaming()`
3. **Test**: Verify with sample data

```python
def create_new_plot(data: Dict[str, Any], vis_dir: Path) -> Optional[str]:
    """Create a new visualization plot."""
    try:
        # Plot creation logic
        plot_file = vis_dir / "new_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        return str(plot_file)
    except Exception as e:
        print(f"Error creating new plot: {e}")
        return None
```

### Adding New Statistics

1. **Extend StreamingStats**: Add new metrics to class
2. **Update Categorization**: Modify `categorize_features()` if needed
3. **Integrate**: Use in `SpliceDatasetProfiler` methods

### Testing

```bash
# Test with small dataset
python profile_dataset.py test_data.parquet --plots --output test_results/

# Validate output structure
ls test_results/
cat test_results/dataset_profile.json | jq '.basic_info'
```

---

*Last Updated: 2025-01-20*  
*Version: 2.0 (Refactored)*
