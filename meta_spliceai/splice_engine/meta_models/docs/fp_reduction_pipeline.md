# FP Reduction Analysis Pipeline

This document describes the False Positive (FP) reduction analysis pipeline implemented in the MetaSpliceAI meta-models module.

## Overview

The FP reduction pipeline addresses the problem of splice site false positives by:

1. Identifying genes with high numbers of false-negative predictions.
2. Generating enhanced context-aware features for these genes.
3. Analyzing the potential to rescue these false negatives using well-defined, context-specific features.
4. Exploring representative FN examples with detailed visualizations to demonstrate potential rescue effectiveness.

## Pipeline Structure

The pipeline is implemented in `meta_models/analysis/run_fp_reduction_pipeline.py` and follows a three-step process:

1. **Gene Identification**: Find genes with the highest numbers of false positives
2. **Enhanced Feature Generation**: Run the enhanced splice prediction workflow on the identified genes
3. **Reduction Analysis**: Analyze the potential to filter FPs using enhanced context-aware features

## Usage

```bash
python -m meta_spliceai.splice_engine.meta_models.analysis.run_fp_reduction_pipeline [OPTIONS]
```

### Command-Line Options

- `--top-genes N`: Analyze the top N genes with highest FP counts (default: 5)
- `--output-dir PATH`: Custom output directory (default: auto-generated timestamped directory)
- `--skip-workflow`: Skip running the enhanced workflow if you've already run it
- `--force-extraction`: Force extraction of genomic data even if files already exist
- `--mode {gene|transcript}`: Mode for sequence extraction (default: gene)
- `--seq-type {full|minmax}`: Type of gene sequences to extract (default: full)
- `-v` or `--verbose`: Increase verbosity (use -vv or -vvv for more details)
- `--gene-types LIST`: Filter to specific gene types (e.g., protein_coding, lncRNA)
- `--detailed-counts`: Use detailed splice site counting metrics for gene ranking
- `--full-dataset`: Analyze the full aggregated dataset across all genes

### Output Directory Structure

The pipeline creates an organized output directory with the following structure:

```
fp_reduction_analysis/
├── 1_gene_identification/  # Contains the CSV with top FP genes
│   ├── by_gene_type/       # Analysis by gene type
│   ├── donor_fps/          # Analysis of donor-specific FPs
│   └── acceptor_fps/       # Analysis of acceptor-specific FPs
├── 2_enhanced_workflow/    # Contains all workflow outputs and enhanced data
└── 3_reduction_analysis/   # Contains the final reduction analysis results
```

## Smart Resource Management

The pipeline includes intelligent resource management:

1. **Genomic Data Detection**: Automatically detects if key genomic files already exist
2. **Selective Extraction**: Only extracts data that doesn't already exist, avoiding redundant processing
3. **Override Capability**: Provides `--force-extraction` flag to regenerate all files if needed

## Algorithm Flow

The pipeline supports multiple analysis modes:

### Full Dataset Mode
1. Load full positions dataset from the shared directory
2. Identify genes with the highest FP counts (separately for donor and acceptor sites)
3. Analyze FP reduction potential across selected genes
4. Generate gene-specific reports and visualizations

### Standard Analysis Mode
1. Load positions data for a specific dataset
2. Analyze prediction type distribution
3. Evaluate FP reduction potential with detailed rule analysis
4. Explore promising FP examples with visualizations

### Cross-Gene Analysis
1. Check if genomic files exist to determine which extraction steps to run
2. Run the enhanced workflow on the selected genes
3. Analyze the FP reduction potential using enhanced context features
4. Generate comparative visualizations across genes

## Key Functions

- `load_full_positions_data()`: Loads dataset with fallback to shared directory if needed
- `analyze_genes_with_most_fps()`: Identifies genes with highest FP counts
- `analyze_genes_with_most_fps_by_type()`: Analyzes FPs by gene type and splice site type
- `run_enhanced_splice_prediction_workflow()`: Generates enhanced features
- `analyze_fp_reduction_potential_by_gene()`: Analyzes FP reduction potential for specific genes
- `identify_fp_filter_rules()`: Discovers potential rules for filtering FPs
- `explore_promising_fp_examples()`: Generates visualizations for representative FP examples

## Example Visualization and Analysis

The pipeline generates several types of visualizations to understand FP patterns:

### Feature Distribution Analysis
- Histograms comparing FP vs TP distributions for key features
- Boxplots showing statistical differences between prediction types
- Correlation heatmaps to identify feature relationships

### Probability Context Visualization
- Plots showing donor and acceptor probabilities at positions surrounding FPs
- Highlights the characteristic patterns of filterable FPs
- Marks important thresholds and decision boundaries

### Rule Effectiveness Analysis
- Quantifies the potential improvement from each filter rule
- Shows the percentage of FPs that could be filtered
- Analyzes rule overlap and importance

## FP Filter Rules

The analysis identifies several types of filter rules:

1. **Peak Height Ratio**: FPs often have weak peaks compared to surrounding positions
2. **Local Peak Status**: True positives should be local maxima in the probability landscape
3. **Signal Strength**: FPs tend to have weaker overall signal strength
4. **Peak Shape**: FPs often have irregular peak shapes with poor second derivatives
5. **Context Differential**: FPs typically have poor differentiation from surrounding context
6. **Cross-Type Evidence**: FPs may show competing signals between donor and acceptor probabilities

## Integration with Meta-Models

The FP reduction rules can be integrated into meta-models to improve overall prediction accuracy:

1. As a post-processing filter on base model predictions
2. As features for training more sophisticated meta-models
3. As weights in ensemble prediction systems

This approach complements the FN rescue pipeline, allowing for a comprehensive improvement of splice site prediction accuracy by both reducing false positives and rescuing false negatives.
