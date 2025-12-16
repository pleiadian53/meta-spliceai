# FN Rescue Analysis Pipeline

This document describes the False Negative (FN) rescue analysis pipeline implemented in the MetaSpliceAI meta-models module.

## Overview

The FN rescue pipeline addresses the problem of splice site false negatives by:

1. Identifying genes with high numbers of false negative predictions
2. Generating enhanced features for these genes
3. Analyzing the potential to rescue these false negatives using context-aware features
4. Exploring promising FN examples that could be rescued with detailed visualizations

## Pipeline Structure

The pipeline is implemented in `meta_models/analysis/run_fn_rescue_pipeline.py` and follows a three-step process:

1. **Gene Identification**: Find genes with the highest numbers of false negatives
2. **Enhanced Feature Generation**: Run the enhanced splice prediction workflow on the identified genes
3. **Rescue Analysis**: Analyze the potential to rescue FNs using enhanced context-aware features

## Usage

```bash
python -m meta_spliceai.splice_engine.meta_models.analysis.run_fn_rescue_pipeline [OPTIONS]
```

### Command-Line Options

- `--top-genes N`: Analyze the top N genes with highest FN counts (default: 5)
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
fn_rescue_analysis/
├── 1_gene_identification/  # Contains the CSV with top FN genes
├── 2_enhanced_workflow/    # Contains all workflow outputs and enhanced data
└── 3_rescue_analysis/      # Contains the final rescue analysis results
```

## Smart Resource Management

The pipeline includes intelligent resource management:

1. **Genomic Data Detection**: Automatically detects if key genomic files already exist
2. **Selective Extraction**: Only extracts data that doesn't already exist, avoiding redundant processing
3. **Override Capability**: Provides `--force-extraction` flag to regenerate all files if needed

## Implementation Details

### Directory Handling with Modern Python Paths

The pipeline uses Python's `pathlib.Path` for modern, intuitive path manipulation:

```python
from pathlib import Path

# Create the main output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Create structured subdirectories using the / operator
identification_dir = output_dir / "1_gene_identification"
workflow_dir = output_dir / "2_enhanced_workflow"
analysis_dir = output_dir / "3_rescue_analysis"

# Create all directories
for directory in [identification_dir, workflow_dir, analysis_dir]:
    directory.mkdir(exist_ok=True)
```

This approach is preferred over traditional `os.path.join()` as it's:
- More intuitive (resembles filesystem paths visually)
- Platform-independent (works correctly on all operating systems)
- Object-oriented (provides methods like `mkdir()`, `exists()`, etc.)
- Type-safe (path objects know they are paths, unlike raw strings)

### Comprehensive Genomic Sequence File Detection

The pipeline has sophisticated logic to detect genomic sequence files following multiple patterns:

```python
def check_genomic_files_exist(mode='gene', seq_type='full'):
    """
    Check if the main genomic data files already exist to avoid extraction.
    
    Parameters
    ----------
    mode : str, default='gene'
        Mode for sequence extraction ('gene' or 'transcript')
    seq_type : str, default='full'
        Type of gene sequences to extract ('full' or 'minmax')
    """
    # Basic files check
    
    # Define all potential sequence patterns
    sequence_patterns = {
        "gene_regular": "gene_sequence_{chrom}.{fmt}",      # Regular gene sequences
        "gene_minmax": "gene_sequence_minmax_{chrom}.{fmt}", # Minmax gene sequences
        "transcript": "tx_sequence_{chrom}.{fmt}"           # Transcript sequences
    }
    
    # Select the appropriate pattern based on mode and seq_type
    if mode == 'transcript':
        patterns_to_check = {"transcript": sequence_patterns["transcript"]}
    else:  # mode == 'gene'
        if seq_type == 'minmax':
            patterns_to_check = {"gene_minmax": sequence_patterns["gene_minmax"]}
        else:  # seq_type == 'full'
            patterns_to_check = {"gene_regular": sequence_patterns["gene_regular"]}
    
    # Check each pattern against multiple formats
    formats = ['parquet', 'tsv', 'csv', 'pkl']
    
    # Ensure all chromosomes (1-22, X, Y) exist for at least one pattern+format combination
    standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
```

This approach ensures we only check mutually exclusive file patterns based on the sequence parameters:

1. **Gene Mode, Full Type** (default): Checks only `gene_sequence_1.parquet`, etc.
2. **Gene Mode, Minmax Type**: Checks only `gene_sequence_minmax_1.parquet`, etc.
3. **Transcript Mode**: Checks only `tx_sequence_1.parquet`, etc.

The detection respects the same mode and sequence type parameters used by the workflow, ensuring logical consistency between detection and extraction processes.

Output example with mode=gene, seq_type=full:
```
Checking for existing genomic files:
  ✓ Found: annotations
  ✓ Found: splice_sites
  ✓ Found: genomic_sequences
    - Complete set of gene_regular files exists (parquet)
```

## Enhanced vs. Shared Directory Logic

The pipeline aligns with MetaSpliceAI's data organization principles:

1. **Shared Data**: Common genomic resources used across multiple analysis modules
   - Located in: `meta-spliceai/data/ensembl/spliceai_eval/`
   - Accessed using: `use_shared_dir=True` in the data handler

2. **Enhanced/Subject-Specific Data**: Data related to specific analysis modules
   - Located in: `meta-spliceai/data/ensembl/spliceai_eval/meta_models/`
   - Default directory used by the data handler

This organization enables graceful fallback to the shared data when enhanced features
are not available for all genes, while keeping the subject-specific enhanced features
in their own directory.

## Algorithm Flow

The pipeline supports multiple analysis modes:

### Full Dataset Mode
1. Load full positions dataset from the shared directory
2. Identify genes with the highest FN counts
3. Analyze FN rescue potential across selected genes
4. Generate gene-specific reports and visualizations

### Standard Analysis Mode
1. Load positions data for a specific dataset
2. Analyze prediction type distribution
3. Evaluate FN rescue potential with detailed rule analysis
4. Explore promising FN examples with visualizations

### Cross-Gene Analysis
1. Check if genomic files exist to determine which extraction steps to run
2. Run the enhanced workflow on the selected genes
3. Analyze the FN rescue potential using enhanced context features
4. Generate comparative visualizations across genes

## Key Functions

- `load_full_positions_data()`: Loads dataset with fallback to shared directory if needed
- `analyze_genes_with_most_fns()`: Identifies genes with highest FN counts
- `run_enhanced_splice_prediction_workflow()`: Generates enhanced features
- `analyze_fn_rescue_potential_by_gene()`: Analyzes FN rescue potential
- `identify_fn_rescue_rules()`: Discovers potential rules for rescuing FNs
- `explore_promising_fn_examples()`: Generates visualizations for representative FN examples

## Example Visualization and Analysis

The pipeline generates several types of visualizations to understand FN patterns:

### Feature Distribution Analysis
- Histograms comparing FN vs TN distributions for key features
- Boxplots showing statistical differences between prediction types
- Correlation heatmaps to identify feature relationships

### Probability Context Visualization
- Plots showing donor and acceptor probabilities at positions surrounding FNs
- Highlights the characteristic patterns of rescuable FNs
- Marks important thresholds and decision boundaries

### Rule Effectiveness Analysis
- Quantifies the potential improvement from each rescue rule
- Shows the percentage of FNs that could be rescued
- Analyzes rule overlap and importance
