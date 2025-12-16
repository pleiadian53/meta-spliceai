# Meta Models for MetaSpliceAI

This package enhances splice-site prediction capabilities using meta-models built upon predictions from foundational splice site predictors such as SpliceAI. By incorporating contextual sequence information and genomic features, the meta-learning layer corrects predictive errors (e.g., false positives and false negatives) from the base model and effectively adapts to the complexity of alternative splicing under diverse biological conditions such as disease states and treatments. Context-dependent alternative splicing can be captured by different "adaptor models" within the meta-learning layer, enabling precise recalibration of base-model predictions.

## Overview

The meta_models package offers:

1. **Three-probability scoring system** - Preserves donor, acceptor, and neither probabilities for comprehensive analysis.
2. **Automatic splice site adjustments** - Detects and corrects systematic offsets in model predictions
3. **Enhanced genomic context integration** - Incorporates comprehensive features such as k-mer frequencies, gene-level statistics, and regulatory elements. And, in the case of the deep learning models, the sequence context is also included.
4. **Efficient chromosome-specific processing** - Optimizes large-scale genomic analyses through chromosome-specific data loading.
5. **Cross-framework compatibility** - Operates seamlessly with both Pandas and Polars dataframes.
6. **Type-safe configurations** - Uses dataclasses for robust parameter validation and management
7. **Derived probability features** - Calculates informative metrics like entropy and probability ratios for meta-model training.
8. **Design Blueprint** - [META_MODEL_DESIGN.md](META_MODEL_DESIGN.md) documents architecture decisions, training strategy, and future roadmap.
9. **Comprehensive Documentation & Examples** - Extensive guidelines, workflow explanations, and practical scripts.    

## Package Structure

```
meta_models/
├─ __init__.py
├─ core/                         ← "building blocks"
│  ├─ analyzers/                     - Specialized analyzers package
│  │  ├─ base.py                     - Base Analyzer with shared functionality
│  │  ├─ splice.py                   - SpliceAnalyzer for splice site analysis
│  │  └─ feature.py                  - FeatureAnalyzer for genomic features
│  ├─ data_types.py                  - Configuration dataclasses
│  ├─ enhanced_evaluation.py         - Donor/acceptor evaluators + offset fix
│  └─ enhanced_workflow.py           - Orchestrator for compound feature derivation & probability transformations
├─ utils/
│  ├─ dataframe_utils.py             - Cross-framework (Pandas/Polars) compatibility
│  ├─ sequence_utils.py              - Sequence data handling with chromosome-specificity
│  ├─ chrom_utils.py                 - Chromosome selection and filtering
│  ├─ feature_utils.py               - Merge/join helpers (k-mers + gene stats)
│  ├─ model_utils.py                 - SpliceAI ensemble loading, probability floor
│  ├─ infer_splice_site_adjustments.py - Automatic detection of splice site position adjustments
│  ├─ analyze_splice_adjustment.py   - Visualization and analysis tools for position adjustments
│  ├─ verify_splice_adjustment.py    - Verification utilities for adjustment effects
│  └─ workflow_utils.py              - Alignment, timers, pretty prints
├─ workflows/
│  ├─ data_preparation.py            - Genomic data preparation functions
│  ├─ splice_prediction_workflow.py - End-to-end workflow with new architecture
│  ├─ splice_predictions.py  - Legacy workflow (maintained for compatibility)
│  └─ GENOMIC_DATA_PROCESSING.md     - Detailed documentation of genomic data processing steps
├─ io/
│  └─ handlers.py                    - Data loading and saving operations for meta-models
├─ docs/
│  ├─ dataclass_best_practices.md    - Guidelines for using dataclasses
│  └─ polars_best_practices.md       - Polars usage patterns and best practices
├─ tests/
│  ├─ test_overlapping_genes.py      - Tests for overlapping gene handling
│  └─ test_subset_genomic_sequences.py - Tests for sequence extraction
└─ examples/                        - Example scripts demonstrating key functionality
   ├─ README.md                      - Documentation of example scripts and features
   ├─ tri_score_and_auto_infer_splice_sites.py - Comprehensive meta-model pipeline demo
   ├─ tri_score_example.py           - Basic tri-probability score system
   ├─ enhanced_predictions_example.py - Enhanced prediction workflow basics
   └─ als_genes_prediction_example.py - Gene-specific prediction example
```

## Key New Components

| New Component | What it Does | Why it Matters |
|---------------|--------------|----------------|
| **core/analyzers/** | Specialized analyzer classes that centralize genomic data processing logic | Provides a clean, object-oriented API for working with different types of genomic data |
| **core/analyzers/splice.py** | SpliceAnalyzer centralizes all splice-annotation logic: loading GTFs, handling overlapping genes, auto-offset detection | Cleaner API → any workflow can call SpliceAnalyzer.adjust_coordinates() once and forget about GENCODE vs Ensembl quirks |
| **core/analyzers/feature.py** | Generates rich genomic and regulatory feature matrices | Simplifies feature engineering for meta-models |
| **core/enhanced_workflow.py** | Computes derived probability metrics for enhanced modeling | Provides informative inputs for accurate meta-prediction |
| **utils/dataframe_utils.py** | Provides a uniform interface for working with both Pandas and Polars dataframes | Ensures broad compatibility and ease of use |
| **utils/sequence_utils.py** | Handles efficient loading of chromosome-specific sequence files | Optimizes memory usage and speeds up processing by working with smaller, targeted datasets |
| **utils/infer_splice_site_adjustments.py** | Automatically identifies and corrects prediction offsets | Improves prediction accuracy by learning and correcting systematic biases |
| **utils/feature_utils.merge_contextual_features()** | Joins positions + probabilities with k-mer & gene features | Keeps the main workflow lightweight; no more ad-hoc joins |
| **utils/model_utils.load_spliceai_ensemble()** | Loads the 5-model SpliceAI ensemble once; handles CPU/GPU placement | 5-fold speed-up vs re-loading inside each chunk loop |
| **workflows/splice_prediction_workflow.py** | Refactored and optimized genome-scale prediction workflow | End-to-end, efficient and reliable large-scale genomic analysis |
| **examples/tri_score_and_auto_infer_splice_sites.py** | Genes-specific example of the meta-model pipeline | Demonstrates tri-score system with automatic adjustment detection |

## Directory Structure

The meta-models framework uses a well-defined directory structure to organize genomic data, intermediate files, and analysis outputs:

```
project_root/
├── data/                      # The global data directory
│   └── ensembl/               # Source-specific data (e.g., Ensembl)
│       ├── annotations.db     # Shared genomic annotations database
│       ├── overlapping_gene_counts.tsv # Shared overlapping gene metadata
│       ├── splice_sites.tsv   # Shared splice site annotations
│       ├── spliceai_eval/     # Evaluation directories
│       │   └── meta_models/   # Meta-models specific outputs
│       └── spliceai_analysis/ # Analysis directories
```

## Genomic Data Processing

The meta-models package implements a comprehensive genomic data processing pipeline as documented in `workflows/GENOMIC_DATA_PROCESSING.md`. Key steps include:

1. **Gene Annotation Preparation** - Extraction and normalization of gene features from GTF files
2. **Splice Site Annotation** - Identification of donor and acceptor sites from exon boundaries
3. **Genomic Sequence Extraction** - Retrieval of gene sequences from reference genome
4. **Overlapping Gene Handling** - Management of genes that share genomic coordinates
5. **Chromosome Determination** - Selection of chromosomes for focused processing
6. **Model Loading** - Initialization of SpliceAI deep learning models
7. **Position Adjustment Detection** - Automatic calibration of prediction coordinates

## Key Features

### 1. Comprehensive Probability Analysis
- Preserves all three probability scores from SpliceAI (donor, acceptor, neither) for every position
- Enables detailed analysis of model behavior and confidence levels
- Maintains transcript-splice site relationships throughout the workflow

### 2. Derived Probability Features
- **Entropy calculation** - Measures uncertainty in probability distributions
- **Probability ratios** - Compares different splice site types (donor:acceptor, splice:neither)
- **Normalized differences** - Captures relative strength of predictions
- **Log-odds transformations** - Provides statistical measures of prediction confidence
- These features form the foundation for training meta-models that correct base model errors

### 3. Automatic Splice Site Adjustment Detection
- Automatically identifies systematic offsets in model predictions
- Applies strand-specific and site-specific adjustments to improve prediction accuracy
- Transforms raw probability scores into biologically meaningful predictions
- Includes visualization and verification tools to validate adjustment effects

### 4. Enhanced Error Analysis
- Detailed classification of true/false positives/negatives
- Configurable windows for error detection and analysis
- Flexible sampling of true negatives for balanced datasets
- Integration with overlapping gene information for accurate evaluation

### 5. Memory-Efficient Genomic Data Processing
- Chromosome-specific file loading for reduced memory footprint
- Lazy evaluation using Polars for large dataframes
- Extensive file existence checks to prevent redundant computation

## Usage Example

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import run_enhanced_splice_prediction_workflow

# Run the enhanced splice prediction workflow
results = run_enhanced_splice_prediction_workflow(
    local_dir="/path/to/data",
    target_genes=["STMN2", "UNC13A"],  # Works with both gene names and Ensembl IDs
    do_extract_sequences=True,         # Extract sequence data
    do_extract_annotations=True,       # Extract gene annotations
    do_extract_splice_sites=True,      # Extract splice sites
    do_find_overlaping_genes=True,     # Find overlapping genes
    verbosity=1                        # Set verbosity level
)

# Access the results - check that the dataframes were successfully created
if "positions" in results and not results["positions"].is_empty():
    positions_df = results["positions"]
    print(f"Generated {positions_df.height} positions with derived probability features")
    
    # Example: Get positions with high entropy (uncertainty)
    uncertain_positions = positions_df.filter(positions_df["probability_entropy"] > 0.8)
    print(f"Found {uncertain_positions.height} positions with high prediction uncertainty")
    
    # Get positions with strong donor site predictions relative to acceptor
    donor_enriched = positions_df.filter(positions_df["relative_donor_probability"] > 0.7)
    print(f"Found {donor_enriched.height} positions with strong donor site characteristics")
```

## Additional Documentation

- **examples/README.md** - Detailed documentation of example scripts and their functionality
- **workflows/GENOMIC_DATA_PROCESSING.md** - Deep dive into the genomic data processing steps
- **[META_MODEL_DESIGN.md](META_MODEL_DESIGN.md)** - Design principles and architectural decisions
- **core/README.md** - Core components documentation
