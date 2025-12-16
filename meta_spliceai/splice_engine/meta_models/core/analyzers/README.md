# MetaSpliceAI Analyzers Package

This package contains specialized analyzer classes for genomic data processing and splice site analysis in the MetaSpliceAI project. It provides a modular approach to handling different types of genomic data and analysis tasks.

## Overview

The analyzers package is a refactored and expanded version of the original `core/analyzer.py` module, providing better separation of concerns and more specialized functionality for different analysis tasks. The original module is still available for backward compatibility, but this package subsumes all its functionality with improvements.

> **Note:** This package replaces the functionality previously defined in `core/analyzer.py`. The original module is maintained for backward compatibility only, as noted in April 2025.

## Package Structure

```
analyzers/
├── __init__.py      - Package exports and initialization
├── base.py          - Base Analyzer class with shared functionality
├── splice.py        - SpliceAnalyzer for splice site prediction and analysis
└── feature.py       - FeatureAnalyzer for genomic feature extraction
```

## Analyzer Classes

### `Analyzer` (base.py)

The base analyzer class that provides standardized paths and configuration settings for all derived analyzers.

**Key Features:**
- Standardized path structure for genomic data files
- Configuration management
- Access to shared resources through `shared_dir`
- Consistent interfaces for file access

**Usage Example:**
```python
from meta_spliceai.splice_engine.meta_models.core.analyzers import Analyzer

# Create a basic analyzer
analyzer = Analyzer()

# Access standardized paths
gtf_path = analyzer.get_path('gtf')
```

### `SpliceAnalyzer` (splice.py)

Specialized analyzer for splice site prediction and analysis.

**Key Features:**
- Loading and manipulation of splice site annotations
- Handling of overlapping gene information
- Access to enhanced splice site data with three-probability scores
- Automatic retrieval of splice sites with fallback to computation

**Usage Example:**
```python
from meta_spliceai.splice_engine.meta_models.core.analyzers import SpliceAnalyzer

# Create a splice analyzer
splice_analyzer = SpliceAnalyzer()

# Load splice sites
splice_sites = splice_analyzer.load_splice_sites()

# Retrieve overlapping gene metadata
overlapping_genes = splice_analyzer.retrieve_overlapping_gene_metadata(
    min_exons=2, 
    filter_valid_splice_sites=True
)
```

### `FeatureAnalyzer` (feature.py)

Specialized analyzer for genomic feature extraction and processing.

**Key Features:**
- Extraction of exons, introns, and other genomic features
- Sequence context retrieval
- Gene and transcript filtering
- Integration with gene annotations

**Usage Example:**
```python
from meta_spliceai.splice_engine.meta_models.core.analyzers import FeatureAnalyzer

# Create a feature analyzer
feature_analyzer = FeatureAnalyzer()

# Extract features for a specific gene
features = feature_analyzer.extract_gene_features('ENSG00000139618')
```

## Migration from Original Module

If you're currently using the original `analyzer.py` module, you can migrate to this package with minimal changes. The specialized analyzers maintain the same interfaces while providing enhanced functionality.

```python
# Old code
from meta_spliceai.splice_engine.meta_models.core.analyzer import SpliceAnalyzer

# New code (recommended)
from meta_spliceai.splice_engine.meta_models.core.analyzers import SpliceAnalyzer
```

## Shared Directory Concept

The analyzers use a shared directory concept (`shared_dir`) for storing global genomic datasets that can be reused across different analysis runs:

- Global datasets (gene annotations, splice sites, overlapping genes) are stored in `Analyzer.shared_dir`
- This allows different analysis runs to share the same reference data
- Avoids redundant storage and computation
- Typically set to be the parent directory of analysis_dir and eval_dir

This approach makes the system more efficient when working with large genomic datasets.
