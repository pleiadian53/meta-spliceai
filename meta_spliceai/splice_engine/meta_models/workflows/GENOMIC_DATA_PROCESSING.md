# Genomic Data Processing in Enhanced Splice Prediction Workflow

This document details the key genomic data processing steps implemented in the `run_enhanced_splice_prediction_workflow` function found in `splice_prediction_workflow.py`. These steps prepare the necessary genomic datasets before the per-chromosome and per-gene processing begins.

## Overview of Enhanced Workflow

The enhanced splice prediction workflow extends the base SpliceAI functionality by:
- Capturing all three probability scores (donor, acceptor, neither) for each position
- Preserving transcript ID mapping for splice sites shared across multiple transcripts
- Enabling targeted gene and chromosome analysis
- Supporting automatic splice site annotation adjustment detection
- Generating derived probability features for meta-model training

## Key Preprocessing Steps

### 1. Prepare Gene Annotations

```python
annot_result = prepare_gene_annotations(
    local_dir=local_dir,
    gtf_file=gtf_file,
    do_extract=do_extract_annotations,
    target_chromosomes=target_chromosomes,
    use_shared_db=True,
    separator=separator,
    verbosity=verbosity
)
```

**Purpose**: Extracts and processes gene annotations from the GTF file.

**Key Features**:
- Parses standard GTF format into structured dataframes
- Filters for specific chromosomes if provided
- Uses shared database to avoid redundant extraction across runs
- Normalizes chromosome names for consistent processing
- Ensures proper gene ID and transcript ID mapping

**Technical Notes**:
- This step creates the foundation for all downstream analysis
- When `do_extract_annotations=False`, uses precomputed annotations if available
- Includes gene types, transcript biotypes, and other essential metadata

### 2. Prepare Splice Site Annotations

```python
ss_result = prepare_splice_site_annotations(
    local_dir=local_dir,
    gtf_file=gtf_file,
    do_extract=do_extract_splice_sites,
    target_chromosomes=target_chromosomes,
    consensus_window=consensus_window,
    separator=separator,
    verbosity=verbosity
)
```

**Purpose**: Extracts donor and acceptor splice site positions from gene annotations.

**Key Features**:
- Identifies all canonical splice sites from exon boundaries
- Classifies sites as donor or acceptor based on strand and position
- Associates splice sites with their parent transcripts and genes
- Applies consensus window for systematic position evaluation
- Filters to specific chromosomes if requested

**Technical Notes**:
- Creates the ground truth dataset for model evaluation
- Handles strand-specific site designation (donor/acceptor)
- Manages complex cases like alternative splicing events

### 3. Prepare Genomic Sequences

```python
seq_result = prepare_genomic_sequences(
    local_dir=local_dir,
    gtf_file=gtf_file,
    genome_fasta=genome_fasta,
    mode='gene',
    seq_type='full',
    do_extract=do_extract_sequences,
    chromosomes=target_chromosomes,
    test_mode=test_mode,
    seq_format=seq_format,
    verbosity=verbosity
)
```

**Purpose**: Extracts genomic DNA sequences for genes from the reference genome.

**Key Features**:
- Retrieves full gene sequences including flanking regions
- Organizes sequences by chromosome for efficient access
- Handles both gene-level and transcript-level sequence extraction
- Supports different sequence formats (FASTA, Parquet)
- Adjusts for test mode to limit sequence volume

**Technical Notes**:
- Sequences provide input for the SpliceAI deep learning models
- Includes properly oriented sequences accounting for strand
- Enables sequence context windows required by SpliceAI

### 4. Handle Overlapping Genes

```python
overlap_result = handle_overlapping_genes(
    local_dir=local_dir,
    gtf_file=gtf_file,
    do_find=do_find_overlaping_genes,
    filter_valid_splice_sites=kwargs.get('filter_valid_splice_sites', True),
    min_exons=kwargs.get('min_exons', 2),
    separator=separator,
    verbosity=verbosity
)
```

**Purpose**: Identifies and manages genes that overlap in genomic coordinates.

**Key Features**:
- Maps overlapping gene relationships and counts
- Identifies genes sharing genomic regions on same/opposite strands
- Filters for genes with valid splice sites (multiple exons)
- Handles complex genomic architectures (nested genes, antisense pairs)
- Enables special handling of splice site prediction in overlapping regions

**Technical Notes**:
- Critical for accurate splice site prediction in complex genomic regions
- Affects interpretation of true/false positives in error analysis
- Created database can be reused across different runs

### 5. Determine Which Chromosomes to Process

```python
chrom_result = determine_target_chromosomes(
    local_dir=local_dir,
    gtf_file=gtf_file,
    target_genes=target_genes,
    chromosomes=chromosomes,
    test_mode=test_mode,
    separator=separator,
    verbosity=verbosity
)
```

**Purpose**: Finalizes the list of chromosomes for processing based on configuration and targets.

**Key Features**:
- Resolves chromosome list from multiple potential sources
- Maps target genes to their chromosomes if gene-specific analysis is requested
- Handles test mode by selecting a limited set of chromosomes
- Normalizes chromosome naming across different reference formats
- Optimizes processing order for efficiency

**Technical Notes**:
- Supports focused analysis on specific chromosomes
- Enables efficient batch processing by chromosome
- Critical for workflow parallelization strategies

### 6. Load SpliceAI Models

```python
model_result = load_spliceai_models(verbosity=verbosity)
```

**Purpose**: Loads the pre-trained SpliceAI deep learning models.

**Key Features**:
- Initializes the ensemble of SpliceAI models with different context lengths
- Prepares models for sequence analysis and splice site prediction
- Configures prediction parameters (context window size, batch size)
- Validates model availability and integrity
- Returns initialized model objects ready for prediction

**Technical Notes**:
- SpliceAI uses multiple CNN models with different context window sizes
- Models are pre-trained on large genomic datasets
- This step does not require GPU, but inference will benefit from it

### 7. Prepare Splice Site Position Adjustments

```python
if config.use_auto_position_adjustments:
    # Logic for auto-detecting position adjustments
    # Using sample predictions to empirically determine adjustments
```

**Purpose**: Detects and corrects systematic biases in splice site position predictions.

**Key Features**:
- Automatically detects shifts between predicted and annotated splice sites
- Calibrates donor and acceptor position predictions separately by strand
- Optimizes prediction accuracy by correcting systematic biases
- Uses empirical analysis of sample predictions to determine adjustments
- Applies computed adjustments to all subsequent predictions

**Technical Notes**:
- SpliceAI models sometimes predict splice sites slightly offset from true positions
- Strand-specific and site-type-specific adjustments improve accuracy
- Critical for precise splice site localization in downstream applications
- Adjustments are learned empirically from data, not hardcoded

## Integration with Meta-Model Pipeline

These genomic data preprocessing steps establish the foundation for:

1. **Feature Generation**: Creating derived probability features (entropy, ratios)
2. **Tri-Score System**: Capturing all three probability scores at each position
3. **Meta-Model Training**: Preparing comprehensive datasets for error correction
4. **Systematic Error Analysis**: Identifying and addressing prediction biases

After these preprocessing steps, the workflow proceeds to the per-chromosome and per-gene processing logic, where predictions are generated and enhanced with derived features for meta-model training.
