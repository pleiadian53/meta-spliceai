# Test Data Preparation Utility

## Overview
The `prepare_test_data.py` script creates minimal, balanced test datasets from full `analysis_sequences_*` artifacts for rapid error model workflow validation.

## Use Cases
- **Workflow Testing**: Validate the complete pipeline with minimal data (CI/CD)
- **Development**: Quick iteration during feature development
- **Debugging**: Isolate issues with small, reproducible datasets
- **Resource-Limited Testing**: Run on systems with memory constraints

## Key Features
- **Balanced Sampling**: Ensures equal representation across prediction classes (FP, TP, FN, TN)
- **Context Trimming**: Reduces sequences to configurable lengths (default 50nt for speed)
- **Feature Selection**: Option to use subset of features for faster processing
- **Reproducible**: Fixed random seed for consistent results
- **Memory Efficient**: Polars support for large input files

## Usage Examples

### Basic Test Dataset (1000 samples, 50nt context)
```bash
python prepare_test_data.py \
    --input data/analysis_sequences_full.tsv \
    --output data/test_minimal.tsv
```

### Custom Configuration
```bash
python prepare_test_data.py \
    --input data/analysis_sequences_full.tsv \
    --output data/test_custom.tsv \
    --context-length 100 \
    --sample-size 500 \
    --features score donor_score acceptor_score
```

### Large File Processing with Polars
```bash
python prepare_test_data.py \
    --input data/large_analysis.tsv \
    --output data/test_subset.tsv \
    --use-polars \
    --sample-size 2000
```

## Parameters
- `--input`: Input TSV file (analysis_sequences_* artifact)
- `--output`: Output TSV file path
- `--context-length`: Sequence context in nucleotides (default: 50)
- `--sample-size`: Number of samples (default: 1000)
- `--features`: Specific features to include (default: all)
- `--use-polars`: Enable Polars for large files
- `--seed`: Random seed (default: 42)

## Output
Creates a TSV file with:
- Balanced class distribution
- Trimmed sequences
- Essential columns preserved
- Selected features included
- Summary statistics logged

## Integration with Workflow
Use the prepared test data with the error model workflow:
```bash
# Prepare test data
python prepare_test_data.py \
    --input data/full.tsv \
    --output data/test.tsv

# Run workflow with test data
python run_error_model_workflow.py \
    --input-file data/test.tsv \
    --error-type FP_TP \
    --output-dir results/test_run
```
