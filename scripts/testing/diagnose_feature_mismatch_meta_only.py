#!/usr/bin/env python
"""
Diagnose feature mismatch in meta-only mode.
"""
import polars as pl
import sys
from pathlib import Path

# Load base model predictions from meta-only mode
test_file = Path('predictions/meta_modes_test/test_meta_only/ENSG00000141736/predictions/meta_only/base_model_predictions.parquet')

if not test_file.exists():
    print(f"❌ File not found: {test_file}")
    sys.exit(1)

df = pl.read_parquet(test_file)

print('=' * 80)
print('FEATURE ANALYSIS FOR META-ONLY MODE')
print('=' * 80)
print(f'Total columns in base_model_predictions.parquet: {len(df.columns)}')
print('')

# Define exclusions (from preprocessing.py and model training)
LEAKAGE_COLUMNS = [
    'is_donor', 'is_acceptor', 'is_neither',
    'label', 'y', 'target',
    'is_true_donor', 'is_true_acceptor',
    'site_type', 'is_splice_site'
]

METADATA_COLUMNS = [
    'gene_id', 'transcript_id', 'position',
    'absolute_position', 'gene_name',
    'tx_name', 'exon_id'
]

SEQUENCE_COLUMNS = ['sequence', 'kmer', 'motif']

REDUNDANT_COLUMNS = [
    'max_score', 'predicted_class',
    'max_confidence', 'score_spread',
    'is_low_confidence', 'is_high_entropy', 'is_uncertain',
    'confidence_category', 'entropy'
]

# Load model's global exclusions
model_exclusions_file = Path('results/meta_model_1000genes_3mers_fresh/global_excluded_features.txt')
model_exclusions = []
if model_exclusions_file.exists():
    with open(model_exclusions_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                model_exclusions.append(line)

print('Model-specific exclusions:')
for feat in model_exclusions:
    print(f'  - {feat}')
print('')

# Combine all exclusions
all_exclusions = set(
    LEAKAGE_COLUMNS +
    METADATA_COLUMNS +
    SEQUENCE_COLUMNS +
    REDUNDANT_COLUMNS +
    model_exclusions
)

print(f'Total exclusion rules: {len(all_exclusions)}')
print('')

# Identify which columns would be kept as features
feature_cols = [col for col in df.columns if col not in all_exclusions]

print(f'Columns that would be kept as features: {len(feature_cols)}')
print('')

# Show the feature columns
print('Feature columns:')
for i, col in enumerate(feature_cols, 1):
    dtype = str(df[col].dtype)
    print(f'  {i:3d}. {col:40s} {dtype}')

print('')
print('=' * 80)
print(f'RESULT: {len(feature_cols)} features would be extracted')
print(f'MODEL EXPECTS: 121 features')
print(f'DIFFERENCE: {len(feature_cols) - 121} extra features')
print('=' * 80)

# Identify the extra features
if len(feature_cols) > 121:
    print('')
    print('LIKELY EXTRA FEATURES (should be excluded):')
    # These are probably the ones causing issues
    suspicious_features = [
        col for col in feature_cols
        if any(keyword in col.lower() for keyword in [
            'adjusted', 'meta', 'strand', 'chrom', 'splice_type'
        ])
    ]
    for feat in suspicious_features:
        print(f'  - {feat} (dtype: {df[feat].dtype})')

