# Gene Mapping System

**Status**: ✅ **IMPLEMENTED**  
**Date**: 2025-11-10  
**Module**: `meta_spliceai.system.genomic_resources.gene_mapper`

## Overview

The Gene Mapping System provides a formal solution for tracking and mapping genes across different genomic builds and annotation sources. This is essential when comparing base models that use different reference data (e.g., SpliceAI on GRCh37/Ensembl vs. OpenSpliceAI on GRCh38/MANE).

## Problem Statement

When comparing base models trained on different genomic resources:

1. **Different Builds**: GRCh37 vs. GRCh38 have different coordinate systems
2. **Different Sources**: Ensembl, MANE, GENCODE use different gene ID formats
3. **Gene Name Variations**: Same gene may have different identifiers:
   - Ensembl: `ENSG00000012048` (stable ID) + `BRCA1` (symbol)
   - MANE: `gene-BRCA1` (ID) + `BRCA1` (symbol)
4. **Missing Genes**: Not all genes exist in all annotations

## Solution Architecture

### Core Components

```
meta_spliceai/system/genomic_resources/
├── gene_mapper.py          # Main mapping logic
├── registry.py             # Resource path resolution (enhanced)
└── __init__.py             # Public API exports
```

### Key Classes

#### 1. `GeneMapper`

Central class for managing gene mappings across sources.

```python
from meta_spliceai.system.genomic_resources import GeneMapper

mapper = GeneMapper()
mapper.add_source_from_file('ensembl', 'GRCh37', 'data/ensembl/gene_features.tsv')
mapper.add_source_from_file('mane', 'GRCh38', 'data/mane/GRCh38/gene_features.tsv')

# Find common genes
common_genes = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
print(f"Found {len(common_genes)} genes in both sources")

# Map gene symbols to source-specific IDs
spliceai_ids = mapper.map_genes_to_source(['BRCA1', 'TP53'], 'ensembl/GRCh37')
# Returns: ['ENSG00000012048', 'ENSG00000141510']

openspliceai_ids = mapper.map_genes_to_source(['BRCA1', 'TP53'], 'mane/GRCh38')
# Returns: ['gene-BRCA1', 'gene-TP53']
```

#### 2. `GeneInfo`

Detailed information about a gene in a specific source:

```python
@dataclass
class GeneInfo:
    gene_symbol: str          # 'BRCA1'
    gene_id: str              # 'ENSG00000012048' or 'gene-BRCA1'
    ensembl_id: Optional[str] # Ensembl stable ID if available
    source: str               # 'ensembl' or 'mane'
    build: str                # 'GRCh37' or 'GRCh38'
    chrom: str                # 'chr17' or '17'
    start: int                # Gene start position
    end: int                  # Gene end position
    strand: str               # '+' or '-'
    gene_type: str            # 'protein_coding', 'lncRNA', etc.
```

#### 3. `GeneMappingResult`

Result of a gene mapping operation:

```python
@dataclass
class GeneMappingResult:
    gene_symbol: str          # Original gene symbol
    source_key: str           # Target source (e.g., 'mane/GRCh38')
    gene_id: Optional[str]    # Mapped gene ID
    found: bool               # Whether mapping was successful
    gene_info: Optional[GeneInfo]  # Full gene information
```

### Global Accessor

```python
from meta_spliceai.system.genomic_resources import get_gene_mapper

# Get global mapper (auto-loads common sources)
mapper = get_gene_mapper()

# Use immediately
common = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
```

## Usage Examples

### Example 1: Basic Gene Mapping

```python
from meta_spliceai.system.genomic_resources import get_gene_mapper

# Get mapper with auto-loaded sources
mapper = get_gene_mapper()

# Find genes in both GRCh37/Ensembl and GRCh38/MANE
common_genes = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])

# Sample 20 genes for testing
import random
test_genes = random.sample(common_genes, 20)

# Map to SpliceAI format (Ensembl IDs)
spliceai_genes = mapper.map_genes_to_source(test_genes, 'ensembl/GRCh37')

# Map to OpenSpliceAI format (MANE IDs)
openspliceai_genes = mapper.map_genes_to_source(test_genes, 'mane/GRCh38')

# Now you can pass these to the respective models
from meta_spliceai import run_base_model_predictions

spliceai_results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=spliceai_genes  # Ensembl IDs
)

openspliceai_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=openspliceai_genes  # MANE IDs
)
```

### Example 2: Get Intersection DataFrame

```python
# Get a DataFrame with gene IDs for both sources
df = mapper.get_intersection_dataframe(
    ['ensembl/GRCh37', 'mane/GRCh38'],
    include_all_info=True
)

print(df.head())
# Output:
# ┌─────────────┬────────────────────────┬───────────────────┬────────────────────┬─────┐
# │ gene_symbol ┆ ensembl_GRCh37_gene_id ┆ mane_GRCh38_gene_id┆ ensembl_GRCh37_chrom│ ... │
# │ ---         ┆ ---                    ┆ ---                ┆ ---                │     │
# │ str         ┆ str                    ┆ str                ┆ str                │     │
# ╞═════════════╪════════════════════════╪════════════════════╪════════════════════╪═════╡
# │ BRCA1       ┆ ENSG00000012048        ┆ gene-BRCA1         ┆ 17                 │ ... │
# │ TP53        ┆ ENSG00000141510        ┆ gene-TP53          ┆ 17                 │ ... │
# └─────────────┴────────────────────────┴────────────────────┴────────────────────┴─────┘
```

### Example 3: Get Detailed Gene Information

```python
# Get all information about BRCA1 across sources
brca1_info = mapper.get_gene_info('BRCA1')

for source_key, gene_info in brca1_info.items():
    print(f"{source_key}:")
    print(f"  ID: {gene_info.gene_id}")
    print(f"  Location: {gene_info.chrom}:{gene_info.start}-{gene_info.end}")
    print(f"  Type: {gene_info.gene_type}")
    print()

# Output:
# ensembl/GRCh37:
#   ID: ENSG00000012048
#   Location: 17:41196312-41277500
#   Type: protein_coding
#
# mane/GRCh38:
#   ID: gene-BRCA1
#   Location: chr17:43044295-43125483
#   Type: protein_coding
```

### Example 4: Custom Source Loading

```python
from meta_spliceai.system.genomic_resources import GeneMapper

# Create mapper without auto-loading
mapper = GeneMapper()

# Add custom sources
mapper.add_source_from_file(
    source='gencode',
    build='GRCh38',
    file_path='data/gencode/v43/gene_features.tsv'
)

mapper.add_source_from_file(
    source='ensembl',
    build='GRCh37',
    file_path='data/ensembl/gene_features.tsv'
)

# Find intersection
common = mapper.get_common_genes(['gencode/GRCh38', 'ensembl/GRCh37'])
```

## Integration with Test Scripts

The gene mapper is designed to integrate seamlessly with comparison test scripts:

```python
# scripts/testing/compare_base_models_robust.py (UPDATED)

from meta_spliceai.system.genomic_resources import get_gene_mapper

# Get mapper
mapper = get_gene_mapper()

# Find common genes
common_genes = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])

# Sample genes (using gene symbols for consistency)
sampled_symbols = sample_genes_by_category(common_genes)  # Returns gene symbols

# Map to SpliceAI format
spliceai_genes = mapper.map_genes_to_source(
    sampled_symbols,
    'ensembl/GRCh37',
    return_type='gene_id'
)

# Map to OpenSpliceAI format
openspliceai_genes = mapper.map_genes_to_source(
    sampled_symbols,
    'mane/GRCh38',
    return_type='gene_id'
)

# Run predictions with correct IDs
spliceai_results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=spliceai_genes
)

openspliceai_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=openspliceai_genes
)
```

## Benefits

### 1. **Correctness**
- Ensures genes are properly mapped to source-specific identifiers
- Prevents "gene not found" errors due to ID format mismatches

### 2. **Transparency**
- Clear tracking of which genes exist in which sources
- Explicit mapping results show success/failure for each gene

### 3. **Flexibility**
- Supports any number of sources and builds
- Easy to add new annotation sources (GENCODE, RefSeq, etc.)

### 4. **Reusability**
- Global mapper instance can be reused across scripts
- Cached registries avoid redundant file loading

### 5. **Maintainability**
- Centralized gene mapping logic
- Single source of truth for cross-build gene identification

## API Reference

### `GeneMapper` Class

#### Methods

- `add_source(source, build, gene_df, ...)`: Add a gene annotation source
- `add_source_from_file(source, build, file_path, ...)`: Load source from file
- `get_common_genes(source_keys, min_sources=None)`: Find genes in multiple sources
- `map_genes_to_source(gene_symbols, target_source_key, return_type='gene_id')`: Map genes to source-specific IDs
- `get_gene_info(gene_symbol, source_key=None)`: Get detailed gene information
- `get_intersection_dataframe(source_keys, include_all_info=False)`: Get DataFrame of common genes
- `get_summary()`: Get mapper statistics
- `print_summary()`: Print formatted summary

### Global Functions

- `get_gene_mapper(auto_load=True, sources=None)`: Get or create global mapper
- `reset_gene_mapper()`: Reset global mapper instance
- `get_genomic_registry(source, build)`: Get genomic resource registry

## Testing

Test the gene mapper:

```bash
cd /Users/pleiadian53/work/meta-spliceai
source ~/.bash_profile && mamba activate surveyor

python3 << 'EOF'
from meta_spliceai.system.genomic_resources import get_gene_mapper

# Get mapper
mapper = get_gene_mapper()

# Print summary
mapper.print_summary()

# Find common genes
common = mapper.get_common_genes(['ensembl/GRCh37', 'mane/GRCh38'])
print(f"\nCommon genes: {len(common):,}")

# Test mapping
test_genes = ['BRCA1', 'TP53', 'EGFR']
for gene in test_genes:
    info = mapper.get_gene_info(gene)
    if info:
        print(f"\n{gene}:")
        for source_key, gene_info in info.items():
            print(f"  {source_key}: {gene_info.gene_id}")
    else:
        print(f"\n{gene}: NOT FOUND")

# Test mapping to sources
spliceai_ids = mapper.map_genes_to_source(test_genes, 'ensembl/GRCh37')
openspliceai_ids = mapper.map_genes_to_source(test_genes, 'mane/GRCh38')

print(f"\nSpliceAI IDs: {spliceai_ids}")
print(f"OpenSpliceAI IDs: {openspliceai_ids}")
EOF
```

## Next Steps

1. **Update Test Scripts**: Modify `compare_base_models_robust.py` to use the gene mapper
2. **Populate MANE Sequences**: Fix MANE sequence files to include `gene_name` column
3. **Add More Sources**: Support GENCODE, RefSeq, etc.
4. **Cross-Build Liftover**: Add coordinate liftover between GRCh37 ↔ GRCh38

## Related Documentation

- `docs/base_models/BASE_MODEL_COMPARISON_GUIDE.md`: Base model comparison workflow
- `docs/base_models/TEST_SCRIPTS_COMPARISON.md`: Test script organization
- `meta_spliceai/system/genomic_resources/README.md`: Genomic resources system

## Summary

The Gene Mapping System provides a **formal, systematic solution** for tracking genes across different genomic builds and annotation sources. It ensures that when comparing base models, genes are correctly identified and mapped to their source-specific identifiers, preventing errors and enabling accurate cross-model comparisons.

**Key Takeaway**: Use `get_gene_mapper()` to get common genes and map them to source-specific IDs before passing to base models.


