# Gene Mapper Quick Reference

**Last Updated**: 2025-11-11

## Quick Start

```python
from meta_spliceai.system.genomic_resources import (
    EnhancedGeneMapper,
    get_or_create_mane_ensembl_mapping
)
from pathlib import Path

# 1. Load external mapping
data_dir = Path('data')
mane_to_ensembl = get_or_create_mane_ensembl_mapping(data_dir)

# 2. Create mapper
mapper = EnhancedGeneMapper()

# 3. Add sources
mapper.add_source_from_file('ensembl', 'GRCh37', 'data/ensembl/GRCh37/gene_features.tsv')
mapper.add_source_from_file('mane', 'GRCh38', 'data/mane/GRCh38/gene_features.tsv',
                            external_id_mapping=mane_to_ensembl)

# 4. Find mappings
mappings = mapper.find_mappings('ensembl/GRCh37', 'mane/GRCh38')

# 5. Get high-confidence only
high_conf = mapper.get_high_confidence_mappings('ensembl/GRCh37', 'mane/GRCh38', min_confidence=0.9)
```

## Common Operations

### Print Summary

```python
mapper.print_summary('ensembl/GRCh37', 'mane/GRCh38')
```

### Filter by Confidence

```python
# High confidence (≥0.9)
high = [m for m in mappings if m.confidence >= 0.9]

# Medium confidence (0.7-0.9)
medium = [m for m in mappings if 0.7 <= m.confidence < 0.9]
```

### Filter by Strategy

```python
from meta_spliceai.system.genomic_resources import MappingStrategy

# Ensembl ID matches only
ensembl_id = [m for m in mappings if m.strategy == MappingStrategy.ENSEMBL_ID]

# Gene symbol matches only
symbol = [m for m in mappings if m.strategy == MappingStrategy.GENE_SYMBOL]
```

### Export to DataFrame

```python
# All mappings
df = mapper.to_dataframe('ensembl/GRCh37', 'mane/GRCh38')

# High-confidence only
df_high = mapper.to_dataframe('ensembl/GRCh37', 'mane/GRCh38', min_confidence=0.9)

# Save
df_high.write_csv('mappings.tsv', separator='\t')
```

### Extract Gene Lists

```python
# Common gene symbols
symbols = [m.gene_symbol for m in mappings]

# Source-specific IDs
ensembl_ids = [m.source1_gene_id for m in mappings]
mane_ids = [m.source2_gene_id for m in mappings]
```

## Mapping Strategies

| Strategy | Confidence | Description |
|----------|------------|-------------|
| `ENSEMBL_ID` | 1.0 | Match by Ensembl stable ID |
| `GENE_SYMBOL` | 0.7-0.9 | Match by gene name |
| `COORDINATES` | 0.5-0.9 | Match by genomic location |

## Key Results (GRCh37 ↔ GRCh38)

| Metric | Value |
|--------|-------|
| Total Mappings | 17,783 |
| Ensembl ID Matches | 17,696 (99.5%) |
| High Confidence (≥0.9) | 17,696 (99.5%) |

## Common Patterns

### Pattern 1: Sample Common Genes

```python
import random

# Get high-confidence mappings
high_conf = mapper.get_high_confidence_mappings('ensembl/GRCh37', 'mane/GRCh38', min_confidence=0.9)

# Sample 30 genes
sampled = random.sample(high_conf, k=min(30, len(high_conf)))

# Extract IDs
ensembl_ids = [m.source1_gene_id for m in sampled]
mane_ids = [m.source2_gene_id for m in sampled]
```

### Pattern 2: Check Specific Genes

```python
test_genes = ['BRCA1', 'TP53', 'EGFR']

for gene in test_genes:
    gene_mappings = [m for m in mappings if m.gene_symbol == gene]
    if gene_mappings:
        m = gene_mappings[0]
        print(f"{gene}: {m.strategy.value} (confidence={m.confidence:.2f})")
```

### Pattern 3: Get Mapping Statistics

```python
summary = mapper.get_mapping_summary('ensembl/GRCh37', 'mane/GRCh38')

print(f"Total: {summary['total_mappings']:,}")
print(f"By strategy: {summary['by_strategy']}")
print(f"By confidence: {summary['by_confidence']}")
```

## Files

### Implementation
- `meta_spliceai/system/genomic_resources/gene_mapper_enhanced.py`
- `meta_spliceai/system/genomic_resources/external_id_mapper.py`

### Data
- `data/mane/GRCh38/mane_to_ensembl_mapping.json`

### Documentation
- `docs/base_models/ENHANCED_GENE_MAPPER_COMPLETE.md` (full guide)
- `docs/base_models/ENHANCED_GENE_MAPPER_SUMMARY.md` (summary)
- `docs/base_models/GENE_MAPPER_QUICK_REFERENCE.md` (this file)

## Troubleshooting

### Issue: Low overlap between sources

**Solution**: Ensure external ID mapping is loaded:

```python
mane_to_ensembl = get_or_create_mane_ensembl_mapping(data_dir)
mapper.add_source_from_file('mane', 'GRCh38', ..., external_id_mapping=mane_to_ensembl)
```

### Issue: Ambiguous mappings

**Solution**: Filter by confidence:

```python
high_conf = mapper.get_high_confidence_mappings(..., min_confidence=0.9)
```

### Issue: Missing genes

**Solution**: Check if genes exist in both sources:

```python
# Get genes only in source1
all_mappings = mapper.find_mappings('ensembl/GRCh37', 'mane/GRCh38')
mapped_genes = set(m.gene_symbol for m in all_mappings)

# Check specific gene
if 'GENE_X' not in mapped_genes:
    print("GENE_X not found in both sources")
```

## See Also

- [ENHANCED_GENE_MAPPER_COMPLETE.md](ENHANCED_GENE_MAPPER_COMPLETE.md) - Complete guide
- [GENE_MAPPING_SYSTEM.md](GENE_MAPPING_SYSTEM.md) - Design rationale
- [GENE_MAPPER_INTEGRATION_COMPLETE.md](GENE_MAPPER_INTEGRATION_COMPLETE.md) - Integration summary

