# Data Documentation

This directory contains documentation for the data sources and formats used
by the meta-layer.

## Documents

| Document | Description |
|----------|-------------|
| [SPLICEVARDB.md](SPLICEVARDB.md) | SpliceVarDB dataset and label derivation |

## Quick Reference

### Data Sources

1. **Base Layer Artifacts** (`data/{build}/*/meta_models/`)
   - Pre-computed base model predictions
   - Derived features (position, context, etc.)
   - Used for: Training canonical splice classification

2. **SpliceVarDB** (`case_studies/workflows/splicevardb/`)
   - Experimentally validated variant effects
   - Classification: Splice-altering, Normal, Low-frequency
   - Used for: Evaluation and validated delta targets

3. **Genomic Resources** (managed by `meta_spliceai/system/genomic_resources/`)
   - FASTA: Reference genome sequences
   - GTF/GFF3: Gene annotations
   - Used for: Sequence extraction, splice site annotation
   - **Path resolution**: Use `Registry` as the single source of truth
   
   ```python
   from meta_spliceai.system.genomic_resources import Registry
   
   registry = Registry(build='GRCh38')  # or 'GRCh37'
   fasta_path = registry.get_fasta_path()
   gtf_path = registry.get_gtf_path()
   ```
   
   See `meta_spliceai/system/genomic_resources/README.md` for full documentation.

### Label Types

| Task | Label Source | Format | Example |
|------|--------------|--------|---------|
| Canonical Classification | GTF annotations | Categorical | "donor", "acceptor", "neither" |
| Binary Classification | SpliceVarDB | Binary | 0 (Normal), 1 (Splice-altering) |
| Delta Prediction | Base model + SpliceVarDB | Continuous | [+0.35, -0.02, -0.33] |

### Genome Build Mapping

| Base Model | Build | Annotations | SpliceVarDB Column |
|------------|-------|-------------|-------------------|
| OpenSpliceAI | GRCh38 | MANE | `hg38` |
| SpliceAI | GRCh37 | Ensembl | `hg19` |

## Related Documentation

- `../LABELING_STRATEGY.md` - Detailed labeling strategy
- `../TRAINING_VS_INFERENCE.md` - Data format differences
- `../DATA_FORMAT_AND_LEAKAGE.md` - Feature schemas and leakage prevention

