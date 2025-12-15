# Registry Quick Reference

**Last Updated**: November 1, 2025

---

## 🚀 Quick Start

```python
from meta_spliceai.system.genomic_resources import Registry

# Use default build from config
registry = Registry()

# Use specific build
registry = Registry(build='GRCh37', release='87')
```

---

## 📂 Common Paths

```python
# Reference files
gtf_path = registry.get_gtf_path()                    # GTF annotation
fasta_path = registry.get_fasta_path()                # Genome FASTA

# Derived datasets
gene_features = registry.resolve('gene_features')     # gene_features.tsv
splice_sites = registry.resolve('splice_sites')       # splice_sites_enhanced.tsv
transcript_features = registry.resolve('transcript_features')
exon_features = registry.resolve('exon_features')
junctions = registry.resolve('junctions')

# Directories
local_dir = registry.get_local_dir()                  # Build-specific dir
eval_dir = registry.get_eval_dir(create=True)         # Evaluation dir
analysis_dir = registry.get_analysis_dir(create=True) # Analysis dir

# Other files
annotations_db = registry.get_annotations_db_path()
overlapping_genes = registry.get_overlapping_genes_path()
chr_seq = registry.get_chromosome_sequence_path('21', format='parquet')
```

---

## 🔍 Search Order

Registry searches in this order:
1. **Build-specific**: `data/ensembl/{build}/`
2. **Top-level**: `data/ensembl/`
3. **Legacy**: `data/ensembl/spliceai_analysis/`

---

## 📋 All Available Methods

### Core Methods
- `resolve(kind)` - Resolve any resource by kind
- `get_gtf_path(validate=True)` - Get GTF file path
- `get_fasta_path(validate=True)` - Get FASTA file path
- `list_all()` - List all resolved paths

### Helper Methods
- `get_annotations_db_path(validate=False)`
- `get_overlapping_genes_path(validate=False)`
- `get_chromosome_sequence_path(chromosome, format='parquet', validate=False)`
- `get_local_dir()` - Returns build-specific directory
- `get_eval_dir(create=False)` - Returns evaluation directory
- `get_analysis_dir(create=False)` - Returns analysis directory

---

## 💡 Common Patterns

### In Workflows
```python
registry = Registry(build='GRCh37', release='87')

config = SpliceAIConfig(
    gtf_file=str(registry.get_gtf_path()),
    genome_fasta=str(registry.get_fasta_path()),
    local_dir=str(registry.get_local_dir()),
    eval_dir=str(registry.get_eval_dir())
)
```

### In Classes
```python
class MyWorkflow:
    def __init__(self, build='GRCh38', release='112'):
        self.registry = Registry(build=build, release=release)
    
    def load_data(self):
        gene_features = self.registry.resolve('gene_features')
        if not gene_features:
            raise FileNotFoundError(
                f"gene_features.tsv not found for {self.registry.cfg.build}"
            )
        return pd.read_csv(gene_features, sep='\t')
```

### Error Handling
```python
# With validation (raises FileNotFoundError if not found)
gtf_path = registry.get_gtf_path(validate=True)

# Without validation (returns None if not found)
annotations_db = registry.get_annotations_db_path(validate=False)
if not annotations_db:
    print("annotations.db not found, will create it")
```

---

## 🎯 Supported Resource Kinds

For use with `registry.resolve(kind)`:

- `gtf` - GTF annotation file
- `fasta` - Genome FASTA file
- `fasta_index` - FASTA index (.fai)
- `splice_sites` - Splice sites annotation
- `gene_features` - Gene features TSV
- `transcript_features` - Transcript features TSV
- `exon_features` - Exon features TSV
- `junctions` - Junctions TSV

---

## 🔧 Environment Variables

Override paths with environment variables:

```bash
export SS_GTF_PATH=/custom/path/to/annotation.gtf
export SS_FASTA_PATH=/custom/path/to/genome.fa
export SS_GENE_FEATURES_PATH=/custom/path/to/gene_features.tsv
```

---

## 📚 Related Documentation

- `meta_spliceai/system/genomic_resources/README.md` - Full Registry documentation
- `docs/development/REGISTRY_REFACTOR_2025-11-01.md` - Refactor details
- `docs/base_models/GENOME_BUILD_COMPATIBILITY.md` - Build compatibility

---

## ✅ Migration Checklist

Replacing hardcoded paths? Follow this checklist:

- [ ] Import Registry: `from meta_spliceai.system.genomic_resources import Registry`
- [ ] Initialize in `__init__`: `self.registry = Registry(build=build, release=release)`
- [ ] Replace hardcoded paths with `registry.resolve()` or specific methods
- [ ] Test with both GRCh37 and GRCh38
- [ ] Update error messages to reference build
- [ ] Remove hardcoded fallback paths

---

## 🎉 Benefits

✓ **Build Isolation** - No cross-contamination  
✓ **Maintainable** - Single source of truth  
✓ **Extensible** - Easy to add new builds  
✓ **Type Safe** - Returns Path objects  
✓ **Backward Compatible** - Works with existing setups

