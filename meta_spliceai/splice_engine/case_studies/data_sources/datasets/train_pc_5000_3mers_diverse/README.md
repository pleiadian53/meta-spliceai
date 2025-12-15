# Dataset: train_pc_5000_3mers_diverse

## Quick Overview

**Purpose**: Diverse protein-coding gene training dataset for meta-model development  
**Size**: 213MB (3,111 genes, 584,379 records)  
**Features**: 143-148 features including SpliceAI predictions, 3-mer composition, and genomic context  
**Created**: 2025-08-22  

## Dataset Location
```
/home/bchiu/work/meta-spliceai/train_pc_5000_3mers_diverse/
```

## Documentation Files

| File | Description |
|------|-------------|
| [`train_pc_5000_3mers_diverse_profile.md`](train_pc_5000_3mers_diverse_profile.md) | Comprehensive dataset overview and characteristics |
| [`train_pc_5000_3mers_diverse_technical_spec.md`](train_pc_5000_3mers_diverse_technical_spec.md) | Detailed schema and technical specifications |
| [`validate_train_pc_5000_3mers_diverse.py`](validate_train_pc_5000_3mers_diverse.py) | Dataset validation and usage demonstration script |

## Quick Start

### Validation
```bash
cd /home/bchiu/work/meta-spliceai
python meta_spliceai/splice_engine/case_studies/data_sources/datasets/train_pc_5000_3mers_diverse/validate_train_pc_5000_3mers_diverse.py train_pc_5000_3mers_diverse
```

### Loading Data
```python
import pandas as pd

# Load gene manifest
manifest = pd.read_csv('train_pc_5000_3mers_diverse/master/gene_manifest.csv')

# Load sample batch
batch = pd.read_parquet('train_pc_5000_3mers_diverse/master/batch_00001.parquet')
```

## Key Statistics

- **Genes**: 3,111 diverse genes (protein-coding and pseudogenes)
- **Chromosomes**: All autosomes (1-22) + sex chromosomes (X, Y)
- **Batch Files**: 20 parquet files
- **Features**: 143-148 comprehensive features (varies by batch)
- **Memory**: ~620MB RAM for full dataset loading
- **Enhanced Manifest**: 15 columns including gene characteristics and splice site density

## Dataset Characteristics

### Gene Selection Strategy
- **Diverse Selection**: Balanced representation across gene types and chromosomes
- **Gene Types**: Protein-coding (primary), lncRNA, pseudogenes, immunoglobulin genes
- **Chromosome Coverage**: Comprehensive genomic distribution

### Feature Engineering
- **3-mer K-mers**: All 64 possible trinucleotides (AAA-TTT)
- **SpliceAI Predictions**: Donor, acceptor, and neither scores
- **Genomic Context**: Position-based features and splice site characteristics
- **Gene Annotations**: Length, exon structure, and biotype information

## Use Cases

- Meta-model training for splice site prediction
- Diverse gene representation analysis
- Cross-chromosome splicing pattern studies
- Pseudogene vs protein-coding comparison
- Alternative splicing research with diverse gene types

## Comparison with Other Datasets

| Dataset | Genes | Size | Focus | Manifest |
|---------|-------|------|-------|----------|
| `train_pc_7000_3mers_opt` | 6,708 | 595MB | Optimized protein-coding | Enhanced |
| **`train_pc_5000_3mers_diverse`** | **3,111** | **213MB** | **Diverse gene types** | **Enhanced** |
| `train_pc_1000_3mers` | 1,002 | ~400MB | Error-focused selection | Basic |

---

**Dataset Version**: 1.0  
**Documentation Version**: 1.0  
**Last Updated**: 2025-01-27
