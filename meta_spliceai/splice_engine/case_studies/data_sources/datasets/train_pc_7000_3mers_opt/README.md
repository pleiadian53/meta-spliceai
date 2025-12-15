# Dataset: train_pc_7000_3mers_opt

## Quick Overview

**Purpose**: Protein-coding gene training dataset for meta-model development  
**Size**: 595MB (6,708 genes, ~7.7M records)  
**Features**: 143 features including SpliceAI predictions, 3-mer composition, and genomic context  
**Created**: 2025-08-23  

## Dataset Location
```
/home/bchiu/work/meta-spliceai/train_pc_7000_3mers_opt/
```

## Documentation Files

| File | Description |
|------|-------------|
| [`train_pc_7000_3mers_opt_profile.md`](train_pc_7000_3mers_opt_profile.md) | Comprehensive dataset overview and characteristics |
| [`train_pc_7000_3mers_opt_technical_spec.md`](train_pc_7000_3mers_opt_technical_spec.md) | Detailed schema and technical specifications |
| [`validate_train_pc_7000_3mers_opt.py`](validate_train_pc_7000_3mers_opt.py) | Dataset validation and usage demonstration script |

## Quick Start

### Validation
```bash
cd /home/bchiu/work/meta-spliceai
python meta_spliceai/splice_engine/case_studies/data_sources/datasets/train_pc_7000_3mers_opt/validate_train_pc_7000_3mers_opt.py train_pc_7000_3mers_opt
```

### Loading Data
```python
import pandas as pd

# Load gene manifest
manifest = pd.read_csv('train_pc_7000_3mers_opt/master/gene_manifest.csv')

# Load sample batch
batch = pd.read_parquet('train_pc_7000_3mers_opt/master/batch_00001.parquet')
```

## Key Statistics

- **Genes**: 6,708 protein-coding genes
- **Chromosomes**: All autosomes (1-22) + sex chromosomes (X, Y)
- **Batch Files**: 28 parquet files
- **Features**: 143 comprehensive features
- **Memory**: ~1.2GB RAM for full dataset loading
- **Enhanced Manifest**: 15 columns including gene characteristics and splice site density
- **Splice Density**: Mean 6.28 sites/kb (higher than diverse dataset)

## Use Cases

- Meta-model training for splice site prediction
- Variant impact assessment
- Disease-specific splicing pattern analysis
- Alternative splicing research

---

**Dataset Version**: 1.0  
**Documentation Version**: 1.0  
**Last Updated**: 2025-08-23
