# Dataset: {DATASET_NAME}

## Quick Overview

**Purpose**: {DATASET_PURPOSE}  
**Size**: {DATASET_SIZE}  
**Features**: {FEATURE_COUNT} features including {KEY_FEATURES}  
**Created**: {CREATION_DATE}  

## Dataset Location
```
/home/bchiu/work/meta-spliceai/{DATASET_NAME}/
```

## Documentation Files

| File | Description |
|------|-------------|
| [`{DATASET_NAME}_profile.md`]({DATASET_NAME}_profile.md) | Comprehensive dataset overview and characteristics |
| [`{DATASET_NAME}_technical_spec.md`]({DATASET_NAME}_technical_spec.md) | Detailed schema and technical specifications |
| [`validate_{DATASET_NAME}.py`](validate_{DATASET_NAME}.py) | Dataset validation and usage demonstration script |

## Quick Start

### Validation
```bash
cd /home/bchiu/work/meta-spliceai
python meta_spliceai/splice_engine/case_studies/data_sources/datasets/{DATASET_NAME}/validate_{DATASET_NAME}.py {DATASET_NAME}
```

### Loading Data
```python
import pandas as pd

# Load gene manifest
manifest = pd.read_csv('{DATASET_NAME}/master/gene_manifest.csv')

# Load sample batch
batch = pd.read_parquet('{DATASET_NAME}/master/batch_00001.parquet')
```

## Key Statistics

- **Genes**: {GENE_COUNT} {GENE_TYPES}
- **Chromosomes**: {CHROMOSOME_COVERAGE}
- **Batch Files**: {BATCH_COUNT} parquet files
- **Features**: {FEATURE_COUNT} comprehensive features
- **Memory**: {MEMORY_USAGE} for full dataset loading

## Use Cases

- {USE_CASE_1}
- {USE_CASE_2}
- {USE_CASE_3}
- {USE_CASE_4}

---

**Dataset Version**: {DATASET_VERSION}  
**Documentation Version**: {DOC_VERSION}  
**Last Updated**: {LAST_UPDATED}
