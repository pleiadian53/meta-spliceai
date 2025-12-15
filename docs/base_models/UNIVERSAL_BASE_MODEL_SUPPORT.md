# Universal Base Model Support

**Date**: November 6, 2025  
**Status**: ✅ COMPLETE - All tests passed

---

## Overview

The MetaSpliceAI framework now supports **universal base model switching** with a single parameter. The system automatically routes to the correct genomic resources (GTF, FASTA, splice sites) based on the selected base model.

---

## Key Features

### 1. Single-Parameter Switching ✅

Users can switch between base models with **one parameter**:

```python
from meta_spliceai import run_base_model_predictions

# Use SpliceAI (automatically uses GRCh37/Ensembl)
results_spliceai = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53']
)

# Use OpenSpliceAI (automatically uses GRCh38/MANE)
results_openspliceai = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53']
)
```

### 2. Automatic Resource Routing ✅

The system automatically handles:
- ✅ Genomic build selection (GRCh37 vs GRCh38)
- ✅ Annotation source (Ensembl vs MANE vs Gencode)
- ✅ Splice site annotations (~2M vs ~370K sites)
- ✅ Model loading (Keras vs PyTorch)
- ✅ Artifact directory structure
- ✅ Schema standardization

### 3. Universal GTF Parser ✅

The `extract_genes_from_gtf()` function now handles:
- ✅ **Ensembl GTF**: Explicit "gene" features
- ✅ **MANE GTF**: Transcript-only format (derives genes by aggregation)
- ✅ **Chromosome naming**: Normalizes "chr1" → "1"
- ✅ **Non-standard chromosomes**: Filters patches/fixes
- ✅ **Attribute variations**: Handles `gene_name` vs `gene` attributes

---

## Supported Base Models

| Base Model | Build | Annotation | Genes | Splice Sites | Framework |
|------------|-------|------------|-------|--------------|-----------|
| **SpliceAI** | GRCh37 | Ensembl 87 | ~58K | ~2M | Keras |
| **OpenSpliceAI** | GRCh38 | MANE v1.3 | ~19K | ~370K | PyTorch |
| **Future Models** | Any | Any | Any | Any | Any |

---

## Architecture

### Dynamic Configuration Flow

```
User specifies: base_model='openspliceai'
                       ↓
BaseModelConfig.__post_init__()
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
Infer build & source        Override Analyzer defaults
(GRCh38 + MANE)            (gtf_file, genome_fasta, eval_dir)
        ↓                             ↓
        └──────────────┬──────────────┘
                       ↓
        get_artifact_manager()
                       ↓
        Create ArtifactManager with:
        - build: GRCh38
        - source: mane
        - base_model: openspliceai
                       ↓
        Artifact paths routed to:
        data/mane/GRCh38/openspliceai_eval/...
```

### Key Components

1. **`BaseModelConfig.__post_init__()`**
   - Detects base model
   - Infers correct build and source
   - Overrides Analyzer defaults if needed
   - Sets up dynamic paths

2. **`get_artifact_manager()`**
   - Creates artifact manager with correct build/source
   - Routes output artifacts to model-specific directories
   - Handles test vs production modes

3. **`extract_genes_from_gtf()`**
   - Universal GTF parser
   - Handles Ensembl and MANE formats
   - Normalizes chromosome names
   - Filters non-standard chromosomes

4. **`load_base_model_ensemble()`**
   - Dispatches to correct model loader
   - Handles Keras (SpliceAI) and PyTorch (OpenSpliceAI)
   - Returns unified metadata

---

## Validation Results

### Test Suite: Universal Base Model Support

```bash
python scripts/testing/test_universal_base_model_support.py
```

**Results**: ✅ **ALL TESTS PASSED**

| Test | Status |
|------|--------|
| GRCh37 gene extraction | ✅ PASS (57,905 genes) |
| GRCh37 splice sites | ✅ PASS (1,998,526 sites) |
| SpliceAI config routing | ✅ PASS |
| GRCh38 gene extraction | ✅ PASS (19,226 genes) |
| GRCh38 splice sites | ✅ PASS (369,918 sites) |
| OpenSpliceAI config routing | ✅ PASS |
| Single-parameter switching | ✅ PASS |

---

## Adding New Base Models

### Step 1: Add Model Loader

```python
# In meta_spliceai/splice_engine/meta_models/utils/model_utils.py

def load_newmodel_ensemble(context: int = 10000, device: str = 'cpu'):
    """Load NewModel ensemble."""
    # Your model loading logic here
    models = load_your_models(...)
    return models, device

def load_base_model_ensemble(base_model: str, ...):
    # Add new case
    elif base_model_lower == 'newmodel':
        models, device_used = load_newmodel_ensemble(...)
        metadata = {
            'base_model': 'newmodel',
            'genome_build': 'GRCh38',  # or whatever build
            'context': context,
            'framework': 'pytorch',  # or 'keras', 'jax', etc.
            'num_models': len(models),
            'device': device_used
        }
```

### Step 2: Add Build/Source Mapping

```python
# In meta_spliceai/splice_engine/meta_models/core/data_types.py
# In BaseModelConfig.__post_init__()

if base_model_lower == 'newmodel':
    registry = Registry(build='GRCh38_NewAnnotation', release='1.0')
```

### Step 3: Download Genomic Resources

```bash
# Create download script
./scripts/setup/download_newmodel_data.sh
```

### Step 4: Test

```python
from meta_spliceai import run_base_model_predictions

results = run_base_model_predictions(
    base_model='newmodel',
    target_genes=['BRCA1']
)
```

That's it! The system handles the rest automatically.

---

## Usage Examples

### Example 1: Basic Prediction

```python
from meta_spliceai import run_base_model_predictions

# SpliceAI
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53', 'EGFR']
)

# OpenSpliceAI
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['gene-BRCA1', 'gene-TP53', 'gene-EGFR']
)
```

### Example 2: Chromosome-Level Analysis

```python
# SpliceAI on chromosome 21
results = run_base_model_predictions(
    base_model='spliceai',
    target_chromosomes=['21']
)

# OpenSpliceAI on chromosome 21
results = run_base_model_predictions(
    base_model='openspliceai',
    target_chromosomes=['21']
)
```

### Example 3: Custom Configuration

```python
from meta_spliceai import BaseModelConfig, run_base_model_predictions

# Create custom config
config = BaseModelConfig(
    base_model='openspliceai',
    mode='test',
    coverage='gene_subset',
    test_name='my_custom_test'
)

# Run with custom config
results = run_base_model_predictions(
    config=config,
    target_genes=['gene-BRCA1']
)
```

### Example 4: Explicit Build Override

```python
from meta_spliceai import BaseModelConfig

# Override default build (advanced use case)
config = BaseModelConfig(
    base_model='spliceai',
    gtf_file='/path/to/custom.gtf',
    genome_fasta='/path/to/custom.fa'
)
```

---

## Technical Details

### GTF Parser Logic

```python
def extract_genes_from_gtf(gtf_file_path):
    # Load GTF
    gtf_df = pl.read_csv(gtf_file_path, ...)
    
    # Try standard Ensembl format (explicit gene features)
    gene_df = gtf_df.filter(pl.col('feature') == 'gene')
    
    # If no genes found, use MANE format (derive from transcripts)
    if gene_df.height == 0:
        transcript_df = gtf_df.filter(pl.col('feature') == 'transcript')
        
        # Aggregate by gene_id
        gene_df = transcript_df.group_by('gene_id').agg([
            pl.col('gene_name').first(),
            pl.col('seqname').first(),
            pl.col('start').min(),  # Gene start = min transcript start
            pl.col('end').max(),    # Gene end = max transcript end
            pl.col('strand').first()
        ])
        
        # Normalize chromosome names (chr1 → 1)
        gene_df = gene_df.with_columns([
            pl.col('seqname').str.replace('^chr', '')
        ])
        
        # Filter to standard chromosomes only
        standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        gene_df = gene_df.filter(pl.col('seqname').is_in(standard_chroms))
    
    return gene_df
```

### Artifact Manager Routing

```python
def get_artifact_manager(self):
    # Infer build and source from base_model
    if self.base_model == 'openspliceai':
        build = 'GRCh38'
        source = 'mane'
    else:  # spliceai or default
        build = 'GRCh37'
        source = 'ensembl'
    
    # Override source if explicitly specified in GTF path
    if self.gtf_file:
        if 'mane' in self.gtf_file.lower():
            source = 'mane'
        elif 'gencode' in self.gtf_file.lower():
            source = 'gencode'
    
    return create_artifact_manager_from_workflow_config(
        mode=self.mode,
        coverage=self.coverage,
        source=source,
        build=build,
        base_model=self.base_model,
        test_name=self.test_name,
        data_root=data_root
    )
```

---

## Directory Structure

### SpliceAI (GRCh37/Ensembl)

```
data/ensembl/GRCh37/
├── Homo_sapiens.GRCh37.87.gtf
├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
├── splice_sites_enhanced.tsv (1,998,526 sites)
├── gene_features.tsv
├── gene_sequence_*.parquet (24 files)
└── spliceai_eval/
    ├── tests/
    │   └── {test_name}/
    │       └── meta_models/predictions/...
    └── production/
        └── meta_models/predictions/...
```

### OpenSpliceAI (GRCh38/MANE)

```
data/mane/GRCh38/
├── MANE.GRCh38.v1.3.refseq_genomic.gtf
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
├── splice_sites_enhanced.tsv (369,918 sites)
├── gene_features.tsv
├── gene_sequence_*.parquet (24 files)
└── openspliceai_eval/
    ├── tests/
    │   └── {test_name}/
    │       └── meta_models/predictions/...
    └── production/
        └── meta_models/predictions/...
```

---

## Benefits

### For Users
- ✅ **Simple API**: One parameter to switch models
- ✅ **No manual configuration**: Automatic resource routing
- ✅ **Consistent interface**: Same API for all models
- ✅ **Clear separation**: Test vs production artifacts

### For Developers
- ✅ **Easy to extend**: Add new models with minimal code
- ✅ **Maintainable**: Centralized routing logic
- ✅ **Testable**: Comprehensive test suite
- ✅ **Documented**: Clear architecture and examples

### For Meta-Learning
- ✅ **Model-agnostic**: Works with any base model
- ✅ **Consistent features**: Standardized output format
- ✅ **Flexible**: Supports different genomic builds
- ✅ **Scalable**: Easy to add new base models

---

## Troubleshooting

### Issue: "Gene features not found"

**Solution**: Extract gene features for the build:
```bash
# For GRCh37/Ensembl
python meta_spliceai/splice_engine/extract_genomic_features.py

# For GRCh38/MANE
python meta_spliceai/splice_engine/extract_gene_features_mane.py
```

### Issue: "Splice sites not found"

**Solution**: Derive splice sites:
```bash
# For GRCh37/Ensembl
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 --release 87

# For GRCh38/MANE
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh38_MANE --release 1.3
```

### Issue: "Sequence files not found"

**Solution**: Extract sequences:
```bash
# For GRCh37/Ensembl
./scripts/setup/download_grch37_data.sh

# For GRCh38/MANE
./scripts/setup/extract_grch38_mane_sequences.sh
```

---

## Related Documentation

- **Setup**: `docs/setup/GENOMIC_RESOURCE_DOWNLOAD_GUIDE.md`
- **Testing**: `docs/base_models/OPENSPLICEAI_TESTING_GUIDE.md`
- **Architecture**: `docs/development/BASE_MODEL_SELECTION_AND_ROUTING.md`
- **Validation**: `docs/base_models/GRCH38_MANE_VALIDATION_COMPLETE.md`

---

## Conclusion

✅ **The MetaSpliceAI framework now supports universal base model switching.**

Key achievements:
1. Single-parameter API for model selection
2. Automatic routing to correct genomic resources
3. Universal GTF parser (Ensembl + MANE + future formats)
4. Comprehensive validation and testing
5. Clear documentation and examples

The system is ready for:
- ✅ Production deployment
- ✅ Adding new base models
- ✅ Meta-learning layer integration
- ✅ Multi-model ensemble predictions

---

**Status**: ✅ PRODUCTION READY  
**Tested**: SpliceAI (GRCh37/Ensembl) + OpenSpliceAI (GRCh38/MANE)  
**Extensible**: Easy to add new base models

*Last Updated: 2025-11-06*


