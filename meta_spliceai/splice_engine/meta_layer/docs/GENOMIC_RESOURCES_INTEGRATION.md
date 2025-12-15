# Genomic Resources Integration

**Last Updated**: December 2025  
**Status**: ✅ Complete

---

## Overview

The meta_layer package integrates with the `genomic_resources` system for consistent, flexible path resolution across all base models. This document explains the integration and how to extend it for new base models.

---

## Architecture

### Before Integration (Hardcoded ❌)

```python
# OLD: Hardcoded paths in meta_layer/core/config.py
_ARTIFACT_PATHS = {
    'spliceai': 'data/ensembl/GRCh37/spliceai_eval/meta_models',
    'openspliceai': 'data/mane/GRCh38/openspliceai_eval/meta_models',
}
```

**Problems:**
- Not scalable for new base models
- Duplicates logic from genomic_resources
- Won't update if config changes

### After Integration (Dynamic ✅)

```python
# NEW: Uses genomic_resources Registry
from meta_spliceai.system.genomic_resources import Registry, load_config

config = load_config()
registry = Registry(build=config.get_base_model_build('openspliceai'))

# Dynamically resolved paths
artifacts_dir = registry.get_meta_models_artifact_dir('openspliceai')
# → data/mane/GRCh38/openspliceai_eval/meta_models/
```

---

## Path Resolution Flow

```
MetaLayerConfig(base_model='openspliceai')
              ↓
genomic_resources.load_config()
              ↓
config.get_base_model_build('openspliceai')
              ↓ returns 'GRCh38'
Registry(build='GRCh38_MANE')
              ↓
registry.get_meta_models_artifact_dir('openspliceai')
              ↓
data/mane/GRCh38/openspliceai_eval/meta_models/
```

---

## Key Components

### 1. genomic_resources/config.py

Added methods for base model information:

```python
class Config:
    def get_base_model_build(self, model_name: str) -> str:
        """Get genomic build for a base model."""
        info = self.get_base_model_info(model_name.lower())
        return info.get('training_build', self.build)
    
    def get_base_model_annotation_source(self, model_name: str) -> str:
        """Get annotation source for a base model."""
        info = self.get_base_model_info(model_name.lower())
        return info.get('annotation_source', self.default_annotation_source)
```

### 2. genomic_resources/registry.py

Added methods for base model artifact paths:

```python
class Registry:
    def get_base_model_eval_dir(self, base_model: str, create: bool = False) -> Path:
        """Get evaluation directory for a specific base model.
        
        Examples:
        - data/ensembl/GRCh37/spliceai_eval/
        - data/mane/GRCh38/openspliceai_eval/
        """
        return self.stash / f'{base_model.lower()}_eval'
    
    def get_meta_models_artifact_dir(self, base_model: str, create: bool = False) -> Path:
        """Get meta_models artifact directory for a specific base model.
        
        Examples:
        - data/ensembl/GRCh37/spliceai_eval/meta_models/
        - data/mane/GRCh38/openspliceai_eval/meta_models/
        """
        return self.get_base_model_eval_dir(base_model) / 'meta_models'
```

### 3. meta_layer/core/config.py

Updated to use genomic_resources:

```python
@dataclass
class MetaLayerConfig:
    base_model: str = 'openspliceai'
    
    # Internal cache for genomic resources
    _genomic_config: Optional[object] = field(default=None, repr=False)
    _registry: Optional[object] = field(default=None, repr=False)
    
    def _init_genomic_resources(self):
        """Initialize genomic resources configuration and registry."""
        from meta_spliceai.system.genomic_resources import load_config, Registry
        
        self._genomic_config = load_config()
        build = self._genomic_config.get_base_model_build(self.base_model)
        self._registry = Registry(build=build)
    
    @property
    def artifacts_dir(self) -> Path:
        """Get artifact directory using Registry."""
        return self._registry.get_meta_models_artifact_dir(self.base_model)
    
    @property
    def genome_build(self) -> str:
        """Get genome build from config."""
        return self._genomic_config.get_base_model_build(self.base_model)
```

---

## Configuration: genomic_resources.yaml

The base model configuration lives in `configs/genomic_resources.yaml`:

```yaml
# Base models and their training specifications
base_models:
  spliceai:
    name: "SpliceAI"
    training_build: "GRCh37"
    training_annotation: "GENCODE V24lift37"
    annotation_source: "ensembl"  # Uses data/ensembl/GRCh37/
    
  openspliceai:
    name: "OpenSpliceAI"
    training_build: "GRCh38"
    training_annotation: "MANE v1.3 RefSeq"
    annotation_source: "mane"  # Uses data/mane/GRCh38/
    
  # Future base models
  # newsplicemodel:
  #   name: "NewSpliceModel"
  #   training_build: "GRCh38"
  #   annotation_source: "gencode"
```

---

## Adding a New Base Model

To add support for a new base model:

### Step 1: Add to genomic_resources.yaml

```yaml
base_models:
  # ...existing models...
  
  newsplicemodel:
    name: "NewSpliceModel"
    training_build: "GRCh38"
    training_annotation: "GENCODE v42"
    annotation_source: "gencode"
```

### Step 2: Ensure build configuration exists

```yaml
builds:
  # ...existing builds...
  
  GRCh38_GENCODE:
    annotation_source: gencode
    gtf: "gencode.v42.annotation.gtf"
    fasta: "GRCh38.primary_assembly.genome.fa"
```

### Step 3: Run base layer

```bash
run_base_model --base-model newsplicemodel --mode production
# Creates: data/gencode/GRCh38/newsplicemodel_eval/meta_models/
```

### Step 4: Use in meta_layer

```python
from meta_spliceai.splice_engine.meta_layer import MetaLayerConfig

# Automatically uses correct paths!
config = MetaLayerConfig(base_model='newsplicemodel')
print(config.artifacts_dir)
# → data/gencode/GRCh38/newsplicemodel_eval/meta_models/
```

**No code changes needed in meta_layer!** ✅

---

## Directory Structure

```
data/
├── ensembl/
│   └── GRCh37/
│       ├── Homo_sapiens.GRCh37.87.gtf
│       ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
│       └── spliceai_eval/
│           └── meta_models/
│               ├── analysis_sequences_*.tsv
│               └── splice_errors_*.tsv
│
├── mane/
│   └── GRCh38/
│       ├── MANE.GRCh38.v1.3.refseq_genomic.gtf
│       ├── GCF_000001405.40_GRCh38.p14_genomic.fna
│       └── openspliceai_eval/
│           └── meta_models/
│               ├── analysis_sequences_*.tsv
│               └── splice_errors_*.tsv
│
└── gencode/  # Future
    └── GRCh38/
        └── newsplicemodel_eval/
            └── meta_models/
```

---

## Verification

Run the verification script to confirm integration:

```bash
mamba activate metaspliceai
python meta_spliceai/splice_engine/meta_layer/examples/verify_artifacts.py
```

Expected output:

```
============================================================
Meta-Layer Artifact Verification
============================================================

This script verifies that the meta_layer package correctly
integrates with the genomic_resources system for path resolution.

============================================================
Base Model: OPENSPLICEAI
============================================================

Configuration (from genomic_resources):
  Artifacts dir: /path/to/data/mane/GRCh38/openspliceai_eval/meta_models
  Annotation source: mane
  Genome build: GRCh38
  Coordinate column: hg38

✅ openspliceai artifacts verified successfully!

============================================================
Base Model: SPLICEAI
============================================================

Configuration (from genomic_resources):
  Artifacts dir: /path/to/data/ensembl/GRCh37/spliceai_eval/meta_models
  Annotation source: ensembl
  Genome build: GRCh37
  Coordinate column: hg19

✅ spliceai artifacts verified successfully!
```

---

## Benefits

### 1. Consistency
- All packages use the same path resolution logic
- No duplicate path definitions

### 2. Flexibility
- Add new base models via configuration
- No code changes needed

### 3. Scalability
- Supports unlimited base models
- Each with its own build and annotation source

### 4. Maintainability
- Single source of truth: `genomic_resources.yaml`
- Changes propagate automatically

---

## Related Documentation

- **genomic_resources README**: `meta_spliceai/system/genomic_resources/README.md`
- **Base Model Support**: `docs/base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md`
- **Data Layout**: `docs/data/DATA_LAYOUT_MASTER_GUIDE.md`

---

*Last Updated: December 2025*






