# Data Organization for OpenSpliceAI Recalibration

## Integration with Genomic Resource Management

This package integrates with the **systematic genomic resource management system** at `meta_spliceai/system/genomic_resources/`.

## Systematic Data Organization

### Overview

All data follows the **authoritative structure** defined in `docs/data/DATA_LAYOUT_MASTER_GUIDE.md`:

```
data/
├── train_pc_1000_3mers/                   # ✅ Training datasets at root
├── train_pc_7000_3mers_opt/
├── splicevardb_meta_training/             # ✅ NEW: Your training output
├── models/                                # Pre-trained model weights
│
├── ensembl/                               # SpliceAI base model data
│   └── GRCh37/                           # ✅ Build-specific
│       ├── Homo_sapiens.GRCh37.87.gtf
│       ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
│       ├── splice_sites_enhanced.tsv
│       └── spliceai_eval/                # SpliceAI outputs
│
└── mane/                                  # OpenSpliceAI base model data
    └── GRCh38/                           # ✅ Build-specific
        ├── MANE.GRCh38.v1.3.refseq_genomic.gff
        ├── GCF_000001405.40_GRCh38.p14_genomic.fna
        ├── splice_sites_enhanced.tsv
        └── openspliceai_eval/            # OpenSpliceAI outputs

meta_spliceai/splice_engine/case_studies/  # ✅ Case studies in code tree
└── data_sources/
    └── datasets/
        ├── splicevardb/                   # SpliceVarDB
        │   ├── splicevardb_20250915/     # Release-dated
        │   │   ├── raw/
        │   │   │   ├── variants.jsonl
        │   │   │   └── download_metadata.json
        │   │   ├── processed/
        │   │   │   ├── splicevardb_validated_GRCh38.parquet
        │   │   │   ├── splicevardb_validated_GRCh38.tsv
        │   │   │   └── splicevardb_validated_GRCh38.vcf.gz
        │   │   └── README.md
        │   └── latest -> splicevardb_20250915/
        │
        └── clinvar/                       # ClinVar
            ├── clinvar_20250831/
            │   ├── raw/
            │   ├── processed/
            │   └── README.md
            └── latest -> clinvar_20250831/
```

**Key Architecture Principles:**
1. ✅ **Training datasets** → `data/` root
2. ✅ **Base model data** → Build-specific: `data/ensembl/GRCh37/`, `data/mane/GRCh38/`
3. ✅ **Case study data** → Code tree: `case_studies/data_sources/datasets/`
4. ✅ **Genome build isolation** → Prevents 44-47% performance drop from build mismatch

## Updated SpliceVarDB Loader

The loader now uses the systematic path structure:

```python
from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBLoader

# Use Registry to get data root
registry = Registry()
data_root = registry.cfg.data_root  # data/ensembl

# Initialize loader with systematic paths
loader = SpliceVarDBLoader(
    output_dir=data_root / "case_studies" / "splicevardb"
)

# Data organization:
# - Raw downloads: {output_dir}/raw/
# - Processed data: {output_dir}/processed/
# - Cache: {output_dir}/cache/
variants_df = loader.load_validated_variants(build="GRCh38")
```

## Path Resolution

### Using CaseStudyResourceManager

```python
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)

# Initialize manager
manager = CaseStudyResourceManager(
    genome_build="GRCh38",
    ensembl_release=112
)

# Get systematic paths
splicevardb_dir = manager.case_study_paths.splicevardb
# → data/ensembl/case_studies/splicevardb/

clinvar_dir = manager.case_study_paths.clinvar
# → data/ensembl/case_studies/clinvar/

# Also get reference genome paths
fasta_path = manager.get_fasta_path()
# → data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa

gtf_path = manager.get_gtf_path()
# → data/ensembl/Homo_sapiens.GRCh38.112.gtf
```

## Integration with OpenSpliceAI Recalibration

### Updated Configuration

```python
from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)
from meta_spliceai.splice_engine.openspliceai_recalibration import (
    SpliceVarDBTrainingPipeline,
    PipelineConfig
)

# Get systematic paths
manager = CaseStudyResourceManager()
registry = Registry()

# Create pipeline config with systematic paths
config = PipelineConfig(
    # Data source
    data_dir=str(manager.case_study_paths.splicevardb),
    
    # Reference genome
    reference_genome=str(manager.get_fasta_path()),
    
    # Training output (stays in data/ root, not under ensembl/)
    output_dir=str(registry.cfg.data_root.parent / "splicevardb_meta_training"),
    
    # Build and release
    genome_build="GRCh38",
)

# Run pipeline
pipeline = SpliceVarDBTrainingPipeline(config=config)
results = pipeline.run()
```

## Migration from Old Locations

### Step 1: Move SpliceVarDB Downloaded Data (50K variants!)

```bash
# Your downloaded SpliceVarDB data (50,716 variants, 6.8MB)
old_splicevardb="meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv"
new_splicevardb="data/ensembl/case_studies/splicevardb/raw/"

# Move the TSV file
mkdir -p "$new_splicevardb"
mv "$old_splicevardb" "$new_splicevardb/splicevardb_download.tsv"
echo "✓ Moved 50K SpliceVarDB variants to systematic location"
```

### Step 2: Move ClinVar (if exists)

```bash
# Move ClinVar from old location to case_studies/
old_clinvar_data="data/ensembl/clinvar/"
new_clinvar="data/ensembl/case_studies/clinvar/"

if [ -d "$old_clinvar_data" ]; then
    mkdir -p "data/ensembl/case_studies/"
    mv "$old_clinvar_data" "$new_clinvar"
    echo "✓ Moved ClinVar to case_studies/ (same level as splicevardb)"
fi

# Move ClinVar docs from code directory (if exists)
old_clinvar_docs="meta_spliceai/splice_engine/case_studies/data_sources/datasets/clinvar_20250831/"
if [ -d "$old_clinvar_docs" ]; then
    mkdir -p "$new_clinvar/"
    mv "$old_clinvar_docs" "$new_clinvar/clinvar_20250831"
    echo "✓ Moved ClinVar documentation"
fi
```

### Step 3: Training Datasets (Already in Correct Location!)

```bash
# Training datasets are already in the RIGHT place: data/train_*
# NO NEED TO MOVE:
# - data/train_pc_1000_3mers/      ✅ Stay here
# - data/train_pc_7000_3mers_opt/  ✅ Stay here
# 
# New training data should also go in data/ root:
# - data/splicevardb_meta_training/  ✅ NEW
```

## Directory Structure Details

### SpliceVarDB Directory

```
data/ensembl/case_studies/splicevardb/
├── raw/                                   # Raw downloads from API
│   ├── variants.jsonl                    # JSONL from paginated API
│   ├── variants_page_*.json              # Individual page cache
│   └── download_metadata.json            # Download timestamp, counts, etc.
│
├── processed/                             # Processed formats
│   ├── splicevardb_validated_GRCh38.parquet  # Primary format
│   ├── splicevardb_validated_GRCh38.tsv      # TSV export
│   ├── splicevardb_validated_GRCh38.vcf.gz   # VCF export
│   └── processing_log.json
│
├── cache/                                 # Cache directory
│   ├── api_responses/                    # Cached API responses
│   └── checksums.txt
│
└── README.md                              # Dataset documentation
```

### ClinVar Directory (Under case_studies/)

```
data/ensembl/case_studies/clinvar/         # ✅ Same level as splicevardb
├── clinvar_20250831/                      # Release-dated
│   ├── raw/
│   │   └── clinvar_20250831.vcf.gz
│   ├── processed/
│   │   ├── clinvar_variants.parquet
│   │   └── clinvar_variants.tsv
│   ├── docs/
│   └── README.md
│
└── latest -> clinvar_20250831/            # Symlink to latest
```

### Training Datasets Directory (In data/ Root)

```
data/                                      # ✅ Training datasets at root
├── train_pc_1000_3mers/                   # Existing training data
├── train_pc_7000_3mers_opt/
│
├── splicevardb_meta_training/             # ✅ NEW: From our pipeline
│   ├── features.parquet                  # Delta scores + context
│   ├── labels.parquet                    # Splice-altering labels
│   ├── metadata.json                     # Dataset info
│   └── README.md
│
├── train_pc_7000_3mers_opt/              # Existing training dataset
└── train_regulatory_enhanced_kmers/       # Existing training dataset
```

## Benefits of Systematic Organization

1. **Consistency**: All external data follows same pattern
2. **Traceability**: Clear separation of raw, processed, and training data
3. **Reusability**: Multiple workflows can access same source data
4. **Maintainability**: Easy to understand where data lives
5. **Scalability**: Easy to add new variant databases
6. **Integration**: Works with existing genomic resource management

## Configuration Updates

### Update genomic_resources.yaml

Add variant database paths to configuration:

```yaml
# configs/genomic_resources.yaml

derived_datasets:
  splice_sites: splice_sites_enhanced.tsv
  gene_features: gene_features.tsv
  # ... existing ...

# NEW: Variant database paths
variant_databases:
  clinvar: clinvar
  splicevardb: case_studies/splicevardb
  mutsplicedb: case_studies/mutsplicedb
  dbass: case_studies/dbass

# NEW: Training dataset paths  
training_datasets:
  root: data                                # ✅ Training datasets in data/ root
  splicevardb_meta: data/splicevardb_meta_training
  pc_1000: data/train_pc_1000_3mers
  pc_7000: data/train_pc_7000_3mers_opt
```

## Usage Examples

### Example 1: Load SpliceVarDB with Systematic Paths

```python
from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBLoader
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)

# Get systematic path
manager = CaseStudyResourceManager()
splicevardb_dir = manager.case_study_paths.splicevardb

# Load data
loader = SpliceVarDBLoader(output_dir=splicevardb_dir)
variants_df = loader.load_validated_variants()

# Data is at: data/ensembl/case_studies/splicevardb/processed/
print(f"Data location: {splicevardb_dir}/processed/")
```

### Example 2: Training Pipeline with Systematic Paths

```python
from meta_spliceai.splice_engine.openspliceai_recalibration import SpliceVarDBTrainingPipeline
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)
from meta_spliceai.system.genomic_resources import Registry

# Get paths
manager = CaseStudyResourceManager()
registry = Registry()

# Run pipeline
pipeline = SpliceVarDBTrainingPipeline(
    data_dir=str(manager.case_study_paths.splicevardb),
    output_dir=str(registry.cfg.data_root / "training_datasets" / "splicevardb_meta_training")
)
results = pipeline.run()

# Output at: data/ensembl/training_datasets/splicevardb_meta_training/
```

### Example 3: Access All Resources

```python
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)

manager = CaseStudyResourceManager()

# Reference genome
fasta = manager.get_fasta_path()
gtf = manager.get_gtf_path()

# Variant databases
splicevardb = manager.case_study_paths.splicevardb
clinvar = manager.case_study_paths.clinvar
mutsplicedb = manager.case_study_paths.mutsplicedb

# Analysis outputs
processed = manager.case_study_paths.processed
results = manager.case_study_paths.results

print(f"All resources under: {manager.genomic_manager.genome.base_data_dir}")
```

## Summary

**Old (Inconsistent):**
```
✗ meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv
✗ meta_spliceai/splice_engine/case_studies/data_sources/datasets/clinvar_20250831/
```

**New (Systematic):**
```
✓ data/ensembl/case_studies/splicevardb/processed/splicevardb_validated_GRCh38.parquet
✓ data/ensembl/clinvar/clinvar_20250831/
✓ data/ensembl/training_datasets/splicevardb_meta_training/
```

**Key Changes:**
1. All external data under `data/ensembl/`
2. Variant databases under `data/ensembl/case_studies/`
3. Training datasets under `data/ensembl/training_datasets/`
4. Use `CaseStudyResourceManager` for path resolution
5. Separate raw, processed, and training data





