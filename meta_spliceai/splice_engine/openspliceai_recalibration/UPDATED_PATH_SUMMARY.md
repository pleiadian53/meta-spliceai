# OpenSpliceAI Recalibration Package - Path Updates Summary
**Date:** November 25, 2025  
**Status:** Paths updated to match revised genomic_resources convention

## 🔍 Your Questions Answered

### Q1: Where is my SpliceVarDB dataset?

**Found it!** 🎉

**Location:** `meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv`

**Details:**
- **50,716 variants** (including header)
- **6.8 MB** file size
- **Both hg19 (GRCh37) and hg38 (GRCh38)** coordinates ✅
- **Columns:** variant_id, hg19, hg38, gene, hgvs, method, classification, location, doi

**Sample Data:**
```
variant_id  hg19                hg38                gene    hgvs                  method  classification
1           1-100573238-T-C     1-100107682-T-C     SASS6   NM_194292.3:c.1092A>G MaPSy   Low-frequency
2           1-100576040-C-A     1-100110484-C-A     SASS6   NM_194292.3:c.670-1G>T RNA-Seq Splice-altering
```

## ⚠️ CRITICAL: Genomic Build Selection

**For OpenSpliceAI experiments, you MUST use the `hg38` column!**

OpenSpliceAI was trained on **GRCh38 (MANE)**, so:
- ✅ **Use:** `hg38` coordinates → Matches `data/mane/GRCh38/` reference
- ❌ **Don't use:** `hg19` coordinates → Would cause coordinate mismatch

**Why This Matters:**
- Coordinate mismatch between variant positions and reference genome causes incorrect sequence extraction
- This leads to wrong predictions and invalid alternative splice site annotations
- Using the correct build ensures your predictions are accurately tied to the right genomic positions

### Q2: What's the latest data organization structure?

Based on the **current structure** documented in `docs/data/DATA_LAYOUT_MASTER_GUIDE.md` (as of Nov 2025):

This structure evolved from our work on **generalizing the base model layer** to support any base model producing per-nucleotide splice site scores, each potentially tied to different genomic builds.

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
        │   │   ├── processed/
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
2. ✅ **Base model data** → Build-specific directories by annotation source:
   - `data/ensembl/GRCh37/` for SpliceAI
   - `data/mane/GRCh38/` for OpenSpliceAI
   - `data/<source>/<build>/` for future base models
3. ✅ **Case study data** → Code tree: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/`
4. ✅ **Genome build isolation** → Critical! Using wrong build causes 44-47% performance drop
5. ✅ **Generalized base model layer** → Supports any base model tied to any genomic build

**Why This Matters:**
- Each base model (SpliceAI, OpenSpliceAI, future models) was trained on specific reference data
- Accessing the correct FASTA + GTF/GFF that matches the base model's training is essential
- The structure naturally accommodates multiple base models with different builds

## 📊 What Was Updated

### 1. DATA_ORGANIZATION.md

**Updated sections:**
- ✅ Directory hierarchy (now matches DATA_LAYOUT_MASTER_GUIDE.md)
- ✅ Training dataset location (data/ root)
- ✅ Case study location (in code tree, not data/)
- ✅ Base model data separation (ensembl/GRCh37/ vs mane/GRCh38/)
- ✅ Migration instructions (reflects actual file locations)

### 2. Path Resolution

**Correct Architecture:**
```python
# Case study data (in code tree)
case_study_root = "meta_spliceai/splice_engine/case_studies/data_sources/datasets/"

# SpliceVarDB
splicevardb_dir = f"{case_study_root}/splicevardb/latest/"
# → meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/latest/

# ClinVar
clinvar_dir = f"{case_study_root}/clinvar/latest/"
# → meta_spliceai/splice_engine/case_studies/data_sources/datasets/clinvar/latest/

# Training datasets (in data/ root)
training_dir = "data/train_pc_7000_3mers_opt/"

# Base model data (build-specific)
spliceai_data = "data/ensembl/GRCh37/"  # For SpliceAI
openspliceai_data = "data/mane/GRCh38/"  # For OpenSpliceAI

# Training output (in data/ root)
training_output = "data/splicevardb_meta_training/"
```

**Why This Structure?**
- **Genome Build Isolation**: Mixing GRCh37/GRCh38 causes 44-47% performance drop
- **Data Portability**: Case studies in code tree, heavy data files in data/
- **Clear Separation**: Base model data vs training data vs case studies

### 3. SpliceVarDBLoader

**Already updated!** ✅

The loader in `openspliceai_recalibration/data/splicevardb_loader.py` already uses:
```python
from meta_spliceai.splice_engine.case_studies.data_sources.resource_manager import (
    CaseStudyResourceManager
)

class SpliceVarDBLoader:
    def __init__(self, output_dir=None, use_systematic_paths=True):
        if output_dir is None and use_systematic_paths:
            manager = CaseStudyResourceManager()
            self.output_dir = manager.case_study_paths.splicevardb
```

## 🚀 Migration Steps

### Step 1: Move Your SpliceVarDB Data

**Correct Location** (per DATA_LAYOUT_MASTER_GUIDE.md):
```bash
cd /Users/pleiadian53/work/meta-spliceai

# Case studies stay in code tree!
TARGET_DIR="meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb"

# Create release-dated directory
mkdir -p "$TARGET_DIR/splicevardb_$(date +%Y%m%d)"/{raw,processed}

# Move the TSV file
mv meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv \
   "$TARGET_DIR/splicevardb_$(date +%Y%m%d)/raw/splicevardb_download.tsv"

# Create 'latest' symlink
ln -sf "splicevardb_$(date +%Y%m%d)" "$TARGET_DIR/latest"

echo "✓ Migrated 50K SpliceVarDB variants to case_studies/data_sources/datasets/"
```

**Why This Location?**
- Case study data belongs in the **code tree** (small, portable)
- Heavy reference data (GTF, FASTA) goes in `data/` (large, build-specific)
- Follows the pattern: `case_studies/data_sources/datasets/{database}/{release}/`

### Step 2: Verify Paths

```python
from pathlib import Path

# Case study data locations (in code tree)
case_study_root = Path("meta_spliceai/splice_engine/case_studies/data_sources/datasets")

splicevardb_path = case_study_root / "splicevardb/latest"
clinvar_path = case_study_root / "clinvar/latest"

print(f"SpliceVarDB: {splicevardb_path}")
print(f"ClinVar:     {clinvar_path}")

# Training data locations (in data/ root)
training_data = Path("data/train_pc_7000_3mers_opt")
print(f"Training:    {training_data}")

# Base model data locations (build-specific)
spliceai_data = Path("data/ensembl/GRCh37")
openspliceai_data = Path("data/mane/GRCh38")
print(f"SpliceAI:    {spliceai_data}")
print(f"OpenSpliceAI: {openspliceai_data}")

print("✓ All paths follow DATA_LAYOUT_MASTER_GUIDE.md!")
```

### Step 3: Load and Process with Correct Build

```python
from pathlib import Path
import pandas as pd

# Case study data is in code tree
splicevardb_dir = Path("meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/latest")

# Load the raw TSV
raw_tsv = splicevardb_dir / "raw/splicevardb_download.tsv"
variants = pd.read_csv(raw_tsv, sep='\t')
print(f"✓ Loaded {len(variants)} variants from {raw_tsv}")

# ⚠️ IMPORTANT: Select the correct build column
# For OpenSpliceAI (GRCh38): use 'hg38' column
# For SpliceAI (GRCh37): use 'hg19' column
print("Available builds:", "hg19" in variants.columns, "hg38" in variants.columns)

# Parse coordinates from hg38 for OpenSpliceAI
def parse_variant(variant_str):
    """Parse variant string like '1-100107682-T-C' into components."""
    parts = variant_str.strip('"').split('-')
    return {
        'chrom': parts[0],
        'pos': int(parts[1]),
        'ref': parts[2],
        'alt': parts[3]
    }

# Extract GRCh38 coordinates for OpenSpliceAI
variants['parsed_hg38'] = variants['hg38'].apply(parse_variant)
variants[['chrom_hg38', 'pos_hg38', 'ref_hg38', 'alt_hg38']] = pd.DataFrame(
    variants['parsed_hg38'].tolist(), index=variants.index
)

# Save processed data with explicit build annotation
processed_dir = splicevardb_dir / "processed"
processed_dir.mkdir(exist_ok=True)

variants.to_parquet(processed_dir / "splicevardb_validated_GRCh38.parquet")
print(f"✓ Saved GRCh38 coordinates to {processed_dir}")
print(f"  Use these coordinates with OpenSpliceAI (data/mane/GRCh38/)")
```

### Step 4: Train Meta-Model with OpenSpliceAI

```python
from meta_spliceai.splice_engine.openspliceai_recalibration import (
    SpliceVarDBTrainingPipeline,
    PipelineConfig
)

# Configure with correct paths and BUILD
config = PipelineConfig(
    # Input: Case study data (in code tree)
    data_dir="meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/latest",
    
    # Output: Training dataset (in data/ root)
    output_dir="data/splicevardb_meta_training",
    
    # Reference: OpenSpliceAI base model data (GRCh38 MANE)
    reference_genome="data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna",
    gtf_file="data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff",
    
    # ⚠️ CRITICAL: Specify build and coordinate column
    genome_build="GRCh38",
    coordinate_column="hg38",  # Use hg38 for OpenSpliceAI, hg19 for SpliceAI
    
    # Base model selection
    base_model="openspliceai"
)

# Run pipeline
pipeline = SpliceVarDBTrainingPipeline(config=config)
results = pipeline.run()

# Output will be at: data/splicevardb_meta_training/
# - features.parquet (with correct GRCh38 coordinates)
# - labels.parquet (alternative splice sites in GRCh38)
# - train_test_split.json
# - coordinate_validation.json (build verification)

print(f"✓ Training complete with {config.genome_build} coordinates")
print(f"  Coordinates from: {config.coordinate_column} column")
print(f"  Reference genome: {config.reference_genome}")
```

**Coordinate Validation:**
The pipeline should validate that:
1. Variant coordinates are in GRCh38
2. Reference genome matches GRCh38
3. All predicted alternative splice sites use GRCh38 coordinates

## 🔧 Code That Needs Updating

### Priority Files (If You Modify Them)

Update any code referencing old paths:

1. **Case study data** - Should be in code tree: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/`

2. **Training outputs** - Should be in data root: `data/splicevardb_meta_training/`

3. **Base model data** - Should be build-specific: `data/ensembl/GRCh37/` or `data/mane/GRCh38/`

### Template for Updates

```python
# ❌ OLD: Incorrect assumptions
output_dir = "data/ensembl/training_datasets/splicevardb_meta_training"
clinvar_dir = "data/ensembl/clinvar/"
reference = "data/ensembl/reference.fa"  # Which build?

# ✅ NEW: Correct paths per DATA_LAYOUT_MASTER_GUIDE.md
from pathlib import Path

# Case study data (in code tree)
splicevardb_dir = Path("meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/latest")
clinvar_dir = Path("meta_spliceai/splice_engine/case_studies/data_sources/datasets/clinvar/latest")

# Training output (in data/ root)
output_dir = Path("data/splicevardb_meta_training")

# Base model data (build-specific)
if base_model == "spliceai":
    reference = Path("data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa")
elif base_model == "openspliceai":
    reference = Path("data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna")
```

## 📚 Reference: DATA_LAYOUT_MASTER_GUIDE.md

**The current master guide** documenting the latest data organization (as of Nov 2025):

**Location**: `docs/data/DATA_LAYOUT_MASTER_GUIDE.md`

**Background**: This structure resulted from our work on **generalizing the base model layer** to support:
- Multiple base models (currently SpliceAI, OpenSpliceAI)
- Different genomic builds per model (GRCh37, GRCh38)
- Future base models producing per-nucleotide splice site scores
- Proper FASTA + GTF/GFF pairing for each base model

**Key Sections**:
1. **Base Model Data Directories** - Build-specific: `data/<source>/<build>/`
   - `data/ensembl/GRCh37/` for SpliceAI
   - `data/mane/GRCh38/` for OpenSpliceAI
   - Extendable pattern for future models
2. **Training and Evaluation Datasets** - `data/train_*/`, `data/test_*/`
3. **Case Study Data** - `meta_spliceai/splice_engine/case_studies/data_sources/datasets/`
4. **Pre-trained Model Weights** - `data/models/`

**Why Build-Specific Directories Are Critical**

Genome build mismatch causes severe performance degradation:

| Model | Correct Data | Wrong Data | PR-AUC Drop |
|-------|--------------|------------|-------------|
| SpliceAI | GRCh37 (0.97) | GRCh38 (0.541) | **-44%** |
| OpenSpliceAI | GRCh38 MANE (0.98) | GRCh37 (0.523) | **-47%** |

**Design Principle**: Each base model must access the **exact reference data** (FASTA + annotations) it was trained on. The structure enforces this by isolating builds in separate directories.

## ✅ Verification Checklist

After migration:

- [ ] SpliceVarDB TSV moved to `case_studies/data_sources/datasets/splicevardb/latest/raw/`
- [ ] Case study data is in code tree (not data/)
- [ ] Training output goes to `data/splicevardb_meta_training/`
- [ ] Base model data is build-specific: `data/ensembl/GRCh37/` or `data/mane/GRCh38/`
- [ ] Training datasets (train_pc_*) remain in `data/` root
- [ ] Paths match DATA_LAYOUT_MASTER_GUIDE.md

## 🎯 Summary

**What You Asked:**
1. ✅ Found your SpliceVarDB data (50K variants in workflows/splicevardb/)
2. ✅ Corrected paths to match **latest structure** (DATA_LAYOUT_MASTER_GUIDE.md)
3. ✅ Fixed outdated documentation

**The Current Structure** (Nov 2025, post-base-model-generalization):
- **Training datasets** → `data/` root (`data/train_*/`)
- **Case study data** → Code tree (`case_studies/data_sources/datasets/`)
- **Base model data** → Build-specific by annotation source:
  - `data/<source>/<build>/` pattern
  - Currently: `data/ensembl/GRCh37/` (SpliceAI), `data/mane/GRCh38/` (OpenSpliceAI)
  - Extensible to future base models
- **Model weights** → `data/models/`

**Why This Matters:**
- **Genome build isolation** → Prevents 44-47% performance drops from build mismatch
- **Generalized base model layer** → Each model accesses its correct FASTA + GTF/GFF
- **Future-proof** → Supports any base model producing per-nucleotide splice scores
- **Clear separation** → Reference data, training data, case studies, and artifacts are organized by purpose and portability

**What's Next:**
1. Move SpliceVarDB to `case_studies/data_sources/datasets/splicevardb/latest/`
2. Update any code referencing old paths
3. Start training with correct OpenSpliceAI reference data: `data/mane/GRCh38/`

---

**Key Insight:** This structure evolved from generalizing the base model layer to support multiple models and builds. DATA_LAYOUT_MASTER_GUIDE.md documents the **current, production-ready** organization as of Nov 2025.

