# Annotation Source Directory Structure Proposal

**Date**: November 1, 2025  
**Status**: Proposal  
**Goal**: Generalize directory structure to support multiple annotation sources

---

## Current Problem

**Current Structure**:
```
data/ensembl/
├── GRCh38/
│   ├── Homo_sapiens.GRCh38.112.gtf
│   ├── splice_sites_enhanced.tsv
│   └── ...
└── GRCh37/
    ├── Homo_sapiens.GRCh37.87.gtf
    ├── splice_sites_enhanced.tsv
    └── ...
```

**Issue**: The directory name `ensembl` implies all data comes from Ensembl, but:
- **OpenSpliceAI** uses **MANE RefSeq** annotations (not Ensembl)
- Mixing different annotation sources under `ensembl/` is misleading
- Users can't tell which annotation source was used

---

## Proposed Solution

### Option A: Annotation Source as Top-Level Directory (RECOMMENDED)

**Structure**:
```
data/
├── ensembl/
│   ├── GRCh38/
│   │   ├── Homo_sapiens.GRCh38.112.gtf
│   │   ├── splice_sites_enhanced.tsv
│   │   └── ...
│   └── GRCh37/
│       ├── Homo_sapiens.GRCh37.87.gtf
│       ├── splice_sites_enhanced.tsv
│       └── ...
├── mane/
│   └── GRCh38/
│       ├── MANE.GRCh38.v1.3.refseq_genomic.gff
│       ├── splice_sites_enhanced.tsv
│       └── ...
└── gencode/
    ├── GRCh38/
    │   └── ...
    └── GRCh37/
        └── ...
```

**Benefits**:
- ✅ Clear separation by annotation source
- ✅ Easy to understand which data comes from where
- ✅ Prevents mixing incompatible annotations
- ✅ Supports multiple sources per build

**Drawbacks**:
- ⚠️ Requires updating all existing code that assumes `data/ensembl/`
- ⚠️ Need to migrate existing data

---

### Option B: Annotation Source as Build Suffix

**Structure**:
```
data/ensembl/
├── GRCh38_ensembl/
│   └── ...
├── GRCh38_mane/
│   └── ...
├── GRCh37_ensembl/
│   └── ...
└── GRCh37_gencode/
    └── ...
```

**Benefits**:
- ✅ Minimal code changes (still under `data/ensembl/`)
- ✅ Clear annotation source in build name

**Drawbacks**:
- ❌ Misleading top-level `ensembl/` directory
- ❌ Less intuitive structure
- ❌ Harder to manage multiple sources

---

### Option C: Hybrid Approach (BEST)

**Structure**:
```
data/
├── genomic_resources/  # New generic name
│   ├── ensembl/
│   │   ├── GRCh38/
│   │   └── GRCh37/
│   ├── mane/
│   │   └── GRCh38/
│   └── gencode/
│       ├── GRCh38/
│       └── GRCh37/
└── ensembl/  # Legacy symlink → genomic_resources/ensembl/
```

**Benefits**:
- ✅ Clear, intuitive structure
- ✅ Generic top-level name (`genomic_resources`)
- ✅ Backward compatibility via symlink
- ✅ Easy migration path

**Drawbacks**:
- ⚠️ Requires updating config and code
- ⚠️ Need migration script

---

## Recommended Approach: Option C (Hybrid)

### Phase 1: Update Configuration

**File**: `configs/genomic_resources.yaml`

**New Structure**:
```yaml
species: homo_sapiens
default_build: GRCh38
default_release: "112"
default_annotation_source: ensembl  # NEW
data_root: data/genomic_resources  # CHANGED

# Derived datasets configuration
derived_datasets:
  splice_sites: "splice_sites_enhanced.tsv"
  gene_features: "gene_features.tsv"
  transcript_features: "transcript_features.tsv"
  exon_features: "exon_features.tsv"
  annotations_db: "annotations.db"
  overlapping_genes: "overlapping_gene_counts.tsv"

# Annotation sources
annotation_sources:
  ensembl:
    name: "Ensembl"
    description: "Comprehensive genome annotation from Ensembl"
    url: "https://www.ensembl.org"
    
  mane:
    name: "MANE RefSeq"
    description: "Matched Annotation from NCBI and EMBL-EBI (RefSeq)"
    url: "https://www.ncbi.nlm.nih.gov/refseq/MANE/"
    
  gencode:
    name: "GENCODE"
    description: "Encyclopedia of genes and gene variants"
    url: "https://www.gencodegenes.org"

# Builds with annotation source specification
builds:
  GRCh38:
    annotation_source: ensembl  # NEW
    gtf: "Homo_sapiens.GRCh38.{release}.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    ensembl_base: "https://ftp.ensembl.org/pub/release-{release}"
    subpaths:
      gtf: "gtf/homo_sapiens"
      fasta: "fasta/homo_sapiens/dna"
      
  GRCh37:
    annotation_source: ensembl  # NEW
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    fasta: "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    ensembl_base: "https://ftp.ensembl.org/pub/grch37/release-{release}"
    subpaths:
      gtf: "gtf/homo_sapiens"
      fasta: "fasta/homo_sapiens/dna"
  
  GRCh38_MANE:  # NEW
    annotation_source: mane  # NEW
    gtf: "MANE.GRCh38.v{release}.refseq_genomic.gff"
    fasta: "GCF_000001405.40_GRCh38.p14_genomic.fna"
    base_url: "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_{release}"
    subpaths:
      gtf: ""  # Files are at root of release directory
      fasta: ""
    notes: "MANE v1.3 uses RefSeq GFF format, not GTF"
```

### Phase 2: Update Config Dataclass

**File**: `meta_spliceai/system/genomic_resources/config.py`

**Changes**:
```python
@dataclass
class Config:
    """Configuration for genomic resources."""
    species: str
    build: str
    release: str
    data_root: Path
    builds: dict
    derived_datasets: dict = None
    annotation_sources: dict = None  # NEW
    default_annotation_source: str = "ensembl"  # NEW
    
    def get_annotation_source(self, build: str = None) -> str:
        """Get annotation source for a build."""
        if build is None:
            build = self.build
        return self.builds[build].get("annotation_source", self.default_annotation_source)
    
    def get_data_dir(self, build: str = None, annotation_source: str = None) -> Path:
        """Get data directory for a build and annotation source."""
        if build is None:
            build = self.build
        if annotation_source is None:
            annotation_source = self.get_annotation_source(build)
        
        # data_root / annotation_source / build
        return self.data_root / annotation_source / build
```

**Update `load_config()`**:
```python
def load_config(path: str = None) -> Config:
    # ... existing code ...
    
    return Config(
        species=os.getenv("SS_SPECIES", y["species"]),
        build=os.getenv("SS_BUILD", y["default_build"]),
        release=os.getenv("SS_RELEASE", y["default_release"]),
        data_root=data_root,
        builds=y["builds"],
        derived_datasets=y.get("derived_datasets", {}),
        annotation_sources=y.get("annotation_sources", {}),  # NEW
        default_annotation_source=y.get("default_annotation_source", "ensembl")  # NEW
    )
```

### Phase 3: Update Registry

**File**: `meta_spliceai/system/genomic_resources/registry.py`

**Changes**:
```python
class Registry:
    def __init__(self, build: Optional[str] = None, release: Optional[str] = None):
        self.cfg = load_config()
        if build:
            self.cfg.build = build
        if release:
            self.cfg.release = release
        
        # Get annotation source for this build
        self.annotation_source = self.cfg.get_annotation_source(self.cfg.build)
        
        # Build directory structure: data_root / annotation_source / build
        self.top = self.cfg.data_root / self.annotation_source
        self.stash = self.top / self.cfg.build
        self.legacy = self.top / "spliceai_analysis"  # Keep for backward compat
        
        # Build-specific directories
        self.data_dir = self.stash
        self.eval_dir = self.stash / "spliceai_eval"
        self.analysis_dir = self.stash / "spliceai_analysis"
```

### Phase 4: Migration Script

**File**: `scripts/migration/migrate_to_annotation_source_structure.py`

**Purpose**: Migrate existing data from old structure to new structure

```python
#!/usr/bin/env python3
"""
Migrate existing data to new annotation-source-based directory structure.

Old: data/ensembl/GRCh38/
New: data/genomic_resources/ensembl/GRCh38/
"""

import shutil
from pathlib import Path

def migrate_data():
    """Migrate data to new structure."""
    
    old_root = Path("data/ensembl")
    new_root = Path("data/genomic_resources/ensembl")
    
    if not old_root.exists():
        print("No old data to migrate")
        return
    
    print(f"Migrating from {old_root} to {new_root}")
    
    # Create new directory structure
    new_root.mkdir(parents=True, exist_ok=True)
    
    # Move all subdirectories
    for item in old_root.iterdir():
        if item.is_dir():
            dest = new_root / item.name
            if dest.exists():
                print(f"  Skipping {item.name} (already exists)")
            else:
                print(f"  Moving {item.name}")
                shutil.move(str(item), str(dest))
    
    # Create backward compatibility symlink
    if not old_root.exists():
        old_root.symlink_to(new_root)
        print(f"Created symlink: {old_root} -> {new_root}")
    
    print("Migration complete!")

if __name__ == "__main__":
    migrate_data()
```

---

## Implementation Plan

### Step 1: Update Configuration (30 minutes)
- [ ] Update `genomic_resources.yaml` with new structure
- [ ] Add `annotation_sources` section
- [ ] Add `annotation_source` to each build
- [ ] Change `data_root` to `data/genomic_resources`

### Step 2: Update Config Module (1 hour)
- [ ] Add `annotation_sources` to Config dataclass
- [ ] Add `get_annotation_source()` method
- [ ] Add `get_data_dir()` method
- [ ] Update `load_config()` to load new fields

### Step 3: Update Registry (1 hour)
- [ ] Update `__init__` to use annotation source
- [ ] Update path construction logic
- [ ] Test with existing builds

### Step 4: Create Migration Script (30 minutes)
- [ ] Write migration script
- [ ] Test on copy of data
- [ ] Document migration process

### Step 5: Test and Validate (2 hours)
- [ ] Test GRCh37 workflow still works
- [ ] Test GRCh38 workflow still works
- [ ] Verify all paths resolve correctly
- [ ] Check backward compatibility

### Step 6: Add MANE Support (2 hours)
- [ ] Add GRCh38_MANE build to config
- [ ] Test MANE data download
- [ ] Verify directory structure

**Total Time**: ~7 hours

---

## Backward Compatibility

### Symlink Approach
```bash
# Create symlink for backward compatibility
ln -s data/genomic_resources/ensembl data/ensembl
```

This allows:
- Old code using `data/ensembl/` continues to work
- New code uses `data/genomic_resources/ensembl/`
- Gradual migration of code

### Environment Variable Override
```bash
# Old behavior (if needed)
export SS_DATA_ROOT=data/ensembl

# New behavior (default)
export SS_DATA_ROOT=data/genomic_resources
```

---

## Benefits of This Approach

1. **Clear Separation**: Each annotation source has its own directory
2. **Intuitive**: `data/genomic_resources/mane/GRCh38/` is self-documenting
3. **Flexible**: Easy to add new annotation sources
4. **Backward Compatible**: Symlink preserves old paths
5. **Future-Proof**: Supports multiple sources per build
6. **Organized**: Clear hierarchy: data_root → annotation_source → build

---

## Example Usage

### After Migration

```python
from meta_spliceai.system.genomic_resources import Registry

# Ensembl GRCh37
registry_ensembl = Registry(build="GRCh37")
# Resolves to: data/genomic_resources/ensembl/GRCh37/

# MANE GRCh38
registry_mane = Registry(build="GRCh38_MANE")
# Resolves to: data/genomic_resources/mane/GRCh38/

# Get paths
gtf_path = registry_mane.resolve("gtf")
# Returns: data/genomic_resources/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff
```

---

## Conclusion

**Recommendation**: Implement Option C (Hybrid Approach)

**Why**:
- Clear, intuitive directory structure
- Supports multiple annotation sources
- Backward compatible via symlink
- Minimal disruption to existing code
- Future-proof for additional sources

**Next Steps**:
1. Get approval for this approach
2. Implement configuration changes
3. Run migration script
4. Test thoroughly
5. Add MANE support
6. Document new structure



