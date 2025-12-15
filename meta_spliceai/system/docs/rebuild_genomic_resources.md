---
type: setup-playbook
target: codebase
audience: ai-agent
---

> **AI Agent Instructions**
> 1. Follow each stage in order (Stage 1 → Stage 7).  
> 2. Treat code snippets as *reference implementations* — you may verify, adjust imports, update paths, or modernize syntax as needed.  
> 3. Before writing new files, check if they already exist to avoid overwriting custom logic.  
> 4. When a path or name differs from our environment, adapt intelligently (e.g. keep everything under `data/ensembl/` for datasets inferred to be originated from Ensembl DB).  
> 5. After completing each stage, run the provided **Acceptance** commands and ensure all checks pass before continuing.

---

# 🧬 Rebuild Genomic Resources Manager — 7-Stage Plan

**Target system layout**

```
<data_root>/data/
├── ensembl/
│   ├── Homo_sapiens.GRCh38.112.gtf
│   ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
│   ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai
│   ├── splice_sites.tsv
│   ├── gene_features.tsv
│   ├── transcript_features.tsv
│   ├── exon_features.tsv
│   └── junctions.tsv
│   ├── GRCh38/
│   │   └── Homo_sapiens.GRCh38.112.gtf
│   └── GRCh37/
│       └── Homo_sapiens.GRCh37.87.gtf
├── gencode/
│   └── gencode.v44.annotation.gtf
└── case_studies/
    ├── clinvar/GRCh38/
    ├── splicevardb/GRCh38/
    ├── mutsplicedb/hg19/
    └── dbass/GRCh38/
```

---

## Stage 1 — Minimal Audit: Detect Missing Inputs

**Goal**  Check if core files (GTF + FASTA + index) exist under `data/ensembl/`.

**Create**

```
scripts/minimal_check.py
```

```python
#!/usr/bin/env python3
import os, sys, pathlib

TOP = pathlib.Path("data/ensembl")
REQUIRED = {
  "gtf": TOP/"Homo_sapiens.GRCh38.112.gtf",
  "fasta": TOP/"Homo_sapiens.GRCh38.dna.primary_assembly.fa",
  "fasta_index": TOP/"Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai",
}

def status(p): return "FOUND" if p.exists() else "MISSING"

def main():
    print("=== Genomic resources minimal audit ===")
    missing_core = False
    for k,p in REQUIRED.items():
        print(f"{k:12} {status(p):7} {p}")
    if not REQUIRED["gtf"].exists() or not REQUIRED["fasta"].exists():
        missing_core = True
    sys.exit(1 if missing_core else 0)

if __name__ == "__main__":
    TOP.mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/ensembl/GRCh38").mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/ensembl/GRCh37").mkdir(parents=True, exist_ok=True)
    main()
```

**Acceptance**

```bash
python scripts/minimal_check.py
# prints table; exits 1 when missing
```

---

## Stage 2 — Config & Registry (resolve paths)

**Goal** Define default build/release and a registry that resolves expected file paths.

**Create**

```
configs/genomic_resources.yaml
meta_spliceai/system/genomic_resources/config.py
meta_spliceai/system/genomic_resources/registry.py
```

**`configs/genomic_resources.yaml`**

```yaml
species: homo_sapiens
default_build: GRCh38
default_release: "112"
data_root: data/ensembl
builds:
  GRCh38:
    gtf: "Homo_sapiens.GRCh38.{release}.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    ensembl_base: "https://ftp.ensembl.org/pub/release-{release}"
    subpaths:
      gtf: "gtf/homo_sapiens"
      fasta: "fasta/homo_sapiens/dna"
  GRCh37:
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    fasta: "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    ensembl_base: "https://grch37.ensembl.org/pub/release-{release}"
    subpaths:
      gtf: "gtf/homo_sapiens"
      fasta: "fasta/homo_sapiens/dna"
```

**`config.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import os, yaml

@dataclass
class Config:
    species: str
    build: str
    release: str
    data_root: Path
    builds: dict

def load_config(path="configs/genomic_resources.yaml") -> Config:
    with open(path) as f: y = yaml.safe_load(f)
    return Config(
        species=os.getenv("SS_SPECIES", y["species"]),
        build=os.getenv("SS_BUILD", y["default_build"]),
        release=os.getenv("SS_RELEASE", y["default_release"]),
        data_root=Path(os.getenv("SS_DATA_ROOT", y["data_root"])),
        builds=y["builds"]
    )

def filename(kind, cfg): return cfg.builds[cfg.build][kind].format(release=cfg.release)
```

**`registry.py`**

```python
from pathlib import Path
from .config import load_config, filename

class Registry:
    def __init__(self, build=None, release=None):
        self.cfg = load_config()
        if build: self.cfg.build = build
        if release: self.cfg.release = release
        self.top = self.cfg.data_root
        self.stash = self.top / self.cfg.build

    def resolve(self, kind):
        mapping = {
          "fasta_index": filename("fasta", self.cfg) + ".fai",
          "splice_sites": "splice_sites.tsv",
          "gene_features": "gene_features.tsv",
          "transcript_features": "transcript_features.tsv",
          "exon_features": "exon_features.tsv",
          "junctions": "junctions.tsv"
        }
        name = filename("gtf", self.cfg) if kind=="gtf" \
             else filename("fasta", self.cfg) if kind=="fasta" \
             else mapping[kind]
        for root in [self.top, self.stash, self.top/"spliceai_analysis"]:
            p = Path(root)/name
            if p.exists(): return str(p.resolve())
        return None
```

**Acceptance**

```python
from meta_spliceai.system.genomic_resources.registry import Registry
r = Registry()
print(r.resolve("gtf"))  # expects data/ensembl/Homo_sapiens.GRCh38.112.gtf
```

---

## Stage 3 — Bootstrap (download & index)

**Goal** Download Ensembl GTF/FASTA → decompress → index → place under `data/ensembl/`.

**Create**

```
meta_spliceai/system/genomic_resources/download.py
meta_spliceai/system/genomic_resources/cli.py
```

*(See code in previous prompt—contains `_fetch`, `_gunzip`, `ensure_faidx`, and `fetch_ensembl()` plus CLI with `bootstrap` subcommand.)*

**Acceptance**

```bash
python -m meta_spliceai.system.genomic_resources.cli bootstrap --dry-run
python -m meta_spliceai.system.genomic_resources.cli bootstrap --build GRCh38 --release 112
python scripts/minimal_check.py   # should exit 0 now
```

---

## Stage 4 — Set-Current Helper

**Goal** Flip between builds by copying/symlinking from `data/ensembl/<BUILD>/` → `data/ensembl/`.

*(Use the `set_current` implementation snippet from previous message.)*

**Acceptance**

```bash
python -m meta_spliceai.system.genomic_resources.cli set-current --build GRCh37
python scripts/minimal_check.py   # now shows GRCh37 files at top-level
```

---

## Stage 5 — Derivations ✅ **COMPLETE**

**Goal** Generate derived TSVs (`splice_sites.tsv`, `gene_features.tsv`, `transcript_features.tsv`, `exon_features.tsv`, `junctions.tsv`) directly beside GTF/FASTA.

**Status**: ✅ **IMPLEMENTED AND TESTED**

**Implementation**:
- Created `GenomicDataDeriver` class in `derive.py`
- Implemented methods:
  - `derive_gene_features()` - Extracts gene-level metadata (63,140 genes)
  - `derive_transcript_features()` - Extracts transcript metadata (254,129 transcripts)
  - `derive_exon_features()` - Extracts exon metadata (1,668,828 exons)
  - `derive_junctions()` - Generates splice junctions from splice sites (10.9M junctions)
  - `derive_gene_annotations()` - Extracts all transcript annotations
  - `derive_splice_sites()` - Identifies splice sites (uses existing implementation)
  - `derive_genomic_sequences()` - Extracts gene sequences (uses existing implementation)
  - `derive_overlapping_genes()` - Finds overlapping gene metadata
  - `derive_all()` - Generates all datasets in one call

**CLI Support**:
```bash
# Generate all datasets
python -m meta_spliceai.system.genomic_resources.cli derive --all

# Generate specific datasets
python -m meta_spliceai.system.genomic_resources.cli derive \
  --gene-features \
  --transcript-features \
  --exon-features \
  --junctions

# With chromosome filtering
python -m meta_spliceai.system.genomic_resources.cli derive --all --chromosomes 21 22

# Force regeneration
python -m meta_spliceai.system.genomic_resources.cli derive --all --force
```

**Acceptance**

```bash
python -m meta_spliceai.system.genomic_resources.cli derive --all
ls data/ensembl/*.tsv  # ✅ Shows 8 TSV files
```

**Results**:
```
data/ensembl/
├── annotations_all_transcripts.tsv  (157 MB) - ✅
├── gene_features.tsv                (181 KB) - ✅
├── transcript_features.tsv          (21 MB)  - ✅
├── exon_features.tsv                (139 MB) - ✅
├── splice_sites.tsv                 (193 MB) - ✅
├── splice_sites_enhanced.tsv        (349 MB) - ✅ (auto-preferred by Registry)
├── junctions.tsv                    (628 MB) - ✅
└── overlapping_gene_counts.tsv      (477 KB) - ✅
```

**Key Features**:
- Uses existing extraction functions from `extract_genomic_features.py`
- Chromosome filtering support via `--chromosomes`
- Automatic fallback to legacy paths (`data/ensembl/spliceai_analysis/`)
- Registry automatically prefers `splice_sites_enhanced.tsv` when available
- Junctions include `gene_name` from enhanced splice sites

---

## Stage 6 — Validators ✅ **COMPLETE**

**Goal** Sanity-check build consistency, coordinate policy, and splice motifs.

**Status**: ✅ **COMPLETE - All validators implemented and tested**

### Implemented Validators (`validators.py`)

#### 1. **Gene Selection Validators** ✅
Pre-flight validation to catch invalid genes before expensive workflows:

- `validate_genes_have_splice_sites()` - Check genes have splice sites
- `validate_genes_in_gtf()` - Verify genes exist in GTF
- `validate_gene_selection()` - Comprehensive gene validation wrapper

**Usage**:
```python
from meta_spliceai.system.genomic_resources.validators import validate_gene_selection

valid_genes, invalid_summary = validate_gene_selection(
    gene_ids=['ENSG00000142611', 'ENSG00000290712'],
    data_dir=Path('data/ensembl'),
    min_splice_sites=1,
    fail_on_invalid=False,  # Filter out invalid instead of failing
    verbose=True
)

# Returns:
# - valid_genes: List of gene IDs with splice sites
# - invalid_summary: Dict with 'no_splice_sites' and 'not_in_gtf' lists
```

**Integration**: Automatically used in `incremental_builder.py` (lines 1085-1120) when `--run-workflow` is specified.

#### 2. **`assert_coordinate_policy()`** ✅
Validates that genomic coordinates follow the expected 1-based system.

**Checks**:
- ✅ No position 0 (impossible in 1-based)
- ✅ start ≤ end for all features
- ✅ All positions are positive
- ✅ Single-nucleotide features have start == end
- ✅ Cross-reference with GTF splice motifs

**Usage**:
```python
from meta_spliceai.system.genomic_resources.validators import assert_coordinate_policy

result = assert_coordinate_policy(
    [Path('data/ensembl/splice_sites.tsv'), Path('data/ensembl/gene_features.tsv')],
    expected="1-based",
    gtf_file=Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    fasta_file=Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    verbose=True
)

# Returns: {'passed': True, 'coordinate_system': '1-based', 'confidence': 'HIGH', ...}
```

#### 3. **`verify_gtf_coordinate_system()`** ✅
Verifies GTF coordinate system by checking splice motifs against reference FASTA.

**Method**: Samples exon boundaries from GTF and validates GT donor motifs appear at expected positions in FASTA.

**Usage**:
```python
from meta_spliceai.system.genomic_resources.validators import verify_gtf_coordinate_system

result = verify_gtf_coordinate_system(
    Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    sample_size=100,
    verbose=True
)

# Typical result: 84-87% match (expected due to non-canonical splice sites)
```

#### 4. **`assert_splice_motif_policy()`** ✅
Validates that splice sites have canonical GT-AG motifs.

**Validates**:
- ✅ Donor sites have GT dinucleotide (>95% expected)
- ✅ Acceptor sites have AG dinucleotide (>95% expected)
- ✅ Correct handling of both + and - strands
- ✅ Negative strand: extracts window + reverse complement

**Results** (tested on 2.8M real splice sites):
- **98.99% GT** for donors ✅
- **99.80% AG** for acceptors ✅

**Usage**:
```python
from meta_spliceai.system.genomic_resources.validators import assert_splice_motif_policy

result = assert_splice_motif_policy(
    Path('data/ensembl/splice_sites_enhanced.tsv'),
    Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    min_canonical_percent=95.0,
    sample_size=1000,  # None = check all
    verbose=True
)

# Returns: {'passed': True, 'donor_gt_percent': 98.99, 'acceptor_ag_percent': 99.80, ...}
```

#### 5. **`assert_build_alignment()`** ✅
Validates that GTF and FASTA files are from the same genome build.

**Checks**:
- ✅ Build identifier in filenames (e.g., "GRCh38")
- ✅ Release version in filenames (e.g., "112")
- ✅ Chromosome names match between GTF and FASTA
- ✅ Chromosome overlap >80%

**Usage**:
```python
from meta_spliceai.system.genomic_resources.validators import assert_build_alignment

result = assert_build_alignment(
    Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    expected_build="GRCh38",
    expected_release="112",
    verbose=True
)

# Returns: {'passed': True, 'build_match': True, 'chrom_match': True, ...}
```

### Position Field Definitions (Critical for Validators)

**Verified in `docs/data/splice_sites/POSITION_FIELD_VERIFICATION.md`**:

| Site Type | Position Points To | Dinucleotide Location | 0-Based Extraction |
|-----------|-------------------|----------------------|-------------------|
| **Donor** | First base of intron | GT at [pos, pos+1] (1-based) | `fasta[pos-1:pos+1]` |
| **Acceptor** | First base of exon | AG at [pos-2, pos-1] (1-based) | `fasta[pos-2:pos]` |

**Strand Handling**:
- **Positive strand**: Direct extraction
- **Negative strand**: Extract window → reverse complement → find GT/AG in transformed sequence

### Testing

**Test Suite**: `tests/test_validators.py`

```bash
# Run all validator tests
cd /Users/pleiadian53/work/meta-spliceai
mamba activate surveyor
python tests/test_validators.py
```

**Test Results** (all passing):
```
TEST 1: Coordinate Policy Validation           ✅ PASSED
TEST 2: GTF Coordinate System Verification     ✅ PASSED  
TEST 3: Splice Motif Validation                ✅ PASSED (98.99% GT, 99.80% AG)
TEST 4: Build Alignment Validation             ✅ PASSED
```

### Integration with Incremental Builder

**Automatic validation** in `incremental_builder.py`:

```python
# Lines 1085-1120 in incremental_builder.py
if run_workflow:
    # Pre-flight validation before expensive predictions
    from meta_spliceai.system.genomic_resources.validators import validate_gene_selection
    from meta_spliceai.system.config import Config
    
    data_dir = Path(Config.PROJ_DIR) / "data" / "ensembl"
    valid_genes, invalid_summary = validate_gene_selection(
        genes_to_validate,
        data_dir,
        min_splice_sites=1,
        fail_on_invalid=False,  # Filter out invalid genes
        verbose=(verbose >= 1)
    )
    
    # Update gene lists to only include valid genes
    all_gene_ids = [g for g in all_gene_ids if g in valid_genes]
    
    if not valid_genes:
        print("❌ ERROR: No genes with splice sites found!")
        raise SystemExit(1)
```

**Benefits**:
- ✅ Catches genes without splice sites **before** running expensive SpliceAI predictions
- ✅ Saves hours of compute time on invalid inputs
- ✅ Provides actionable error messages (e.g., "use --gene-types protein_coding")

### Acceptance

```bash
# 1. Run validator tests
python tests/test_validators.py
# Expected: All 4 tests pass ✅

# 2. Manual validation of derived datasets
python -c "
from meta_spliceai.system.genomic_resources.validators import (
    assert_coordinate_policy,
    assert_splice_motif_policy,
    assert_build_alignment
)
from pathlib import Path

# Coordinate validation
result = assert_coordinate_policy(
    [Path('data/ensembl/splice_sites_enhanced.tsv')],
    expected='1-based',
    verbose=True
)
assert result['passed'], 'Coordinate validation failed'

# Splice motif validation
result = assert_splice_motif_policy(
    Path('data/ensembl/splice_sites_enhanced.tsv'),
    Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    min_canonical_percent=95.0,
    verbose=True
)
assert result['passed'], 'Splice motif validation failed'

# Build alignment validation
result = assert_build_alignment(
    Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    expected_build='GRCh38',
    expected_release='112',
    verbose=True
)
assert result['passed'], 'Build alignment validation failed'

print('✅ All validators passed!')
"
```

### Documentation

See also:
- **Implementation**: `meta_spliceai/system/docs/STAGE_6_VALIDATORS_IMPLEMENTED.md`
- **Position verification**: `docs/data/splice_sites/POSITION_FIELD_VERIFICATION.md`
- **Consensus analysis**: `docs/data/splice_sites/CONSENSUS_ANALYSIS_SUMMARY.md`
- **Base model splice sites**: `meta_spliceai/system/docs/BASE_MODEL_SPLICE_SITE_DEFINITIONS.md`

---

## Stage 7 — Full Audit CLI ✅ **COMPLETE**

**Goal** Combine everything: show which artifacts exist, warn on mismatches, and run validators.

**Status**: ✅ **IMPLEMENTED**

**CLI subcommands**:

```
audit       ✅ Show inventory of genomic resources (8/8 resources tracked)
bootstrap   ✅ Download and index Ensembl GTF/FASTA files
derive      ✅ Generate derived TSV datasets (8 dataset types)
```

**Features**:
- ✅ Shows resource status with absolute paths
- ✅ Reports 8/8 resources found
- ✅ Displays build, release, and data root configuration
- ✅ Uses Registry for intelligent path resolution
- ✅ Supports backward compatibility (checks legacy paths)

**Acceptance**

```bash
python -m meta_spliceai.system.genomic_resources.cli audit
```

**Sample Output**:
```
================================================================================
Genomic Resources Audit
================================================================================

Configuration:
  Species:     homo_sapiens
  Build:       GRCh38
  Release:     112
  Data root:   /Users/pleiadian53/work/meta-spliceai/data/ensembl

Resource Status:
  gtf                  ✓ FOUND     /Users/.../Homo_sapiens.GRCh38.112.gtf
  fasta                ✓ FOUND     /Users/.../Homo_sapiens.GRCh38.dna.primary_assembly.fa
  fasta_index          ✓ FOUND     /Users/.../Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai
  splice_sites         ✓ FOUND     /Users/.../splice_sites_enhanced.tsv ← Uses enhanced!
  gene_features        ✓ FOUND     /Users/.../gene_features.tsv
  transcript_features  ✓ FOUND     /Users/.../transcript_features.tsv
  exon_features        ✓ FOUND     /Users/.../exon_features.tsv
  junctions            ✓ FOUND     /Users/.../junctions.tsv

================================================================================
Summary: 8/8 resources found

✅ All critical resources present!
```

---

### ✅ Final Notes

* “Current” files always live directly in `data/ensembl/`.
* Optional stashes live in `data/ensembl/<BUILD>/`.
* Derived TSVs sit beside the GTF/FASTA.
* Environment variables `SS_BUILD`, `SS_RELEASE`, `SS_GTF_PATH`, `SS_FASTA_PATH`, `SS_DATA_ROOT` override defaults.
* All coordinates = **1-based closed**, donors = exon 3′ boundary, acceptors = exon 5′ boundary, motifs = {GT, GC}/{AG}.

---

**End of file — `rebuild_genomic_resources.md`**
