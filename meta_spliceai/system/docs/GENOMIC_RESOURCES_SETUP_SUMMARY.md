# Genomic Resources Setup Summary

## Overview
Successfully established the genomic resources management system for MetaSpliceAI on MacBook, including downloading essential Ensembl files and verifying the complete workflow.

## Completed Tasks

### 1. Environment Setup ✅
- Created and activated `surveyor` mamba environment
- Installed all required dependencies via `environment.yml`
- Added missing packages: `pyyaml`, `biopython`, `tabulate`, `pyspark`, `shap`, `pyBigWig`, `seaborn`, `pyarrow`, `psutil`

### 2. Path Configuration Fixes ✅
- Fixed hardcoded paths in:
  - `meta_spliceai/system/config.ini` → Updated `data_prefix` to local path
  - `meta_spliceai/splice_engine/analyzer.py` → Dynamic path resolution via SystemConfig
  - `meta_spliceai/splice_engine/meta_models/core/analyzer.py` → Uses `create_systematic_manager()`
  - `meta_spliceai/splice_engine/extract_genomic_features.py` → Uses `Analyzer.gtf_file`

### 3. Genomic Resources Package ✅
Created `meta_spliceai/system/genomic_resources/` with:

#### `config.py`
- YAML-based configuration (`configs/genomic_resources.yaml`)
- Environment variable overrides (SS_SPECIES, SS_BUILD, SS_RELEASE, SS_DATA_ROOT)
- Dynamic project root resolution via `SystemConfig.PROJ_DIR`

#### `registry.py`
- Unified path resolution for GTF, FASTA, and derived datasets
- Search order: explicit env vars → `data/ensembl/` → `data/ensembl/<BUILD>/` → `data/ensembl/spliceai_analysis/`
- Methods: `resolve()`, `get_gtf_path()`, `get_fasta_path()`, `list_all()`

#### `download.py`
- Ensembl file fetching with checksum verification
- Automatic decompression of .gz files
- FASTA indexing via `pyfaidx`
- Function: `fetch_ensembl(kind, build, release, output_dir)`

#### `cli.py`
- Command-line interface with subcommands:
  - `audit` - Show inventory of genomic resources
  - `bootstrap` - Download missing GTF/FASTA files

#### `__init__.py`
- Exports: `Config`, `load_config`, `filename`, `Registry`, `create_systematic_manager`

### 4. Downloaded Genomic Data ✅
Successfully downloaded from Ensembl release 112:
- **GTF**: `Homo_sapiens.GRCh38.112.gtf` (1.5 GB)
- **FASTA**: `Homo_sapiens.GRCh38.dna.primary_assembly.fa` (3.0 GB)
- **FASTA Index**: `.fai` file generated

### 5. SpliceAI Models ✅
- Located pre-trained models in `spliceai` package installation
- Created symlinks: `data/models/spliceai/spliceai{1-5}.h5`

### 6. Pre-flight Validation ✅
Implemented early validation to catch invalid genes before expensive workflows:
- Created `meta_spliceai/system/genomic_resources/validators.py`
- Validates genes have splice sites before running predictions
- Integrated into `incremental_builder.py` via `--run-workflow`
- Provides helpful error messages and filtering options

### 7. Workflow Verification ✅
Successfully tested `incremental_builder.py --run-workflow`:
- ✅ SpliceAI predictions running correctly
- ✅ Splice site detection working (228 positions analyzed for 2 genes)
- ✅ Training dataset generation: 175 rows × 4,155 features
- ✅ Output files created in proper format
- ✅ Pre-flight validation catches genes without splice sites early

## Current System State

### Available Resources
```
data/ensembl/
├── Homo_sapiens.GRCh38.112.gtf (1.5 GB)
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa (3.0 GB)
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai
├── splice_sites.tsv (2.8M splice sites)
├── annotations_all_transcripts.tsv
├── overlapping_gene_counts.tsv
├── gene_sequence_*.parquet (24 chromosome files)
└── spliceai_analysis/
    ├── gene_features.tsv (63,140 genes)
    ├── transcript_features.tsv (254,129 transcripts)
    └── exon_features.tsv

data/models/spliceai/
├── spliceai1.h5 → (symlink to package)
├── spliceai2.h5 → (symlink to package)
├── spliceai3.h5 → (symlink to package)
├── spliceai4.h5 → (symlink to package)
└── spliceai5.h5 → (symlink to package)
```

### Configuration
- **Project Root**: `/Users/pleiadian53/work/meta-spliceai`
- **Data Root**: `/Users/pleiadian53/work/meta-spliceai/data/ensembl`
- **Build**: GRCh38
- **Release**: 112
- **Species**: homo_sapiens

## Usage Examples

### Audit Resources
```bash
python -m meta_spliceai.system.genomic_resources.cli audit
```

### Download Missing Files
```bash
python -m meta_spliceai.system.genomic_resources.cli bootstrap --species homo_sapiens --build GRCh38 --release 112
```

### Run Incremental Builder
```bash
# With gene type filtering (recommended for testing)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --n-genes 10 \
  --subset-policy random \
  --gene-types protein_coding lncRNA \
  --output-dir output/ \
  --run-workflow \
  --overwrite \
  -v

# With custom gene list
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --subset-policy custom \
  --gene-ids-file tests/test_genes.txt \
  --output-dir output/ \
  --run-workflow \
  --overwrite \
  -v
```

## Key Learnings

1. **Gene Selection and Filtering**: 
   - Use `--gene-types` to filter by gene type (e.g., `protein_coding`, `lncRNA`)
   - **19,087 protein_coding genes** (95%) have splice sites
   - **16,055 lncRNA genes** (83%) have splice sites
   - Random selection without filtering may pick genes without splice sites (e.g., some pseudogenes, single-exon genes)
   - For testing, use `--gene-types protein_coding lncRNA` to ensure genes with splice sites

2. **Path Management**: The system now uses centralized path management via `genomic_resources.Registry`, eliminating hardcoded paths.

3. **Workflow Dependencies**: The `--run-workflow` flag triggers `splice_prediction_workflow.py` which generates essential artifacts (splice positions, errors, sequences) needed for meta-model training.

4. **Model Integration**: MetaSpliceAI is designed to work with any splice site predictor that produces per-nucleotide donor/acceptor/neither scores, making it extensible to models beyond SpliceAI (e.g., OpenSpliceAI).

## Next Steps (Pending)

1. **Refactor Data Preparation Functions** (Stage 5)
   - Move `prepare_gene_annotations()` to `genomic_resources/derive.py`
   - Move `prepare_splice_site_annotations()` to `genomic_resources/derive.py`
   - Move `prepare_genomic_sequences()` to `genomic_resources/derive.py`
   - Move `handle_overlapping_genes()` to `genomic_resources/derive.py`

2. **Implement Validators** (Stage 6)
   - GTF/FASTA format validation
   - Derived dataset sanity checks
   - Cross-reference validation (genes in GTF vs. sequences)

3. **Complete CLI** (Stage 7)
   - Add `derive` subcommand for generating TSV datasets
   - Add `index` subcommand for FASTA indexing
   - Add `set-current` helper for build switching

## Testing

### Minimal Check
```bash
python scripts/minimal_check.py
```

### Test with Known Genes
Create `tests/test_genes.txt` with genes that have splice sites (e.g., ENSG00000198042, ENSG00000172497).

## Troubleshooting

### Issue: "No donor/acceptor annotations for gene"
**Cause**: Selected genes don't have splice sites in GTF  
**Solution**: Use `--gene-types protein_coding lncRNA` for gene type filtering  
**Prevention**: Pre-flight validation now catches this early (see validation output)

### Issue: "FileNotFoundError: GTF file not found"
**Cause**: GTF file not downloaded or path misconfigured  
**Solution**: Run `bootstrap` command or check `configs/genomic_resources.yaml`

### Issue: "SpliceAI models not found"
**Cause**: Models not in expected location  
**Solution**: Create symlinks from package installation to `data/models/spliceai/`

## API Reference

### Core Classes

#### **Registry**
Unified path resolution for all genomic resources.

```python
from meta_spliceai.system.genomic_resources import Registry

# Create registry for specific build/release
registry = Registry(build="GRCh38", release="112")

# Resolve paths (returns absolute path or None)
gtf_path = registry.resolve("gtf")
fasta_path = registry.resolve("fasta")
splice_sites = registry.resolve("splice_sites")  # Auto-prefers enhanced version!

# Get validated paths (raises FileNotFoundError if missing)
gtf_path = registry.get_gtf_path(validate=True)
fasta_path = registry.get_fasta_path(validate=True)
```

**Search Order**:
1. Environment variable override (e.g., `SS_GTF_PATH`)
2. `data/ensembl/` (preferred location)
3. `data/ensembl/<BUILD>/` (build-specific stash)
4. `data/ensembl/spliceai_analysis/` (legacy location)

**Auto-Selection**: Registry automatically prefers `splice_sites_enhanced.tsv` when resolving `splice_sites`.

#### **GenomicDataDeriver**
Generate derived datasets from GTF/FASTA.

```python
from meta_spliceai.system.genomic_resources import GenomicDataDeriver

# Create deriver
deriver = GenomicDataDeriver(verbosity=1)

# Derive all datasets
result = deriver.derive_all(
    consensus_window=2,
    target_chromosomes=['21', '22'],  # Optional: specific chromosomes
    force_overwrite=False
)

# Or derive specific datasets
result = deriver.derive_gene_features(force_overwrite=True)
result = deriver.derive_splice_sites(consensus_window=2)
result = deriver.derive_junctions()

# Each returns:
# {'success': bool, '<dataset>_file': str, '<dataset>_df': pl.DataFrame, 'error': str}
```

**Available Methods**:
- `derive_gene_features()` - Gene-level metadata
- `derive_transcript_features()` - Transcript metadata
- `derive_exon_features()` - Exon metadata
- `derive_junctions()` - Splice junctions (with gene_name)
- `derive_gene_annotations()` - All transcript annotations
- `derive_splice_sites()` - Splice site positions
- `derive_genomic_sequences()` - Gene sequences
- `derive_overlapping_genes()` - Overlapping gene metadata
- `derive_all()` - Generate all datasets

### Validation Functions

#### **validate_gene_selection()**
Pre-flight validation for gene selection.

```python
from meta_spliceai.system.genomic_resources.validators import validate_gene_selection
from pathlib import Path

valid_genes, invalid_summary = validate_gene_selection(
    gene_ids=['ENSG00000142611', 'ENSG00000290712', 'ENSG00000235993'],
    data_dir=Path('data/ensembl'),
    min_splice_sites=1,
    check_gtf_presence=False,
    fail_on_invalid=False,  # Filter out invalid genes instead of raising error
    verbose=True
)

# Returns:
# - valid_genes: List[str] - genes with splice sites
# - invalid_summary: Dict[str, List[str]] - {'no_splice_sites': [...], 'not_in_gtf': [...]}
```

**Used in**: `incremental_builder.py` (lines 1085-1120) for pre-flight checks before workflow.

#### **assert_coordinate_policy()**
Validate 1-based coordinate system.

```python
from meta_spliceai.system.genomic_resources.validators import assert_coordinate_policy

result = assert_coordinate_policy(
    [Path('data/ensembl/splice_sites.tsv'), Path('data/ensembl/gene_features.tsv')],
    expected="1-based",
    gtf_file=Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    fasta_file=Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    verbose=True
)

# Returns: {'passed': bool, 'coordinate_system': str, 'confidence': str, 'checks': [...], ...}
assert result['passed']
```

#### **assert_splice_motif_policy()**
Validate GT-AG canonical splice motifs.

```python
from meta_spliceai.system.genomic_resources.validators import assert_splice_motif_policy

result = assert_splice_motif_policy(
    Path('data/ensembl/splice_sites_enhanced.tsv'),
    Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    min_canonical_percent=95.0,
    sample_size=1000,  # None = check all 2.8M sites
    verbose=True
)

# Returns: {'passed': bool, 'donor_gt_percent': float, 'acceptor_ag_percent': float, ...}
# Typical results: 98.99% GT for donors, 99.80% AG for acceptors
```

#### **assert_build_alignment()**
Validate GTF/FASTA build consistency.

```python
from meta_spliceai.system.genomic_resources.validators import assert_build_alignment

result = assert_build_alignment(
    Path('data/ensembl/Homo_sapiens.GRCh38.112.gtf'),
    Path('data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa'),
    expected_build="GRCh38",
    expected_release="112",
    verbose=True
)

# Returns: {'passed': bool, 'build_match': bool, 'chrom_match': bool, ...}
```

### Derived Dataset Schemas

#### **gene_features.tsv** (181 KB, 63,140 genes)
```
Columns: gene_id, gene_name, gene_type, chrom, strand, start, end, gene_length
Source: extract_gene_features_from_gtf() from extract_genomic_features.py
Usage: Gene-level metadata, filtering by gene type, length statistics
```

#### **transcript_features.tsv** (21 MB, 254,129 transcripts)
```
Columns: transcript_id, gene_id, chrom, strand, start, end, exon_count, 
         transcript_length, cds_length, transcript_biotype
Source: extract_transcript_features_from_gtf() from extract_genomic_features.py
Usage: Transcript isoform analysis, exon counting, CDS analysis
```

#### **exon_features.tsv** (139 MB, 1,668,828 exons)
```
Columns: exon_id, transcript_id, gene_id, chrom, strand, start, end, 
         exon_number, exon_rank, exon_length
Source: extract_exon_features_from_gtf() from extract_genomic_features.py
Usage: Exon boundary analysis, alternative splicing, exon size distribution
```

#### **splice_sites.tsv** (193 MB, 2,829,398 sites)
```
Columns: chrom, start, end, position, strand, site_type, gene_id, transcript_id
Source: prepare_splice_site_annotations() from data_preparation.py
Usage: Basic splice site locations for donor/acceptor prediction
Note: Superseded by splice_sites_enhanced.tsv for most uses
```

#### **splice_sites_enhanced.tsv** (349 MB, 2,829,398 sites) ✨
```
Columns: chrom, start, end, position, strand, site_type, gene_id, gene_name,
         transcript_id, exon_id, exon_number, exon_rank, 
         gene_biotype, transcript_biotype
Source: Enhanced version with additional metadata
Usage: Preferred by Registry; includes gene names and biotype information
Benefits: Enables gene name lookups, biotype filtering, exon tracking
```

#### **junctions.tsv** (628 MB, 10,881,854 junctions)
```
Columns: chrom, donor_pos, acceptor_pos, strand, gene_id, gene_name, 
         transcript_id, intron_length
Source: Derived from splice_sites_enhanced.tsv (pairs donors with acceptors)
Usage: Intron analysis, junction spanning reads, alternative splicing events
Note: Includes gene_name from enhanced splice sites!
```

#### **annotations_all_transcripts.tsv** (157 MB)
```
Columns: Gene-level and transcript-level annotations
Source: prepare_gene_annotations() from data_preparation.py
Usage: Complete gene and transcript metadata
```

#### **overlapping_gene_counts.tsv** (477 KB)
```
Columns: gene_id, overlapping_gene_count, overlapping_gene_ids
Source: handle_overlapping_genes() from data_preparation.py
Usage: Identify genes with overlapping genomic regions
```

### Configuration

#### **Environment Variables**
Override default configuration:

```bash
export SS_SPECIES=homo_sapiens
export SS_BUILD=GRCh38
export SS_RELEASE=112
export SS_DATA_ROOT=/custom/path/to/data
export SS_GTF_PATH=/custom/path/to/annotation.gtf
export SS_FASTA_PATH=/custom/path/to/genome.fa
```

#### **Configuration File**
`configs/genomic_resources.yaml`:

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
```

### Integration Examples

#### **Example 1: Incremental Builder Integration**
```python
# Pre-flight validation before workflow
from meta_spliceai.system.genomic_resources.validators import validate_gene_selection
from meta_spliceai.system.config import Config

data_dir = Path(Config.PROJ_DIR) / "data" / "ensembl"
valid_genes, invalid_summary = validate_gene_selection(
    gene_ids,
    data_dir,
    min_splice_sites=1,
    fail_on_invalid=False,
    verbose=True
)

# Filter to valid genes only
gene_ids = [g for g in gene_ids if g in valid_genes]

if not valid_genes:
    print("❌ No genes with splice sites found!")
    raise SystemExit(1)
```

#### **Example 2: Path Resolution for Analysis**
```python
# Get paths using Registry
from meta_spliceai.system.genomic_resources import Registry

registry = Registry()
splice_sites_path = registry.resolve('splice_sites')  # Gets enhanced version
gene_features_path = registry.resolve('gene_features')

# Load data
import polars as pl
splice_sites = pl.read_csv(splice_sites_path, separator='\t', schema_overrides={'chrom': pl.Utf8})
gene_features = pl.read_csv(gene_features_path, separator='\t', schema_overrides={'chrom': pl.Utf8})
```

#### **Example 3: Generate Missing Datasets**
```python
# Check and regenerate missing datasets
from meta_spliceai.system.genomic_resources import GenomicDataDeriver

deriver = GenomicDataDeriver(verbosity=1)

# Generate only if missing
result = deriver.derive_gene_features(force_overwrite=False)
if result['success']:
    print(f"✅ Gene features: {result['gene_features_file']}")

# Or regenerate all datasets
results = deriver.derive_all(force_overwrite=True)
```

## References

- Ensembl FTP: https://ftp.ensembl.org/pub/
- SpliceAI Paper: Jaganathan et al. (2019) Cell
- Project Documentation: `docs/`, `tutorial/`
- Stage 6 Validators: `meta_spliceai/system/docs/STAGE_6_VALIDATORS_IMPLEMENTED.md`
- Position Field Verification: `docs/data/splice_sites/POSITION_FIELD_VERIFICATION.md`
