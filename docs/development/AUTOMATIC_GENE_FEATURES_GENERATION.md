# Automatic Gene Features Generation in Base Model Workflow

## Overview

The base model pass workflow (`splice_prediction_workflow.py`) now automatically generates build-specific gene features during the data preparation phase. This eliminates the need for separate manual scripts to generate `gene_features.tsv`.

## What Changed

### 1. Integrated Gene Features Derivation

**Location**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py` (lines 218-263)

The workflow now includes a new step (1.5) that automatically derives gene features:

```python
# 1.5. Derive gene features (gene_type, gene_length, etc.)
# This is needed for biotype-specific gene sampling and analysis
try:
    from meta_spliceai.system.genomic_resources.derive import GenomicDataDeriver
    from meta_spliceai.system.genomic_resources import Registry
    
    # Create deriver using the local_dir (build-specific directory)
    # Infer build and release from the GTF file path if possible
    registry = None
    if 'GRCh37' in gtf_file:
        registry = Registry(build='GRCh37', release='87')
    elif 'GRCh38' in gtf_file:
        # Try to infer release from filename
        import re
        match = re.search(r'\.(\d+)\.gtf', gtf_file)
        release = match.group(1) if match else '112'
        registry = Registry(build='GRCh38', release=release)
    
    deriver = GenomicDataDeriver(
        data_dir=local_dir,
        registry=registry,
        verbosity=verbosity
    )
    
    # Derive gene features if not already present
    gene_features_result = deriver.derive_gene_features(
        output_filename='gene_features.tsv',
        target_chromosomes=target_chromosomes,
        force_overwrite=False  # Use existing if available
    )
    
    if gene_features_result['success'] and verbosity >= 1:
        print_with_indent(
            f"[info] Gene features available: {gene_features_result['gene_features_file']}",
            indent_level=1
        )
except Exception as e:
    if verbosity >= 1:
        print_with_indent(
            f"[warning] Could not derive gene features: {e}",
            indent_level=1
        )
        print_with_indent(
            "[warning] Gene biotype information may not be available for downstream analysis",
            indent_level=1
        )
```

### 2. Build-Specific Storage

Gene features are now stored in build-specific directories:
- `data/ensembl/GRCh37/gene_features.tsv` (for GRCh37)
- `data/ensembl/GRCh38/gene_features.tsv` (for GRCh38)
- `data/mane/GRCh38/gene_features.tsv` (for MANE RefSeq/OpenSpliceAI)

### 3. Smart Caching

The workflow checks if `gene_features.tsv` already exists before regenerating:
- If it exists: Uses the existing file (fast)
- If it doesn't exist: Generates it from the GTF (slower, but only happens once)
- If `force_overwrite=True`: Always regenerates (useful for updates)

## Benefits

### 1. No Manual Intervention Required

Users no longer need to run separate scripts like:
```bash
python scripts/setup/generate_grch37_gene_features.py
```

The workflow handles this automatically.

### 2. Guaranteed Consistency

Gene features are always derived from the same GTF file used for the rest of the workflow, ensuring consistency.

### 3. Build-Specific by Design

Each genome build gets its own `gene_features.tsv`, preventing cross-build contamination.

### 4. Biotype-Specific Analysis

The generated `gene_features.tsv` includes the `gene_type` column, enabling:
- Biotype-specific gene sampling (protein_coding, lncRNA, etc.)
- Gene filtering by type
- Gene metadata lookup

## Gene Features Contents

The `gene_features.tsv` file contains:

| Column | Description |
|--------|-------------|
| `gene_id` | Ensembl gene ID |
| `gene_name` | Gene symbol |
| `gene_type` | Gene biotype (protein_coding, lincRNA, pseudogene, etc.) |
| `gene_length` | Gene length in base pairs |
| `chrom` | Chromosome |
| `start`, `end` | Genomic coordinates |
| `strand` | +/- |
| `seqname` | Sequence name (chromosome) |
| Additional GTF attributes | Source, feature, score, frame, etc. |

### Example Statistics (GRCh37)

```
Total genes: 57,905
Unique chromosomes: 61

Gene type distribution (top 10):
  protein_coding                : 20,356 genes
  pseudogene                    : 13,940 genes
  lincRNA                       :  7,109 genes
  antisense                     :  5,273 genes
  miRNA                         :  3,111 genes
  misc_RNA                      :  2,038 genes
  snRNA                         :  1,923 genes
  snoRNA                        :  1,459 genes
  sense_intronic                :    741 genes
  rRNA                          :    533 genes
```

## Workflow Integration

### Data Preparation Steps

The base model workflow now has these data preparation steps:

1. **Prepare gene annotations** (`prepare_gene_annotations`)
   - Extracts transcript-level annotations from GTF
   - Creates `annotations_all_transcripts.tsv` and `annotations.db`

2. **Derive gene features** (`GenomicDataDeriver.derive_gene_features`) ← **NEW**
   - Extracts gene-level metadata from GTF
   - Creates `gene_features.tsv` with biotype information

3. **Prepare splice site annotations** (`prepare_splice_site_annotations`)
   - Extracts splice sites from GTF
   - Creates `splice_sites.tsv` or uses `splice_sites_enhanced.tsv`

4. **Prepare genomic sequences** (`prepare_genomic_sequences`)
   - Extracts gene sequences from FASTA
   - Creates sequence files for prediction

5. **Handle overlapping genes** (`handle_overlapping_genes`)
   - Identifies genes with overlapping coordinates

6. **Determine target chromosomes** (`determine_target_chromosomes`)
   - Decides which chromosomes to process

7. **Load SpliceAI models** (`load_spliceai_models`)
   - Loads pre-trained SpliceAI models

8. **Prepare splice site adjustments** (`prepare_splice_site_adjustments`)
   - Detects coordinate alignment adjustments if enabled

### When Gene Features Are Generated

Gene features are generated during the **first run** of the base model workflow for a given build:

```bash
# First run for GRCh37
python scripts/setup/run_grch37_full_workflow.py
# → Generates data/ensembl/GRCh37/gene_features.tsv

# Subsequent runs
python scripts/setup/run_grch37_full_workflow.py
# → Uses existing data/ensembl/GRCh37/gene_features.tsv (fast!)
```

## Manual Generation (If Needed)

In rare cases where you need to regenerate `gene_features.tsv` manually:

```bash
# For GRCh37
python scripts/setup/generate_grch37_gene_features.py

# Or using GenomicDataDeriver directly
python -c "
from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.system.genomic_resources.derive import GenomicDataDeriver

registry = Registry(build='GRCh37', release='87')
deriver = GenomicDataDeriver(
    data_dir=registry.data_dir,
    registry=registry,
    verbosity=2
)
result = deriver.derive_gene_features(
    output_filename='gene_features.tsv',
    force_overwrite=True
)
print(f'Generated: {result[\"gene_features_file\"]}')
"
```

## Error Handling

The workflow gracefully handles errors during gene features derivation:

- **If derivation fails**: Logs a warning but continues with the workflow
- **Impact**: Biotype-specific gene sampling may not be available
- **Fallback**: Downstream scripts should check for `gene_features.tsv` existence

## Testing

The comprehensive base model test (`scripts/testing/test_base_model_comprehensive.py`) now:

1. **Expects** `gene_features.tsv` to exist in the build-specific directory
2. **Fails fast** with a clear error message if it's missing
3. **Uses** `gene_features.tsv` for biotype-specific gene sampling

This ensures that any issues with gene features generation are caught early.

## Related Documentation

- [Build-Specific Datasets](../data/BUILD_SPECIFIC_DATASETS.md)
- [Annotation Source Directory Structure](ANNOTATION_SOURCE_DIRECTORY_STRUCTURE.md)
- [Genome Build Compatibility](../base_models/GENOME_BUILD_COMPATIBILITY.md)

## Implementation Details

### GenomicDataDeriver Class

The `GenomicDataDeriver` class (`meta_spliceai/system/genomic_resources/derive.py`) provides a systematic interface for deriving genomic datasets:

```python
class GenomicDataDeriver:
    """Derives genomic datasets from GTF and FASTA files."""
    
    def derive_gene_features(
        self,
        output_filename: str = "gene_features.tsv",
        target_chromosomes: Optional[List[str]] = None,
        force_overwrite: bool = False
    ) -> Dict[str, Any]:
        """Derive gene-level features from GTF."""
        # Implementation uses extract_gene_features_from_gtf
        # from meta_spliceai.splice_engine.extract_genomic_features
```

### Build and Release Inference

The workflow infers the genome build and release from the GTF file path:

- **GRCh37**: Looks for 'GRCh37' in the path, uses release '87'
- **GRCh38**: Looks for 'GRCh38' in the path, extracts release from filename (e.g., `Homo_sapiens.GRCh38.112.gtf` → release '112')

This automatic inference ensures the correct `Registry` is used for path resolution.

## Future Enhancements

1. **Parallel Extraction**: Speed up gene features extraction for large GTF files
2. **Incremental Updates**: Only update changed genes when GTF is updated
3. **Additional Features**: Extract more gene-level metadata (e.g., exon counts, transcript counts)
4. **Validation**: Add checksums to verify data integrity

